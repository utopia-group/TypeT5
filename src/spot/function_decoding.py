import asyncio
import copy
import random

import torch

from .data import CtxArgs, src_to_chunks
from .function_dataset import (
    SignatureMap,
    ctx_modules_for_elem,
    mk_preamble,
    reformat_elems,
    wrap_main_code,
)
from .model import DatasetPredResult, ModelWrapper
from .static_analysis import (
    ElemSignature,
    FunctionSignature,
    ModuleName,
    ModulePath,
    ProjectPath,
    PythonElem,
    PythonFunction,
    PythonProject,
    PythonVariable,
    SignatureErrorAnalysis,
    UsageAnalysis,
    VariableSignature,
)
from .tokenized_src import PreprocessArgs, TokenSeq, tokenized_src_from_segs
from .type_check import PythonType, parse_type_str
from .type_env import (
    AccuracyMetric,
    AnnotPath,
    collect_user_annotations,
    type_accuracies,
)
from .utils import *


@dataclass
class RolloutPrediction:
    assignments: SignatureMap
    elem2preds: dict[ProjectPath, Sequence[PythonType]]
    elem2inputs: dict[ProjectPath, dict]


@dataclass
class EvalResult:
    project_roots: list[Path]
    predictions: list[RolloutPrediction]
    label_maps: list[SignatureMap]

    def error_analysis(
        self,
        project: int | str | None,
        metric: AccuracyMetric,
    ) -> SignatureErrorAnalysis:
        projects = self.find_projects(project)

        label_maps = dict[str, SignatureMap]()
        pred_maps = dict[str, SignatureMap]()

        for i in projects:
            pname = self.project_roots[i].name
            label_maps[pname] = self.label_maps[i]
            pred_maps[pname] = self.predictions[i].assignments

        return SignatureErrorAnalysis(
            pred_maps,
            label_maps,
            metric,
            error_on_mismatched_signature=True,  # there shouldn't be any mismatch
        )

    def find_projects(self, identifier: int | str | None) -> list[int]:
        if isinstance(identifier, int):
            projects = [identifier]
        elif isinstance(identifier, str):
            projects = [
                i for i, p in enumerate(self.project_roots) if p.name == identifier
            ]
            assert projects, f"Project not found: {identifier}"
        else:
            projects = list(range(len(self.predictions)))
        return projects

    def inspect_elem(self, identifier: int | str | None, path: ProjectPath) -> None:
        pid = self.find_projects(identifier)[0]

        print("Expected: ", self.label_maps[pid][path])
        print("Predicted:", self.predictions[pid].assignments[path])
        print("Code:")
        print(decode_tokens(self.predictions[pid].elem2inputs[path]["input_ids"]))


@dataclass
class RolloutCtx:
    model: ModelWrapper

    async def evaluate_on_projects(
        self,
        projects: Sequence[PythonProject],
        pre_args: PreprocessArgs,
        decode_order: "DecodingOrder",
        concurrency: int = DefaultWorkers,
        tqdm_args: dict = {},
    ) -> EvalResult:
        """Evaluate the model's prediction accuracy on a given set of projects, masking
        any existing type annotations and treating them as the ground truth.
        Note that the model does make predictions for those places with a missing type
        annotation, but they are not counted in the accuracy computation (and only serve as
        an intermediate for information propogation).
        """
        n_total_elems = sum(1 for p in projects for e in p.all_elems())
        project_roots = [p.root_dir for p in projects]
        rollouts: list[Any] = [None for _ in projects]
        label_maps: list[Any] = [dict() for _ in projects]
        with tqdm(
            total=decode_order.traverse_length(n_total_elems),
            desc="evaluate_on_projects",
            smoothing=0.01,
            **tqdm_args,
        ) as pbar, ThreadPoolExecutor(1) as model_executor, ProcessPoolExecutor(
            concurrency
        ) as cpu_executor:

            async def eval_project(id_project: tuple[int, PythonProject]):
                id, project = id_project
                label_sigs = {e.path: e.get_signature() for e in project.all_elems()}
                label_maps[id] = label_sigs
                r = await self.project_rollout(
                    project.mask_types(),
                    pre_args,
                    decode_order,
                    cpu_executor=cpu_executor,
                    model_executor=model_executor,
                    progress_cbk=lambda e, p: pbar.update(),
                )
                rollouts[id] = r

            await throttled_async_run(
                eval_project, list(enumerate(projects)), concurrency=concurrency
            )

        return EvalResult(project_roots, rollouts, label_maps)

    async def project_rollout(
        self,
        project: PythonProject,
        pre_args: PreprocessArgs,
        decode_order: "DecodingOrder",
        cpu_executor: ProcessPoolExecutor,
        model_executor: ThreadPoolExecutor,
        progress_cbk: Callable[
            [PythonElem, Sequence[PythonType]], Any
        ] = lambda x, y: None,
    ) -> RolloutPrediction:
        """Note: when evaluating on dataset with ground truth labels, we need to
        first replace all labels with `SpecialNames.TypeMask` before feeding to
        this function.
        """
        # Model executor needs to be single threaded.
        assert_eq(model_executor._max_workers, 1)

        eloop = asyncio.get_event_loop()
        analysis: UsageAnalysis = await eloop.run_in_executor(
            cpu_executor,
            UsageAnalysis,
            project,
            pre_args.add_override_usages,
            pre_args.add_implicit_rel_imports,
        )
        to_visit = [analysis.path2elem[p] for p in decode_order.traverse(analysis)]
        visit_set = {e.path for e in to_visit}
        for e in project.all_elems():
            assert e.path in visit_set, f"{e.path} not in the decoder order."
        preamble_cache = dict[ModuleName, tuple[str, TokenSeq]]()

        assignments = SignatureMap()
        elem2preds = dict[ProjectPath, Sequence[PythonType]]()
        elem2inputs = dict[ProjectPath, dict]()
        mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))

        # Parallelize computation between dependency-free elements
        for elem in to_visit:
            # construct input for the model
            # first, create or retrieve the preamble
            cur_module = elem.path.module
            if cur_module not in preamble_cache:
                preamble_tuple = await eloop.run_in_executor(
                    cpu_executor,
                    mk_preamble,
                    project.modules[cur_module].tree,
                    pre_args,
                )
                preamble_cache[cur_module] = preamble_tuple
            preamble, tokenized_preamble = preamble_cache[cur_module]

            # then, make all missing types in the signature a prediction target
            if isinstance(elem, PythonVariable):
                sig = elem.get_signature()
                elem_sig = copy.deepcopy(sig)
                if sig.annot is None:
                    elem_sig.annot = mask_annot
                elem_map = {elem.path: elem_sig}
            elif isinstance(elem, PythonFunction):
                sig = elem.get_signature()
                elem_sig = copy.deepcopy(sig)
                for n, a in elem_sig.params.items():
                    if a is None:
                        elem_sig.params[n] = mask_annot
                if elem_sig.returns is None:
                    elem_sig.returns = mask_annot
                elem_map = {elem.path: elem_sig}
            else:
                raise NotImplemented(f"Unsupported element type {type(elem)}")
            main_lines = reformat_elems(
                [elem],
                analysis.path2class,
                cast(SignatureMap, elem_map),
                keep_body_types=True,
            )

            left_m, right_m = ctx_modules_for_elem(
                elem,
                analysis,
                pre_args,
                assignments if decode_order.types_in_ctx() else {},
            )

            model_inputs = await eloop.run_in_executor(
                cpu_executor,
                construct_model_inputs,
                cst.Module(main_lines),
                left_m,
                right_m,
                preamble,
                tokenized_preamble,
                self.model.args.ctx_args,
            )

            pred_types = list[PythonType]()
            if model_inputs:
                for chunk in model_inputs:
                    chunk = {
                        "input_ids": torch.tensor([chunk["input_ids"]]),
                        "labels": torch.tensor([chunk["labels"]]),
                        "n_labels": torch.tensor([chunk["n_labels"]]),
                    }
                    preds, _ = await eloop.run_in_executor(
                        model_executor, self.model.predict_on_batch, chunk
                    )
                    pred_types.extend(preds[0])
                elem2inputs[elem.path] = model_inputs[0]

                # update the signature with the predicted types
                sig = copy.deepcopy(sig)
                if isinstance(sig, VariableSignature):
                    assert sig.annot is None or is_mask_annot(
                        sig.annot
                    ), f"For {elem}, sig={sig}"
                    assert_eq(len(pred_types), 1)
                    sig.annot = cst.Annotation(cst.parse_expression(str(pred_types[0])))
                elif isinstance(elem, PythonFunction):
                    assert len(pred_types) >= len(sig.params) + 1
                    for i, (n, a) in enumerate(sig.params.items()):
                        if a is None or is_mask_annot(a):
                            new_type = cst.parse_expression(str(pred_types[i]))
                            sig.params[n] = cst.Annotation(new_type)
                    if sig.returns is None or is_mask_annot(sig.returns):
                        sig.returns = cst.Annotation(
                            cst.parse_expression(str(pred_types[len(sig.params)]))
                        )
            assignments[elem.path] = sig
            elem2preds[elem.path] = pred_types
            progress_cbk(elem, pred_types)

        return RolloutPrediction(assignments, elem2preds, elem2inputs)


def is_mask_annot(a: cst.Annotation) -> bool:
    match a.annotation:
        case cst.Name(value=SpecialNames.TypeMask):
            return True
    return False


class DecodingOrder(ABC):
    @abstractmethod
    def traverse(self, analysis: UsageAnalysis) -> list[ProjectPath]:
        ...

    def traverse_length(self, n_elems: int) -> int:
        return n_elems

    def types_in_ctx(self) -> bool:
        return True


class DecodingOrders:
    class IndependentOrder(DecodingOrder):
        """Decode each element independently: the types predicted by the model will
        not be added to the context for later elements"""

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            return [e.path for e in analysis.project.all_elems()]

        @staticmethod
        def types_in_ctx() -> bool:
            return False

    class RandomOrder(DecodingOrder):
        "Visit all elements once in a random order."

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            elems = [e.path for e in analysis.project.all_elems()]
            random.shuffle(elems)
            return elems

    class Caller2Callee(DecodingOrder):
        """Visit the callers first before visiting the callees. The order among
        elements in a dependency cycle is arbitrary."""

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            sorted = list[ProjectPath]()
            visited = set[ProjectPath]()

            def visit(p: ProjectPath) -> None:
                if p in visited or p not in analysis.path2elem:
                    return
                visited.add(p)
                for u in analysis.used2user.get(p, []):
                    visit(u.user)
                sorted.append(p)

            for m in reversed(list(analysis.project.all_elems())):
                # start with the latest elements in the project
                visit(m.path)
            return sorted

    class Callee2Caller(DecodingOrder):
        """Visit the callees before visiting the callers. Give the reverse ordering
        of `Caller2Callee`"""

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            return list(reversed(DecodingOrders.Caller2Callee.traverse(analysis)))

    class DoubleTraversal(DecodingOrder):
        """Visit each element twice: `Callee2Caller` followed by `Caller2Callee`."""

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            pass1 = DecodingOrders.Callee2Caller.traverse(analysis)
            pass2 = list(reversed(pass1))
            return pass1 + pass2[1:]

        @staticmethod
        def traverse_length(n_elems: int) -> int:
            return 2 * n_elems - 1

    class RandomTwice(DecodingOrder):
        """Perform random traversal twice."""

        @staticmethod
        def traverse(analysis: UsageAnalysis) -> list[ProjectPath]:
            pass1 = DecodingOrders.RandomOrder.traverse(analysis)
            pass2 = DecodingOrders.RandomOrder.traverse(analysis)
            return pass1 + pass2

        @staticmethod
        def traverse_length(n_elems: int) -> int:
            return 2 * n_elems


def construct_model_inputs(
    main_mod: cst.Module,
    left_m: cst.Module | None,
    right_m: cst.Module | None,
    preamble: str,
    preamble_tkns: TokenSeq,
    ctx_args: CtxArgs,
) -> list[dict]:
    "Return a list of model inputs."
    main_code_string = wrap_main_code(main_mod.code)
    code_segs = main_code_string.split(SpecialNames.TypeMask)
    n_labels = len(code_segs) - 1

    if n_labels == 0:
        return []

    left_tks = None
    if left_m is not None:
        left_tks = DefaultTokenizer.encode(left_m.code, add_special_tokens=False)
    right_tks = None
    if right_m is not None:
        right_tks = DefaultTokenizer.encode(right_m.code, add_special_tokens=False)

    annots, types = collect_user_annotations(main_mod)
    assert_eq(
        len(annots), n_labels, extra_message=lambda: f"main code:\n{main_code_string}"
    )

    src = tokenized_src_from_segs(
        file=Path("[construct_model_inputs]"),
        repo=Path("[construct_model_inputs]"),
        preamble=preamble,
        tokenized_preamble=preamble_tkns,
        code_segs=code_segs,
        types=types,
        types_str=[SpecialNames.TypeMask] * n_labels,
        annots_info=annots,
        cst_code=main_code_string,
        left_extra_tks=left_tks,
        right_extra_tks=right_tks,
    )
    chunks, _ = src_to_chunks(src, (0, n_labels), ctx_args)
    return chunks
