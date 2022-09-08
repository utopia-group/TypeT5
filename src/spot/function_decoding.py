import asyncio
import copy
import random

import torch

from .data import CtxArgs, src_to_chunks, type_accuracies
from .function_dataset import (
    ElemSignature,
    ctx_modules_for_elem,
    mk_preamble,
    reformat_elems,
)
from .model import ModelWrapper
from .static_analysis import (
    FunctionSignature,
    ModuleName,
    ProjectPath,
    PythonElem,
    PythonFunction,
    PythonProject,
    PythonVariable,
    UsageAnalysis,
    VariableSingature,
)
from .tokenized_src import PreprocessArgs, TokenSeq, tokenized_src_from_segs
from .type_check import PythonType, parse_type_expr
from .type_env import AnnotCat, collect_user_annotations
from .utils import *


@dataclass
class RolloutPrediction:
    assignments: dict[ProjectPath, ElemSignature]
    elem2preds: dict[ProjectPath, Sequence[PythonType]]
    elem2inputs: dict[ProjectPath, dict]


@dataclass
class EvalResult:
    predictions: Sequence[RolloutPrediction]
    accuracies: dict[str, Any]


@dataclass
class RolloutCtx:
    model: ModelWrapper

    async def evaluate_on_projects(
        self,
        projects: Sequence[PythonProject],
        pre_args: PreprocessArgs,
        decode_order: "DecodingOrder",
        common_type_names: set[str],
        concurrency: int = DefaultWorkers,
        tqdm_args: dict = {},
    ):
        """Evaluate the model's prediction accuracy on a given set of projects, masking
        any existing type annotations and treat them as the ground truth.
        Note that the model does make predictions for those places with a missing type
        annotation, but they are not counted in the accuracy computation (and only serve as
        an intermediate for information propogation).
        """
        n_total_elems = sum(1 for p in projects for e in p.all_elems())
        all_label_sigs = list[ElemSignature]()
        all_pred_sigs = list[ElemSignature]()
        rollouts: list[Any] = [None for _ in projects]
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
                r = await self.project_rollout(
                    project.mask_types(),
                    pre_args,
                    decode_order,
                    cpu_executor=cpu_executor,
                    model_executor=model_executor,
                    progress_cbk=lambda e, p: pbar.update(),
                )
                rollouts[id] = r
                for p in label_sigs:
                    all_label_sigs.append(label_sigs[p])
                    all_pred_sigs.append(r.assignments[p])

            await throttled_async_run(
                eval_project, list(enumerate(projects)), concurrency=concurrency
            )

        accs = accuracy_from_signatures(
            all_pred_sigs,
            all_label_sigs,
            common_type_names,
            allow_implicit_none=True,
        )
        return EvalResult(rollouts, accs)

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
            cpu_executor, UsageAnalysis, project
        )
        to_visit = [analysis.path2elem[p] for p in decode_order.traverse(analysis)]
        visit_set = {e.path for e in to_visit}
        for e in project.all_elems():
            assert e.path in visit_set, f"{e.path} not in the decoder order."
        preamble_cache = dict[ModuleName, tuple[str, TokenSeq]]()

        assignments = dict[ProjectPath, ElemSignature]()
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
                for i, a in enumerate(elem_sig.all_annots()):
                    if a is None:
                        elem_sig.set_annot_(i, mask_annot)
                elem_map = {elem.path: elem_sig}
            else:
                raise NotImplemented(f"Unsupported element type {type(elem)}")
            main_lines = reformat_elems(
                [elem],
                analysis.path2class,
                cast(dict[ProjectPath, ElemSignature], elem_map),
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
                    preds, _ = await eloop.run_in_executor(
                        model_executor, self.model.predict_on_batch, chunk
                    )
                    pred_types.extend(preds[0])
                elem2inputs[elem.path] = model_inputs[0]

                # update the signature with the predicted types
                sig = copy.deepcopy(sig)
                if isinstance(sig, VariableSingature):
                    assert sig.annot is None or is_mask_annot(
                        sig.annot
                    ), f"For {elem}, sig={sig}"
                    assert_eq(len(pred_types), 1)
                    sig.annot = cst.Annotation(cst.parse_expression(str(pred_types[0])))
                elif isinstance(elem, PythonFunction):
                    for i, a in enumerate(sig.all_annots()):
                        if a is None or is_mask_annot(a):
                            new_type = cst.parse_expression(str(pred_types[i]))
                            sig.set_annot_(i, cst.Annotation(new_type))
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


def construct_model_inputs(
    main_mod: cst.Module,
    left_m: cst.Module | None,
    right_m: cst.Module | None,
    preamble: str,
    preamble_tkns: TokenSeq,
    ctx_args: CtxArgs,
) -> list[dict]:
    "Return a list of model inputs."
    main_code_string = "# BEGIN\n" + main_mod.code + "# END\n"
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
    for i, chunk in enumerate(chunks):
        chunks[i] = {
            "input_ids": torch.tensor([chunk["input_ids"]]),
            "labels": torch.tensor([chunk["labels"]]),
            "n_labels": [chunk["n_labels"]],
        }
    return chunks


def accuracy_from_signatures(
    predictions: Sequence[ElemSignature],
    labels: Sequence[ElemSignature],
    common_type_names: set[str],
    allow_implicit_none: bool = True,
) -> dict[str, Any]:
    pred_types = list[PythonType]()
    label_types = list[PythonType]()
    types_cat = list[AnnotCat]()
    types_pos = list[int]()
    dummy_mod = cst.Module([])

    def record_pair(
        pred: cst.Annotation | None,
        label: cst.Annotation | None,
        cat: AnnotCat,
        pos: int,
    ):
        if (
            label is None
            or (lt := parse_type_expr(dummy_mod, label.annotation)) is None
        ):
            # no label available
            return
        assert pred is not None
        assert (pt := parse_type_expr(dummy_mod, pred.annotation)) is not None
        label_types.append(lt)
        pred_types.append(pt)
        types_cat.append(cat)
        types_pos.append(pos)

    for p, l in zip(predictions, labels):
        match p, l:
            case (VariableSingature(pa), VariableSingature(la, in_class=in_class)):
                cat = AnnotCat.ClassAtribute if in_class else AnnotCat.GlobalVar
                record_pair(pa, la, cat, 0)
            case (
                FunctionSignature(p_params, p_return),
                FunctionSignature(l_params, l_return, in_class=in_class),
            ):
                for i, (pa, la) in enumerate(zip(p_params, l_params)):
                    record_pair(pa, la, AnnotCat.FuncArg, i)
                record_pair(p_return, l_return, AnnotCat.FuncReturn, len(l_params))

    return type_accuracies(
        pred_types,
        label_types,
        types_cat,
        types_pos,
        common_type_names,
        allow_implicit_none=allow_implicit_none,
    )
