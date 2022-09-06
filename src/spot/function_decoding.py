import asyncio
import copy
import random

import torch

from .data import CtxArgs, SrcChunkInfo, src_to_chunks

from .type_env import AnnotInfo, collect_user_annotations

from .function_dataset import (
    ElemSignature,
    ctx_modules_for_elem,
    mk_preamble,
    reformat_elems,
)

from .tokenized_src import (
    PreprocessArgs,
    TokenSeq,
    TokenizedSrc,
    tokenized_src_from_segs,
)

from .type_check import PythonType
from .model import ModelWrapper
from .utils import *
from .static_analysis import (
    ModuleName,
    ProjectPath,
    PythonElem,
    PythonFunction,
    PythonProject,
    PythonVariable,
    UsageAnalysis,
    VariableSingature,
)


@dataclass
class RolloutPrediction:
    assignments: dict[ProjectPath, ElemSignature]
    elem2preds: dict[ProjectPath, Sequence[PythonType]]
    elem2inputs: dict[ProjectPath, dict]


@dataclass
class RolloutCtx:
    model: ModelWrapper

    async def project_rollout(
        self,
        project: PythonProject,
        pre_args: PreprocessArgs,
        decode_order: Callable[[UsageAnalysis], Sequence[ProjectPath]],
        cpu_executor: ProcessPoolExecutor,
        model_executor: ThreadPoolExecutor,
        progress_cbk: Callable[
            [PythonElem, Sequence[PythonType]], None
        ] = lambda x, y: None,
    ) -> RolloutPrediction:
        """Note: when evaluating on dataset with ground truth labels, we need to
        first replace all labels with `SpecialNames.TypeMask` before feeding to
        this function."""

        if model_executor._max_workers > 1:
            logging.warning("Model executor is not single threaded.")

        eloop = asyncio.get_event_loop()
        analysis: UsageAnalysis = await eloop.run_in_executor(
            cpu_executor, UsageAnalysis, project
        )
        elements = [analysis.path2elem[p] for p in decode_order(analysis)]
        assert_eq(len(elements), len(list(project.all_elems())))
        preamble_cache = dict[ModuleName, tuple[str, TokenSeq]]()

        assignments = dict[ProjectPath, ElemSignature]()
        elem2preds = dict[ProjectPath, Sequence[PythonType]]()
        elem2inputs = dict[ProjectPath, dict]()
        mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))

        # Parallelize computation between dependency-free elements
        for elem in elements:
            assert (
                elem.path not in assignments
            ), f"Element with path {elem.path} already assigned with signature {assignments[elem.path]}"
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
                elem_map = {
                    elem.path: VariableSingature(sig.annot if sig.annot else mask_annot)
                }
            elif isinstance(elem, PythonFunction):
                sig = elem.get_signature()
                elem_sig = copy.deepcopy(sig)
                for i, a in enumerate(elem_sig.annots):
                    if a is None:
                        elem_sig.annots[i] = mask_annot
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
                elem, analysis, pre_args, assignments
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
                    for i, a in enumerate(sig.annots):
                        if a is None or is_mask_annot(a):
                            new_type = cst.parse_expression(str(pred_types[i]))
                            sig.annots[i] = cst.Annotation(new_type)
            assignments[elem.path] = sig
            elem2preds[elem.path] = pred_types
            progress_cbk(elem, pred_types)

        return RolloutPrediction(assignments, elem2preds, elem2inputs)


def is_mask_annot(a: cst.Annotation) -> bool:
    match a.annotation:
        case cst.Name(value=SpecialNames.TypeMask):
            return True
    return False


class DecodingOrders:
    @staticmethod
    def random_order(analysis: UsageAnalysis) -> Sequence[ProjectPath]:
        elems = [e.path for e in analysis.project.all_elems()]
        random.shuffle(elems)
        return elems

    @staticmethod
    def caller2callee(analysis: UsageAnalysis) -> Sequence[ProjectPath]:
        """Visit the callers first before visiting the callees. The order among
        elements in a dependency cycle is arbitrary."""
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


def construct_model_inputs(
    main_code: cst.Module,
    left_m: cst.Module | None,
    right_m: cst.Module | None,
    preamble: str,
    preamble_tkns: TokenSeq,
    ctx_args: CtxArgs,
) -> list[dict]:
    "Return a list of model inputs."
    main_code_string = "# BEGIN\n" + main_code.code + "# END\n"
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

    annots, types = collect_user_annotations(main_code)
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
