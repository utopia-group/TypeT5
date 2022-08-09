from .type_env import apply_annotations, collect_user_annotations
from .utils import *
from .tokenized_src import (
    PreprocessArgs,
    TokenizedSrc,
    tokenized_src_from_segs,
    remove_imports,
)
from .data import SrcDataset
from .static_analysis import (
    FunctionUsage,
    PythonFunction,
    UsageAnalysis,
    PythonProject,
    PythonModule,
    guess_src_root,
    remove_types,
)


def dataset_from_repos(
    repos_root: Path,
    repos_paths: Iterable[Path],
    preprocess_args: PreprocessArgs,
    max_line_width: int = 200,
    max_workers: int | None = None,
    tqdm_args: dict = {},
) -> "SrcDataset":
    repos = list(repos_paths)
    srcs_list = pmap(
        repo_to_tk_srcs,
        repos_paths,
        [preprocess_args] * len(repos),
        [max_line_width] * len(repos),
        desc="Generating dataset from repos",
        max_workers=max_workers,
        tqdm_args=tqdm_args,
    )
    all_srcs = list[TokenizedSrc]()
    for srcs in srcs_list:
        for src in srcs:
            src.repo = src.repo.relative_to(repos_root)
            src.file = src.repo / src.file
            all_srcs.append(src)
    for g in groupby(all_srcs, lambda s: s.file).values():
        assert len(g) == 1, f"Multiple srcs for file {g[0].file}"
    return SrcDataset(repos_root, all_srcs)


def repo_to_tk_srcs(
    repo: Path,
    preprocess_args: PreprocessArgs,
    max_line_width: int = 200,
) -> list[TokenizedSrc]:
    def src_filter(text):
        width = max(len(l) for l in text.split("\n"))
        return width <= max_line_width

    def get_masked_fun_code(f: PythonFunction) -> list[int]:
        tree = remove_types(f.tree)
        f_location = str(f.path)[: -(len(f.name) + 1)]
        el = cst.EmptyLine(comment=cst.Comment("# " + f_location))
        code = cst.Module([tree.with_changes(leading_lines=[el])]).code
        return DefaultTokenizer.encode(code, add_special_tokens=False)

    proj = PythonProject.from_root(
        repo,
        True,
        src_filter,
        drop_comments=preprocess_args.drop_comments,
    )
    analysis = UsageAnalysis(proj)
    p2tks = {p: get_masked_fun_code(f) for p, f in analysis.path2func.items()}

    srcs = list[TokenizedSrc]()
    for mpath, mod in proj.modules.items():
        preamble_segs = list[str]()
        if preprocess_args.imports_in_preamble:
            wo_imports, imports = remove_imports(mod.tree)
            imports_part = cst.Module([cst.SimpleStatementLine([s]) for s in imports])
            preamble_segs.append(imports_part.code)
        preamble = "".join(preamble_segs)
        tokenized_preamble = DefaultTokenizer.encode(preamble, add_special_tokens=False)

        mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))

        for fun in mod.all_funcs():
            # the main code is the function body
            fm = functions_to_module([fun])
            annots_info, types = collect_user_annotations(fm)
            if len(annots_info) == 0:
                continue  # skip files with no label
            for info in annots_info:
                # the current implementation invalidates the src locations
                info.annot_range = None
            types_str = [
                fm.code_for_node(not_none(info.annot).annotation)
                for info in annots_info
            ]
            replaces = dict()
            for info in annots_info:
                replaces[info.path] = mask_annot
            new_code = apply_annotations(fm, replaces).code
            code_segs = new_code.split(SpecialNames.TypeMask)
            assert (
                len(code_segs) == len(types) + 1
            ), f"{len(code_segs)} != {len(types) + 1}. replaces: {replaces}\ncode: {new_code}"

            # Right context: assemble code for callers
            caller_us = [
                u
                for u in analysis.callee2callers.get(fun.path, [])
                if u.caller != u.callee
            ]
            # want certain and smaller usages to come first
            certain_callers = sorted(
                [p2tks[u.caller] for u in caller_us if u.is_certain], key=len
            )
            potential_callers = sorted(
                [p2tks[u.caller] for u in caller_us if not u.is_certain], key=len
            )
            right_tks = list(seq_flatten(certain_callers + potential_callers))

            # Left context: assemble code for callees
            callee_us = [
                u
                for u in analysis.caller2callees.get(fun.path, [])
                if u.caller != u.callee
            ]
            # want certain and smaller usages to come last
            certain_callees = sorted(
                [p2tks[u.callee] for u in callee_us if u.is_certain],
                key=len,
                reverse=True,
            )
            potential_callees = sorted(
                [p2tks[u.callee] for u in callee_us if not u.is_certain],
                key=len,
                reverse=True,
            )
            left_tks = list(seq_flatten(potential_callees + certain_callees))
            file = Path(fun.path.module.replace(".", "/")) / fun.path.path

            src = tokenized_src_from_segs(
                file=file,
                repo=repo,
                preamble=preamble,
                tokenized_preamble=tokenized_preamble,
                code_segs=code_segs,
                types=types,
                types_str=types_str,
                annots_info=annots_info,
                cst_code=fm.code,
                left_extra_tks=left_tks,
                right_extra_tks=right_tks,
            )
            srcs.append(src)

    return srcs


def functions_to_module(fun: Sequence[PythonFunction]) -> cst.Module:
    # TODO: put into classes
    stmts = []
    for f in fun:
        f_location = str(f.path)[: -(len(f.name) + 1)]
        el = cst.EmptyLine(comment=cst.Comment("# " + f_location))
        stmts.append(f.tree.with_changes(leading_lines=[el]))
    return cst.Module(stmts)
