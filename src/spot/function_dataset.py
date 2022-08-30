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
    ProjectPath,
    ProjectUsage,
    PythonClass,
    PythonElem,
    PythonFunction,
    PythonVariable,
    UsageAnalysis,
    PythonProject,
    PythonModule,
    build_project_namespaces,
    remove_types,
    stub_from_module,
)


def dataset_from_repos(
    repos_root: Path,
    repos_paths: Iterable[Path],
    pre_args: PreprocessArgs,
    max_line_width: int = 200,
    max_workers: int | None = None,
    tqdm_args: dict = {},
) -> "SrcDataset":
    repos = list(repos_paths)
    srcs_list = pmap(
        repo_to_tk_srcs,
        repos_paths,
        [pre_args] * len(repos),
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
        if len(g) > 1:
            logging.warning(
                f"Multiple srcs for file '{g[0].file}' in repo '{repos_root / g[0].repo}'"
            )
    return SrcDataset(repos_root, all_srcs)


def repo_to_tk_srcs(
    repo: Path,
    pre_args: PreprocessArgs,
    max_line_width: int = 200,
) -> list[TokenizedSrc]:
    def src_filter(text):
        width = max(len(l) for l in text.split("\n"))
        return width <= max_line_width

    proj = PythonProject.from_root(
        repo,
        True,
        src_filter,
        drop_comments=pre_args.drop_comments,
    )

    analysis = UsageAnalysis(proj)
    sorted_moduels = analysis.sorted_modules

    # p2tks = {p: get_masked_fun_code(f) for p, f in analysis.path2func.items()}

    srcs = list[TokenizedSrc]()
    for mpath in sorted_moduels:
        mod = proj.modules[mpath]
        preamble_segs = list[str]()
        if pre_args.imports_in_preamble:
            wo_imports, imports = remove_imports(mod.tree)
            imports_part = cst.Module([cst.SimpleStatementLine([s]) for s in imports])
            preamble_segs.append(imports_part.code)
        if pre_args.stub_in_preamble:
            preamble_segs.append(stub_from_module(mod.tree).code)
        preamble = "".join(preamble_segs)
        tokenized_preamble = DefaultTokenizer.encode(preamble, add_special_tokens=False)

        mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))

        for elem in mod.all_elements():
            # the main code is the function body
            main_m = module_from_elems([elem], analysis.path2class)
            annots_info, types = collect_user_annotations(main_m)
            if len(annots_info) == 0:
                continue  # skip files with no label
            for info in annots_info:
                # the current implementation invalidates the src locations
                info.annot_range = None
            types_str = [
                main_m.code_for_node(not_none(info.annot).annotation)
                for info in annots_info
            ]
            replaces = dict()
            for info in annots_info:
                replaces[info.path] = mask_annot
            new_code = (
                "# BEGIN\n" + apply_annotations(main_m, replaces).code + "# END\n"
            )
            code_segs = new_code.split(SpecialNames.TypeMask)
            assert (
                len(code_segs) == len(types) + 1
            ), f"{len(code_segs)} != {len(types) + 1}. replaces: {replaces}\ncode: {new_code}"

            right_tks = None
            if pre_args.show_callers:
                # Right context: assemble code for callers
                caller_us = [
                    u
                    for u in not_none(analysis).used2user.get(elem.path, [])
                    if u.user != u.used
                ]
                # want certain usages to come first
                certain_callers = [u for u in caller_us if u.is_certain]
                potential_callers = [u for u in caller_us if not u.is_certain]
                right_m = module_from_elems(
                    [
                        analysis.path2elem[p.user]
                        for p in certain_callers + potential_callers
                    ],
                    analysis.path2class,
                )
                if pre_args.drop_env_types:
                    right_m = remove_types(right_m)
                right_tks = DefaultTokenizer.encode(
                    right_m.code, add_special_tokens=False
                )

            left_tks = None
            if pre_args.show_callees:
                # Left context: assemble code for callees
                callee_us = [
                    u
                    for u in not_none(analysis).user2used.get(elem.path, [])
                    if u.user != u.used
                ]
                # want certain and smaller usages to come last
                certain_callees = [u for u in callee_us if u.is_certain]
                potential_callees = [u for u in callee_us if not u.is_certain]
                left_m = module_from_elems(
                    [
                        analysis.path2elem[p.used]
                        for p in certain_callees + potential_callees
                    ],
                    analysis.path2class,
                    reversed=True,
                )
                if pre_args.drop_env_types:
                    left_m = remove_types(left_m)
                left_tks = DefaultTokenizer.encode(
                    left_m.code, add_special_tokens=False
                )

            file = Path(elem.path.module) / elem.path.path

            src = tokenized_src_from_segs(
                file=file,
                repo=repo,
                preamble=preamble,
                tokenized_preamble=tokenized_preamble,
                code_segs=code_segs,
                types=types,
                types_str=types_str,
                annots_info=annots_info,
                cst_code=main_m.code,
                left_extra_tks=left_tks,
                right_extra_tks=right_tks,
            )
            srcs.append(src)

    return srcs


def module_from_elems(
    elems: Sequence[PythonElem],
    path2class: dict[ProjectPath, PythonClass],
    reversed: bool = False,
) -> cst.Module:
    gvars = list[PythonVariable]()
    gfuncs = list[PythonFunction]()
    gclasses = dict[ProjectPath, list[PythonElem]]()

    for elem in elems:
        if elem.in_class:
            used = gclasses.setdefault(elem.path.pop(), [])
            used.append(elem)
        else:
            if isinstance(elem, PythonVariable):
                gvars.append(elem)
            elif isinstance(elem, PythonFunction):
                gfuncs.append(elem)

    stmt_groups = list[list]()

    def location_lines(path: ProjectPath):
        return [
            cst.EmptyLine(comment=cst.Comment("# " + path.module)),
        ]

    for var in gvars:
        stmt_groups.append(
            location_lines(var.path) + var.assignments + [cst.EmptyLine()]
        )

    for func in gfuncs:
        stmt_groups.append(
            [
                func.tree.with_changes(leading_lines=location_lines(func.path)),
                cst.EmptyLine(),
            ]
        )
        # stmt_groups.append([cst.EmptyLine(comment=cst.Comment(loc_comment)), func.tree])

    for path, elems in gclasses.items():
        cls = path2class[path]
        stmts = []
        cls_body = []
        for e in elems:
            if isinstance(e, PythonVariable):
                cls_body.extend([cst.SimpleStatementLine([a]) for a in e.assignments])
        for e in elems:
            if isinstance(e, PythonFunction):
                cls_body.append(e.tree)
                cls_body.append(cst.EmptyLine())
        new_body = cst.IndentedBlock(body=cls_body)
        stmts.append(
            cls.tree.with_changes(
                leading_lines=location_lines(cls.path),
                body=new_body,
            ),
        )
        stmts.append(cst.EmptyLine())
        stmt_groups.append(stmts)

    if reversed:
        stmt_groups.reverse()

    return cst.Module(body=list(seq_flatten(stmt_groups)))
