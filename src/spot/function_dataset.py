from .data import TokenizedSrcSet
from .static_analysis import (
    ElemSignature,
    FunctionSignature,
    ProjectPath,
    PythonClass,
    PythonElem,
    PythonFunction,
    PythonProject,
    PythonVariable,
    UsageAnalysis,
    VariableSignature,
    remove_comments,
    remove_types,
    stub_from_module,
)
from .tokenized_src import (
    PreprocessArgs,
    TokenizedSrc,
    TokenSeq,
    remove_imports,
    tokenized_src_from_segs,
)
from .type_env import apply_annotations, collect_user_annotations
from .utils import *


def dataset_from_repos(
    repos_root: Path,
    repos_paths: Iterable[Path],
    pre_args: PreprocessArgs,
    max_line_width: int = 200,
    max_workers: int | None = None,
    tqdm_args: dict = {},
) -> "TokenizedSrcSet":
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
    return TokenizedSrcSet(repos_root, all_srcs)


def mk_preamble(
    mod: cst.Module,
    pre_args: PreprocessArgs,
) -> tuple[str, TokenSeq]:
    preamble_segs = list[str]()
    if pre_args.imports_in_preamble:
        wo_imports, imports = remove_imports(mod)
        imports_part = cst.Module([cst.SimpleStatementLine([s]) for s in imports])
        preamble_segs.append(imports_part.code)
    if pre_args.stub_in_preamble:
        preamble_segs.append(stub_from_module(mod, lightweight=True).code)
    preamble_segs.append("\n# Used:\n")
    preamble = "".join(preamble_segs)
    tokenized_preamble = DefaultTokenizer.encode(preamble, add_special_tokens=False)
    return preamble, tokenized_preamble


def wrap_main_code(code: str) -> str:
    return f"# Target:\n{code}# Users:\n"


def data_project_from_dir(
    root: Path,
    max_line_width: int = 200,
    drop_comments: bool = True,
    file_filter: Callable[[Path], bool] = lambda p: True,
) -> PythonProject:
    def src2module(text: str):
        width = max(len(l) for l in text.split("\n"))
        if width > max_line_width:
            return None
        text = text.replace(SpecialNames.TypeMask, "MaskReplaced")
        mod = cst.parse_module(text)
        if drop_comments:
            mod = remove_comments(mod)
        return mod

    return PythonProject.from_root(
        root,
        True,
        src2module=src2module,
        file_filter=file_filter,
    )


def repo_to_tk_srcs(
    repo: Path,
    pre_args: PreprocessArgs,
    max_line_width: int = 200,
) -> list[TokenizedSrc]:
    proj = data_project_from_dir(
        repo, max_line_width=max_line_width, drop_comments=pre_args.drop_comments
    )
    analysis = UsageAnalysis(proj)
    sorted_moduels = analysis.sorted_modules

    srcs = list[TokenizedSrc]()
    for mpath in sorted_moduels:
        mod = proj.modules[mpath]
        preamble, tokenized_preamble = mk_preamble(mod.tree, pre_args)

        signature_map = dict() if pre_args.drop_env_types else None
        for elem in mod.all_elements():
            # the main code is the function body
            main_m = cst.Module(
                reformat_elems([elem], analysis.path2class, None, keep_body_types=True)
            )
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
            mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))
            replaces = dict()
            for info in annots_info:
                replaces[info.path] = mask_annot
            new_code = wrap_main_code(apply_annotations(main_m, replaces).code)
            code_segs = new_code.split(SpecialNames.TypeMask)
            assert (
                len(code_segs) == len(types) + 1
            ), f"{len(code_segs)} != {len(types) + 1}. replaces: {replaces}\ncode: {new_code}"

            left_m, right_m = ctx_modules_for_elem(
                elem, analysis, pre_args, signature_map
            )
            left_tks = None
            if left_m is not None:
                left_tks = DefaultTokenizer.encode(
                    left_m.code, add_special_tokens=False
                )
            right_tks = None
            if right_m is not None:
                right_tks = DefaultTokenizer.encode(
                    right_m.code, add_special_tokens=False
                )

            file = proj.root_dir / proj.module2src_file[mpath] / elem.path.path
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


def ctx_modules_for_elem(
    elem: PythonElem,
    analysis: UsageAnalysis,
    pre_args: PreprocessArgs,
    signature_map: dict[ProjectPath, ElemSignature] | None,
) -> tuple[cst.Module | None, cst.Module | None]:
    right_m = None
    certain_callers = dict()
    potential_callers = dict()
    if pre_args.max_callers > 0:
        # Right context: gather code for callers
        caller_us = [
            u for u in analysis.used2user.get(elem.path, []) if u.user != u.used
        ]
        # want certain usages to come first
        certain_callers = dict_subset(
            {u.user: None for u in caller_us if u.is_certain},
            pre_args.max_callers,
        )
        max_potential = pre_args.max_callers - len(certain_callers)
        potential_callers = dict_subset(
            {u.user: None for u in caller_us if not u.is_certain}, max_potential
        )

        right_m = cst.Module(
            reformat_elems(
                [analysis.path2elem[u] for u in certain_callers],
                analysis.path2class,
                signature_map,
            )
            + reformat_elems(
                [analysis.path2elem[u] for u in potential_callers],
                analysis.path2class,
                signature_map,
            )
        )

    left_m = None
    if pre_args.max_callees > 0:
        # Left context: assemble code for callees
        # map each displayed element to whether it's certainly relevant
        displayed = dict[ProjectPath, bool]()
        # the function of interest is certainly relevant
        displayed[elem.path] = True
        for p in certain_callers:
            displayed[p] = True
        for p in potential_callers:
            displayed[p] = False

        # the signature of these elements will be shown
        callee2certainty = dict[ProjectPath, bool]()
        for user, certain in displayed.items():
            for u in analysis.user2used.get(user, []):
                if not certain and not u.is_certain:
                    continue  # skip doubly uncertain usages
                if u.used in displayed:
                    continue  # skip already shown elements
                if callee2certainty.get(u.used, False) == False:
                    callee2certainty[u.used] = certain and u.is_certain

        # want certain usageså to come last
        certain_callees = dict_subset(
            {p: None for p in callee2certainty if callee2certainty[p]},
            pre_args.max_callees,
        )
        max_potential = pre_args.max_callees - len(certain_callees)
        potential_callees = dict_subset(
            {p: None for p in callee2certainty if not callee2certainty[p]},
            max_potential,
        )

        left_m = cst.Module(
            reformat_elems(
                [analysis.path2elem[p] for p in potential_callees],
                analysis.path2class,
                signature_map,
                reversed=True,
                signature_only=True,
            )
            + reformat_elems(
                [analysis.path2elem[p] for p in certain_callees],
                analysis.path2class,
                signature_map,
                reversed=True,
                signature_only=True,
            )
        )

    return left_m, right_m


def reformat_elems(
    elems: Sequence[PythonElem],
    path2class: dict[ProjectPath, PythonClass],
    signature_map: dict[ProjectPath, ElemSignature] | None,
    reversed: bool = False,
    signature_only=False,
    keep_body_types=False,
):
    """Generate code for the given list of python elements by
    reordering them and group class memebers into classes.

    If signature_map is not None, it will replace the type signatures
    with the ones in the provided map (or drop all types if not found in the map).

    If signature_only is True, the body of the functions will be omitted.
    """

    gvars = list[PythonVariable]()
    gfuncs = list[PythonFunction]()
    gclasses = dict[ProjectPath, dict[str, PythonElem]]()

    for elem in elems:
        if elem.parent_class:
            used = gclasses.setdefault(elem.parent_class, dict())
            used[elem.name] = elem
        else:
            if isinstance(elem, PythonVariable):
                gvars.append(elem)
            elif isinstance(elem, PythonFunction):
                gfuncs.append(elem)
    stmt_groups = list[
        Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement | cst.EmptyLine]
    ]()

    def location_lines(path: ProjectPath):
        return [
            cst.EmptyLine(comment=cst.Comment("# " + path.module)),
        ]

    def variable_lines(var: PythonVariable):
        assigns = list[cst.Assign]()
        lines = list[cst.Assign | cst.AnnAssign]()
        if signature_map is not None:
            for a in var.assignments:
                if isinstance(a, cst.AnnAssign) and a.value is not None:
                    assigns.append(cst.Assign([cst.AssignTarget(a.target)], a.value))
                if isinstance(a, cst.Assign):
                    assigns.append(a)
            match signature_map.get(var.path, None):
                case VariableSignature(annot=annot) if annot is not None:
                    sig_type = annot
                case _:
                    sig_type = cst.Annotation(cst.Ellipsis())
            if assigns:
                lines.append(
                    cst.AnnAssign(cst.Name(var.name), sig_type, assigns[0].value)
                )
                lines.extend(assigns[1:])
            else:
                lines.append(cst.AnnAssign(cst.Name(var.name), sig_type))
        else:
            lines.extend(var.assignments)
        return [cst.SimpleStatementLine([a]) for a in lines]

    def function_code(func: PythonFunction):
        func_node = func.tree
        if signature_only:
            OMIT = cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])
            func_node = func_node.with_changes(body=OMIT)
        if signature_map is not None:
            if not keep_body_types:
                func_node = remove_types(func_node)
            match signature_map.get(func.path, None):
                case FunctionSignature() as sig:
                    func_node = sig.apply(func_node)
        return func_node

    for var in gvars:
        to_add = variable_lines(var)
        if to_add:
            to_add[0] = to_add[0].with_changes(leading_lines=location_lines(var.path))
            group = to_add + [cst.EmptyLine()]
            stmt_groups.append(group)

    for func in gfuncs:
        func_node = function_code(func).with_changes(
            leading_lines=location_lines(func.path)
        )
        stmt_groups.append(
            [
                func_node,
                cst.EmptyLine(),
            ]
        )

    for path, members in gclasses.items():
        cls = path2class[path]
        cls_body = []
        for e in members.values():
            if isinstance(e, PythonVariable):
                cls_body.extend(variable_lines(e))
        for e in members.values():
            if isinstance(e, PythonFunction):
                method_node = function_code(e).with_changes(leading_lines=[])
                cls_body.append(method_node)
                cls_body.append(cst.EmptyLine())
        if not cls_body:
            continue
        new_body = cst.IndentedBlock(body=cls_body)
        stmts = [
            cls.tree.with_changes(
                leading_lines=location_lines(cls.path),
                body=new_body,
            ),
            cst.EmptyLine(),
        ]
        stmt_groups.append(stmts)

    if reversed:
        stmt_groups.reverse()

    result = list(seq_flatten(stmt_groups))
    # hide empty lines in the type signature to make the checker happy
    return cast(list[cst.SimpleStatementLine | cst.BaseCompoundStatement], result)
