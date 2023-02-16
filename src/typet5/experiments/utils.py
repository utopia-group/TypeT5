from typet5.function_decoding import EvalResult
from typet5.static_analysis import (
    ElemSignature,
    FunctionSignature,
    ModuleName,
    ProjectPath,
    PythonModule,
    PythonProject,
    SignatureMap,
    VariableSignature,
    _VisitKind,
    is_type_rhs,
)
from typet5.type_check import MypyChecker, MypyFeedback, MypyResult
from typet5.type_env import normalize_type, parse_type_expr
from typet5.utils import *

_DefaultImport = cst.parse_statement(
    "from typing import Any, List, Tuple, Dict, Set, Union, Type, Callable # SPOT"
)


@dataclass
class SupportedSyntax:
    pattern_match: bool = True
    union_types: bool = True
    basic_types: bool = True
    named_exprs: bool = True


def remove_newer_syntax(m: cst.Module, supported: SupportedSyntax) -> cst.Module:
    """
    Remove or rewrite any newer python features that Type4Py doesn't support.
    """

    class PatternRewriter(cst.CSTTransformer):
        def leave_MatchAs(self, node, updated: cst.MatchAs):
            if updated.pattern:
                return updated.pattern
            elif updated.name:
                return updated.name
            else:
                # wild card pattern
                return cst.Name("_")

    def pattern_to_expr(pattern: cst.MatchPattern):
        np = cast(cst.BaseExpression, pattern.visit(PatternRewriter()))
        return cst.parse_expression(m.code_for_node(np))

    class Rewriter(cst.CSTTransformer):
        def leave_Annotation(self, node, updated: "cst.Annotation"):
            if supported.union_types:
                return updated
            ty = parse_type_expr(updated.annotation, silent=True)
            if ty is None:
                return cst.RemoveFromParent()
            ty = normalize_type(ty)  # this should get rid of the Union type syntax.
            return updated.with_changes(annotation=cst.parse_expression(str(ty)))

        def leave_Module(self, node, updated: "cst.Module"):
            new_lines = [_DefaultImport] if not supported.basic_types else []
            default_import = updated.code_for_node(_DefaultImport)
            for stmt in updated.body:
                if updated.code_for_node(stmt) != default_import:
                    new_lines.append(stmt)
            return updated.with_changes(body=new_lines)

        def leave_Match(self, node, updated: cst.Match):
            if supported.pattern_match:
                return updated
            subject = updated.subject
            if isinstance(subject, cst.Tuple):
                subject = subject.with_changes(
                    lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
                )

            conditions = [
                cst.Comparison(
                    subject,
                    [
                        cst.ComparisonTarget(
                            cst.Equal(),
                            pattern_to_expr(c.pattern),
                        )
                    ],
                )
                for c in updated.cases
            ]
            bodies = [c.body for c in updated.cases]
            if_clauses = None
            for cond, body in reversed(list(zip(conditions, bodies))):
                if_clauses = cst.If(cond, body, orelse=if_clauses)
            assert isinstance(if_clauses, cst.If)
            return if_clauses

        def leave_NamedExpr(self, node, updated: "cst.NamedExpr"):
            if supported.named_exprs:
                return updated
            return updated.value

    return m.visit(Rewriter())


def remove_newer_syntax_for_file(file: Path, rules: SupportedSyntax) -> bool:
    text = read_file(file)
    m = cst.parse_module(text)
    m = remove_newer_syntax(m, rules)
    new_text = m.code
    if new_text != text:
        write_file(file, new_text)
        return True
    return False


def remove_newer_syntax_for_repo(root: Path, rules: SupportedSyntax) -> None:
    all_files = [p for p in root.glob("**/*.py") if p.is_file()]
    changed = pmap(
        remove_newer_syntax_for_file,
        all_files,
        [rules] * len(all_files),
        desc="Removing newer syntax",
    )
    print(f"{sum(changed)} / {len(all_files)} files have been rewritten.")


def apply_sigmap(
    m: cst.Module,
    sigmap: SignatureMap,
    module_name: ModuleName,
    add_default_imports=True,
) -> cst.Module:
    """
    Apply the signature map to the module.
    """

    class Rewriter(cst.CSTTransformer):
        def __init__(self):
            super().__init__()
            self.path_stack = [ProjectPath(module_name, "")]
            self.visit_stack = [_VisitKind.Root]

        @property
        def current_path(self) -> ProjectPath:
            return self.path_stack[-1]

        @property
        def current_visit_kind(self) -> _VisitKind:
            return self.visit_stack[-1]

        def enter_(self, name: str, kind: _VisitKind):
            self.path_stack.append(self.current_path.append(name))
            self.visit_stack.append(kind)

        def exit_(self):
            self.path_stack.pop()
            self.visit_stack.pop()

        def visit_FunctionDef(self, node: cst.FunctionDef):
            self.enter_(node.name.value, _VisitKind.Function)

        def leave_FunctionDef(self, node, updated: cst.FunctionDef):
            if isinstance(sig := sigmap.get(self.current_path), FunctionSignature):
                try:
                    updated = sig.apply(updated)
                except LookupError:
                    pass
            self.exit_()
            return updated

        def visit_ClassDef(self, node: "cst.ClassDef") -> Optional[bool]:
            self.enter_(node.name.value, _VisitKind.Class)

        def leave_ClassDef(self, node, updated: cst.ClassDef):
            self.exit_()
            return updated

        def leave_AnnAssign(self, node, updated: cst.AnnAssign):
            target = None
            match updated.target:
                case cst.Name(name):
                    target = name
            if (
                target is not None
                and isinstance(
                    sig := sigmap.get(self.current_path.append(target)),
                    VariableSignature,
                )
                and sig.annot is not None
            ):
                updated = updated.with_changes(annotation=sig.annot)
            return updated

        def leave_Assign(self, node, updated: cst.Assign):
            target = None
            if self.current_visit_kind != _VisitKind.Function:
                match updated.targets:
                    case [cst.AssignTarget(target=cst.Name(name))]:
                        target = name
            if (
                target is not None
                and isinstance(
                    sig := sigmap.get(self.current_path.append(target)),
                    VariableSignature,
                )
                and sig.annot is not None
                and not (
                    self.current_visit_kind == _VisitKind.Root
                    and is_type_rhs(updated.value)
                )  # skip annotating type aliases
            ):
                return cst.AnnAssign(cst.Name(target), sig.annot, updated.value)
            return updated

        def leave_Module(self, node, updated: cst.Module):
            if add_default_imports:
                return updated.with_changes(
                    body=[_DefaultImport] + list(updated.body),
                )
            return updated

    return m.visit(Rewriter())


def quote_annotations(m: cst.Module, normalize_types: bool = True) -> cst.Module:
    """
    Quote all type annotations as strings in the module..
    """

    class Rewriter(cst.CSTTransformer):
        def leave_Annotation(self, node, updated: "cst.Annotation"):
            if updated.annotation is None:
                return updated
            if normalize_types:
                ty = parse_type_expr(updated.annotation)
                if ty is not None:
                    text = repr(str(ty.normalized()))
                else:
                    text = repr(show_expr(updated.annotation, quoted=False))
            else:
                text = repr(show_expr(updated.annotation, quoted=False))
            return updated.with_changes(annotation=cst.SimpleString(text))

    return m.visit(Rewriter())


def apply_sigmap_and_typecheck(
    project: PythonProject,
    sigmap: SignatureMap,
    workdir: Path,
    quote_types=True,
    binary_path: Optional[Path] = None,
) -> MypyResult:
    assert workdir.is_dir(), f"Workdir is not a directory: {workdir}"

    # write the type annotated source files to the workdir
    for name, m in project.modules.items():
        # file = workdir / project.module2src_file[name]
        file = workdir / (name.replace(".", "/") + ".py")
        file.parent.mkdir(parents=True, exist_ok=True)
        m1 = apply_sigmap(m.tree, sigmap, name)
        if quote_types:
            m1 = quote_annotations(m1, normalize_types=True)
        write_file(file, m1.code)
    # handle __init__.py files specially
    files = list(workdir.glob("**/*.py"))
    for f in files:
        if (d := f.with_suffix("")).is_dir():
            f.rename(d / "__init__.py")

    # now call the type checker
    r = MypyChecker.check_project(workdir, binary_path)
    if isinstance(r, str):
        raise RuntimeError(f"Type checking failed: {r}")
    return r


def count_type_errors(
    result: Iterable[MypyFeedback],
) -> int:
    error_codes = {
        "name-defined",
        "attr-defined",
        "arg-type",
        "return-value",
        "assignment",
    }

    n = 0
    for e in result:
        if e.error_code in error_codes:
            n += 1
    return n


def collect_project_type_errors(
    proj: PythonProject,
    sigmap: SignatureMap,
    workdir: Path = Path("mypy_temp"),
    binary_path: Path | None = None,
) -> list[MypyFeedback]:
    workdir = workdir / proj.name
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(exist_ok=True, parents=True)
    try:
        check_r = apply_sigmap_and_typecheck(
            proj, sigmap, workdir, binary_path=binary_path
        )
        return [e for es in check_r.error_dict.values() for e in es]
    except RuntimeError as e:
        print("Warning: mypy failed for project:", proj.name)
        return []
