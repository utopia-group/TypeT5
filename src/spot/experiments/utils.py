from spot.utils import *
from spot.type_env import parse_type_expr, normalize_type

_DefaultImport = cst.parse_statement(
    "from typing import Any, List, Tuple, Dict, Set, Union # SPOT"
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
