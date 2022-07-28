from .utils import *
from .data import remove_comments, remove_imports


def gen_stub(m: cst.Module, rm_comments=True, rm_imports=True) -> cst.Module:
    """Removes all comments and docstrings."""
    if rm_comments:
        m = remove_comments(m)
    if rm_imports:
        m, _ = remove_imports(m)
    m = m.visit(StubGenerator())
    m = remove_empty_lines(m)
    return m


def remove_empty_lines(m: cst.Module) -> cst.Module:
    m = m.visit(EmptyLineRemove())
    return m


OMIT = cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])


@dataclass
class ClassNamespace:
    all_elems: set[str] = field(default_factory=set)
    declared_elems: set[str] = field(default_factory=set)


class StubGenerator(cst.CSTTransformer):
    """Generate a stub module from a Python module."""

    def __init__(self):
        self.ns_stack = list[ClassNamespace]()

    def register_elem(self, name: str, declared: bool):
        if self.ns_stack:
            s = self.ns_stack[-1]
            s.all_elems.add(name)
            if declared:
                s.declared_elems.add(name)

    def visit_ClassDef(self, node: cst.ClassDef):
        self.ns_stack.append(ClassNamespace())

    def leave_ClassDef(self, node, updated: cst.ClassDef):
        s = self.ns_stack.pop()
        to_declare = s.all_elems.difference(s.declared_elems)
        if to_declare:
            more_stmts = [cst.parse_statement(f"{n}: ...") for n in to_declare]
            new_stmts = list(updated.body.body) + more_stmts
            updated = updated.with_changes(
                body=updated.body.with_changes(body=new_stmts)
            )
        return updated

    def leave_FunctionDef(self, node, updated: cst.FunctionDef):
        self.register_elem(updated.name.value, True)
        return updated.with_changes(body=OMIT, returns=None)

    def leave_Annotation(self, node, updated: cst.Annotation):
        return updated.with_changes(annotation=cst.Ellipsis())

    def leave_Param(self, node, updated: cst.Param):
        if updated.default is not None:
            updated = updated.with_changes(default=cst.Ellipsis())
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        if updated.value is not None:
            updated = updated.with_changes(value=cst.Ellipsis())
        return updated

    def leave_Assign(self, node, updated: cst.AnnAssign):
        return updated.with_changes(value=cst.Ellipsis())

    def leave_Attribute(self, node, updated: cst.Assign):
        match updated:
            case cst.Attribute(
                value=cst.Name(value="self"),
                attr=cst.Name(value=elem_name),
            ):
                self.register_elem(elem_name, False)
        return updated


class EmptyLineRemove(cst.CSTTransformer):
    def on_leave(self, node, updated):
        if hasattr(updated, "leading_lines") and updated.leading_lines:
            return updated.with_changes(leading_lines=[])
        return updated
