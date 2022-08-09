# -----------------------------------------------------------
# project-level static analysis

from collections import defaultdict
from posixpath import islink

from libcst import CSTNode

from .utils import *
from .type_env import AnnotInfo, AnnotPath, CodePathManager, collect_user_annotations
from libcst.metadata import (
    CodeRange,
    QualifiedNameProvider,
    QualifiedName,
    QualifiedNameSource,
    PositionProvider,
)


class ProjectPath(NamedTuple):
    """The path of a top-level function or method in a project."""

    module: str
    path: str

    def __str__(self) -> str:
        return f"{self.module}/{self.path}"

    def __repr__(self) -> str:
        return f"proj'{str(self)}'"

    @staticmethod
    def from_str(s: str) -> "ProjectPath":
        module, path = s.split("/")
        return ProjectPath(module, path)


@dataclass
class PythonFunction:
    name: str
    path: ProjectPath
    is_method: bool
    tree: cst.FunctionDef

    def __repr__(self):
        return f"PythonFunction(path={self.path})"


@dataclass
class PythonClass:
    name: str
    path: ProjectPath
    methods: list[PythonFunction]
    tree: cst.ClassDef

    def __repr__(self):
        return f"PythonClass(path={self.path}, n_methods={len(self.methods)})"


@dataclass
class PythonModule:
    functions: list[PythonFunction]
    classes: list[PythonClass]
    name: str
    tree: cst.Module

    @staticmethod
    def from_cst(module: cst.Module, name: str) -> "PythonModule":
        builder = PythonModuleBuilder(name)
        module.visit(builder)
        return builder.get_module()

    def __repr__(self):
        return f"PythonModule(n_functions={len(self.functions)}, n_classes={len(self.classes)})"

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        for f in self.functions:
            yield f
        for c in self.classes:
            for f in c.methods:
                yield f


@dataclass
class PythonProject:
    modules: dict[str, PythonModule]
    symlinks: dict[str, str]

    @staticmethod
    def from_modules(modules: Iterable[PythonModule]) -> "PythonProject":
        return PythonProject({m.name: m for m in modules}, dict())

    @staticmethod
    def from_root(
        root: Path,
        discard_bad_files: bool = False,
        src_filter: Callable[[str], bool] = lambda s: True,
        drop_comments: bool = True,
        ignore_dirs: set[str] = {".venv"},
    ) -> "PythonProject":
        """Root is typically the `src/` directory or just the root of the project."""
        modules = dict()
        symlinks = dict()

        all_srcs = [
            p
            for p in root.rglob("*.py")
            if p.relative_to(root).parts[0] not in ignore_dirs
        ]

        for src in all_srcs:
            if src.is_symlink():
                continue
            with src.open() as f:
                src_text = f.read()
            if not src_filter(src_text):
                continue
            try:
                mod = cst.parse_module(src_text)
            except cst.ParserSyntaxError as e:
                if discard_bad_files:
                    continue
                raise

            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            if drop_comments:
                mod = remove_comments(mod)
            modules[mod_name] = PythonModule.from_cst(mod, mod_name)

        for src in all_srcs:
            if not src.is_symlink():
                continue
            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            origin_name = PythonProject.rel_path_to_module_name(
                src.resolve().relative_to(root)
            )
            symlinks[mod_name] = origin_name

        return PythonProject(modules, symlinks)

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        for module in self.modules.values():
            yield from module.all_funcs()

    @staticmethod
    def rel_path_to_module_name(rel_path: Path) -> str:
        parts = rel_path.parts
        assert parts[-1].endswith(".py")
        if parts[0] == "src":
            parts = parts[1:]
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        return Path(*parts).with_suffix("").as_posix().replace("/", ".")


def to_abs_import_path(current_mod: str, path: str) -> str:
    # find all leading dots
    dots = 0
    while dots < len(path) and path[dots] == ".":
        dots += 1
    if dots == 0:
        return path
    mod_segs = split_import_path(current_mod)
    assert len(mod_segs) >= dots, "Cannot go up more levels."
    result_segs = mod_segs[:-dots]
    rest = path[dots:]
    if rest:
        result_segs.append(rest)
    return ".".join(result_segs)


_path_segs_cache = dict[str, list[str]]()


def split_import_path(path: str):
    if path in _path_segs_cache:
        return _path_segs_cache[path]

    segs = path.split(".")
    _path_segs_cache[path] = segs
    return segs


@dataclass
class FunctionUsage:
    caller: ProjectPath
    callee: ProjectPath
    call_site: CodeRange
    is_certain: bool  # some usage might not be certain, e.g. if it's a method call on a variable

    def __str__(self):
        return f"{self.caller} {'' if self.is_certain else 'potentially '}calls {self.callee}"


class ModuleNamespace:
    def __init__(self):
        self.children = dict[str, "ModuleNamespace"]()

    def __repr__(self):
        return f"ModuleNamespace({self.children})"

    def add_module(self, segs: list[str]) -> None:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                namespace.children[s] = ModuleNamespace()
                namespace = namespace.children[s]

    def resolve_path(self, segs: list[str]) -> ProjectPath | None:
        assert len(segs) >= 2, "Path should be at least 2 segments long."
        namespace = self
        matched = 0
        for s in segs[:-1]:
            if s in namespace.children:
                namespace = namespace.children[s]
                matched += 1
            else:
                break
        if matched == 0:
            return None
        return ProjectPath(".".join(segs[:matched]), ".".join(segs[matched:]))

    @staticmethod
    def from_modules(modules: Iterable[str]) -> "ModuleNamespace":
        root = ModuleNamespace()
        for m in modules:
            root.add_module(split_import_path(m))
        return root


class UsageAnalysis:
    all_usages: list[FunctionUsage]
    caller2callees: dict[ProjectPath, list[FunctionUsage]]
    callee2callers: dict[ProjectPath, list[FunctionUsage]]
    path2func: dict[ProjectPath, PythonFunction]
    namespaces: dict[str, str | dict]

    def __init__(self, project: PythonProject):
        path2func = {f.path: f for f in project.all_funcs()}
        all_methods = groupby(
            (f for f in project.all_funcs() if f.is_method), lambda f: f.name
        )
        namespaces = ModuleNamespace.from_modules(project.modules.keys())

        def generate_usages(
            mname: str, caller: ProjectPath, span: CodeRange, qname: QualifiedName
        ):
            def gen_method_usages(method_name: str):
                if method_name in UsageAnalysis.CommonMethods:
                    return
                for f in all_methods.get(method_name, []):
                    yield FunctionUsage(caller, f.path, span, is_certain=False)

            match qname.source:
                case QualifiedNameSource.IMPORT:
                    segs = to_abs_import_path(mname, qname.name).split(".")
                    if len(segs) < 2:
                        logging.warning(
                            f"Cannot resolve import '{qname.name}' in module: '{mname}'."
                        )
                        return
                    callee = namespaces.resolve_path(segs)
                    if callee is None:
                        return
                    if callee in path2func:
                        yield FunctionUsage(caller, callee, span, is_certain=True)
                    elif (
                        cons := ProjectPath(callee.module, callee.path + ".__init__")
                    ) in path2func:
                        yield FunctionUsage(caller, cons, span, is_certain=True)
                case QualifiedNameSource.LOCAL:
                    segs = qname.name.split(".")
                    match segs:
                        case ["<method>", m]:
                            # method fuzzy match case 1
                            yield from gen_method_usages(m)
                            return
                        case [cls, _, "<locals>", "self", m]:
                            segs = [cls, m]

                    callee = ProjectPath(mname, ".".join(segs))
                    if callee in path2func:
                        yield FunctionUsage(caller, callee, span, is_certain=True)
                    elif (
                        cons := ProjectPath(callee.module, callee.path + ".__init__")
                    ) in path2func:
                        yield FunctionUsage(caller, cons, span, is_certain=True)
                    elif len(segs) >= 2 and segs[-2] != "<locals>":
                        # method fuzzy match case 3
                        yield from gen_method_usages(segs[-1])

        best_usages = dict[tuple[ProjectPath, ProjectPath], FunctionUsage]()
        for mname, mod in project.modules.items():
            mod_usages = compute_module_usages(mod)
            for caller, span, qname in mod_usages:
                for u in generate_usages(mname, caller, span, qname):
                    up = u.caller, u.callee
                    if (
                        up not in best_usages
                        or u.is_certain > best_usages[up].is_certain
                    ):
                        best_usages[up] = u
        all_usages = list(best_usages.values())
        self.path2func = path2func
        self.all_usages = all_usages
        self.caller2callees = groupby(all_usages, lambda u: u.caller)
        self.callee2callers = groupby(all_usages, lambda u: u.callee)

    CommonMethods = {
        "__init__",
        "__repr__",
        "__new__",
        "__str__",
        "__hash__",
        "__eq__",
    }


def compute_module_usages(mod: PythonModule):
    """
    Compute a mapping from each method/function to the methods and functions they use.
    """
    wrapper = cst.MetadataWrapper(mod.tree, unsafe_skip_copy=True)
    name_map = wrapper.resolve(QualifiedNameProvider)
    pos_map = wrapper.resolve(PositionProvider)

    recorder = UsageRecorder(name_map, pos_map)
    result = list[tuple[ProjectPath, CodeRange, QualifiedName]]()

    for f in mod.all_funcs():
        f.tree.visit(recorder)
        # we only keep the first usage of each qualified name to avoid quadratic blow up
        callee_set = set[QualifiedName]()
        for u in recorder.usages:
            if u[1] in callee_set:
                continue
            result.append((f.path, u[0], u[1]))
            callee_set.add(u[1])
        recorder.usages.clear()

    return result


# -----------------------------------------------------------
# utilities for static analysis


class PythonModuleBuilder(cst.CSTVisitor):
    """Construct a `PythonModule` from a `cst.Module`."""

    def __init__(self, module_name: str):
        super().__init__()

        self.functions = list[PythonFunction]()
        self.classes = list[PythonClass]()
        self.current_class: PythonClass | None = None
        self.module = None
        self.module_name = module_name

    def get_module(self) -> PythonModule:
        assert self.module is not None, "Must visit a module first"
        return PythonModule(
            functions=self.functions,
            classes=self.classes,
            name=self.module_name,
            tree=self.module,
        )

    def visit_FunctionDef(self, node: cst.FunctionDef):
        is_method = self.current_class is not None
        name = node.name.value
        path = self.current_class.name + "." + name if self.current_class else name
        func = PythonFunction(
            name=node.name.value,
            path=ProjectPath(self.module_name, path),
            tree=node,
            is_method=is_method,
        )
        fs = self.current_class.methods if self.current_class else self.functions
        fs.append(func)
        return False  # don't visit inner functions

    def visit_ClassDef(self, node: cst.ClassDef):
        if self.current_class is not None:
            return False  # don't visit inner classes
        cls = PythonClass(
            name=node.name.value,
            path=ProjectPath(self.module_name, node.name.value),
            methods=[],
            tree=node,
        )
        self.current_class = cls

    def leave_ClassDef(self, node: cst.ClassDef):
        if self.current_class is not None:
            self.classes.append(self.current_class)
            self.current_class = None

    def visit_Module(self, node: cst.Module):
        self.module = node


class UsageRecorder(cst.CSTVisitor):
    """
    Records the (partially resoved) symbol usages.
    """

    def __init__(
        self,
        name_mapping: Mapping[cst.CSTNode, Collection[QualifiedName]],
        span_mapping,
    ):
        super().__init__()

        self.name_mapping = name_mapping
        self.span_mapping = span_mapping
        self.usages = list[tuple[CodeRange, QualifiedName]]()

    def visit_Call(self, node: cst.Call) -> None:
        lhs = node.func
        span = self.span_mapping[node]
        match lhs:
            case _ if is_access_chain(lhs) and lhs in self.name_mapping:
                for qn in self.name_mapping[lhs]:
                    self.usages.append((span, qn))
            case cst.Attribute(attr=cst.Name(attr)):
                # if the lhs cannot be resolved (e.g., is an expression), we record
                # the usage as potential method access.
                qname = QualifiedName(f"<method>.{attr}", QualifiedNameSource.LOCAL)
                self.usages.append((span, qname))

    def visit_Decorator(self, node: cst.Decorator):
        # do not count the calls in decorators
        return False


def is_access_chain(node: cst.CSTNode) -> bool:
    match node:
        case cst.Attribute(lhs, cst.Name()):
            return is_access_chain(lhs)
        case cst.Name():
            return True
        case _:
            return False


def stub_from_module(m: cst.Module, rm_comments=True, rm_imports=True) -> cst.Module:
    """Generate a stub module from normal python code."""
    if rm_comments:
        m = remove_comments(m)
    if rm_imports:
        m, _ = remove_imports(m)
    m = m.visit(StubGenerator())
    m = remove_empty_lines(m)
    return m


CNode = TypeVar("CNode", bound=cst.CSTNode)


def remove_imports(
    m: cst.Module,
) -> tuple[cst.Module, list[cst.Import | cst.ImportFrom]]:
    """Removes all top-level import statements and collect them into a list."""
    remover = ImportsRemover()
    m = m.visit(remover)
    return m, list(remover.import_stmts)


def remove_comments(m: CNode) -> CNode:
    """Removes all comments and docstrings."""
    return cast(CNode, m.visit(CommentRemover()))


def remove_empty_lines(m: CNode) -> CNode:
    m = cast(CNode, m.visit(EmptyLineRemove()))
    return m


def remove_types(m: CNode, type_mask="...") -> CNode:
    return cast(CNode, m.visit(AnnotRemover(type_mask)))


@dataclass
class ClassNamespace:
    all_elems: set[str] = field(default_factory=set)
    declared_elems: set[str] = field(default_factory=set)


class StubGenerator(cst.CSTTransformer):
    """Generate a stub module from a Python module."""

    OMIT = cst.SimpleStatementSuite([cst.Expr(cst.Ellipsis())])

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
        return updated.with_changes(body=StubGenerator.OMIT, returns=None)

    def leave_Annotation(self, node, updated: cst.Annotation):
        return updated.with_changes(annotation=cst.Ellipsis())

    def leave_Param(self, node, updated: cst.Param):
        # remove parameter type annotation and default value
        if updated.default is not None:
            updated = updated.with_changes(default=cst.Ellipsis())
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        # omit rhs of annotated assignments (if any)
        if updated.value is not None:
            updated = updated.with_changes(value=cst.Ellipsis())
        return updated

    def leave_Assign(self, node, updated: cst.AnnAssign):
        # omit rhs of assignments
        return updated.with_changes(value=cst.Ellipsis())

    def leave_Attribute(self, node, updated: cst.Assign):
        # record all atribute accesses involving `self`
        match updated:
            case cst.Attribute(
                value=cst.Name(value="self"),
                attr=cst.Name(value=elem_name),
            ):
                self.register_elem(elem_name, False)
        return updated

    def leave_Decorator(self, node, updated: cst.Decorator):
        # omit decorator call arguments
        match updated.decorator:
            case cst.Call(func=f):
                new_call = cst.Call(f, [cst.Arg(cst.Ellipsis())])
                updated = updated.with_changes(decorator=new_call)

        return updated


class EmptyLineRemove(cst.CSTTransformer):
    def on_leave(self, node, updated):
        if hasattr(updated, "leading_lines") and updated.leading_lines:
            return updated.with_changes(leading_lines=[])
        return updated


class CommentRemover(cst.CSTTransformer):
    """Removes comments and docstrings."""

    def leave_IndentedBlock(
        self, node: cst.IndentedBlock, updated: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        new_body = type(updated.body)(  # type: ignore
            filter(lambda n: not CommentRemover.is_doc_string(n), updated.body)
        )
        if len(new_body) != len(updated.body):
            return updated.with_changes(body=new_body)
        else:
            return updated

    def leave_Module(self, node, updated):
        return self.leave_IndentedBlock(node, updated)

    def leave_EmptyLine(self, node: cst.EmptyLine, updated: cst.EmptyLine):
        if updated.comment is not None:
            return cst.RemoveFromParent()
        else:
            return updated

    def leave_TrailingWhitespace(self, node, updated: cst.TrailingWhitespace):
        if updated.comment is not None:
            return updated.with_changes(comment=None)
        else:
            return updated

    @staticmethod
    def is_doc_string(node: cst.BaseStatement) -> bool:
        match node:
            case cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString())]):
                return True
            case _:
                return False


class ImportsRemover(cst.CSTTransformer):
    """Removes all top-level import statements and collect them into `self.import_stmts`."""

    def __init__(self):
        self.import_stmts = set[cst.Import | cst.ImportFrom]()

    def leave_Import(self, node: cst.Import, updated: cst.Import):
        self.import_stmts.add(updated)
        return cst.RemoveFromParent()

    def leave_ImportFrom(self, node: cst.ImportFrom, updated: cst.ImportFrom):
        self.import_stmts.add(updated)
        return cst.RemoveFromParent()

    def visit_FunctionDef(self, node):
        # stops traversal at inner levels.
        return False


class AnnotRemover(cst.CSTTransformer):
    """Removes all type annotations when possible or place them with a special symbol."""

    def __init__(self, type_mask: str = "..."):
        super().__init__()
        self.type_mask = cst.Ellipsis() if type_mask == "..." else cst.Name(type_mask)

    def leave_FunctionDef(self, node, updated: cst.FunctionDef) -> cst.FunctionDef:
        return updated.with_changes(returns=None)

    def leave_Param(self, node, updated: cst.Param) -> cst.Param:
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign) -> cst.AnnAssign:
        return updated.with_changes(annotation=cst.Annotation(self.type_mask))


def guess_src_root(proj_root: Path):
    if (proj_root / "src").exists():
        return proj_root / "src"
    return proj_root
