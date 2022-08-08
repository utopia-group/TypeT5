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

    @staticmethod
    def from_modules(modules: Iterable[PythonModule]) -> "PythonProject":
        return PythonProject({m.name: m for m in modules})

    @staticmethod
    def from_root(root: Path) -> "PythonProject":
        """Root is typically the `src/` directory or just the root of the project."""
        modules = dict()

        for src in root.rglob("*.py"):
            if src.is_symlink():
                continue
            with src.open() as f:
                mod = cst.parse_module(f.read())

            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            modules[mod_name] = PythonModule.from_cst(mod, mod_name)

        for src in root.rglob("*.py"):
            if not src.is_symlink():
                continue
            mod_name = PythonProject.rel_path_to_module_name(src.relative_to(root))
            origin_name = PythonProject.rel_path_to_module_name(
                src.resolve().relative_to(root)
            )
            modules[mod_name] = modules[origin_name]

        return PythonProject(modules)

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        for module in self.modules.values():
            yield from module.all_funcs()

    @staticmethod
    def rel_path_to_module_name(rel_path: Path) -> str:
        return rel_path.with_suffix("").as_posix().replace("/", ".")

    @staticmethod
    def to_abs_import_path(current_mod: str, path: str) -> str:
        # take all leading dots
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


class UsageAnalysis:
    all_usages: list[FunctionUsage]
    caller2callees: dict[ProjectPath, list[FunctionUsage]]
    callee2callers: dict[ProjectPath, list[FunctionUsage]]
    path2func: dict[ProjectPath, PythonFunction]

    def __init__(self, project: PythonProject):
        path2func = {f.path: f for f in project.all_funcs()}
        all_methods = groupby(
            (f for f in project.all_funcs() if f.is_method), lambda f: f.name
        )

        def generate_usages(caller: ProjectPath, span: CodeRange, qname: QualifiedName):
            def gen_method_usages(method_name: str):
                if method_name in UsageAnalysis.CommonMethods:
                    return
                for f in all_methods.get(method_name, []):
                    yield FunctionUsage(caller, f.path, span, is_certain=False)

            match qname.source:
                case QualifiedNameSource.IMPORT:
                    segs = PythonProject.to_abs_import_path(mname, qname.name).split(
                        "."
                    )
                    assert len(segs) >= 2
                    callee = ProjectPath(".".join(segs[:-1]), segs[-1])
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
                            # handle self.method() calls
                            segs = [cls, m]

                    callee = ProjectPath(mname, ".".join(segs))
                    if callee in path2func:
                        yield FunctionUsage(caller, callee, span, is_certain=True)
                    elif (
                        cons := ProjectPath(callee.module, callee.path + ".__init__")
                    ) in path2func:
                        yield FunctionUsage(caller, cons, span, is_certain=True)
                    elif len(segs) >= 2 and segs[-2] != "<locals>":
                        # method fuzzy match case 2
                        yield from gen_method_usages(segs[-1])

        all_usages = list[FunctionUsage]()
        for mname, mod in project.modules.items():
            mod_usages = compute_module_usages(mod)
            for caller, span, qname in mod_usages:
                all_usages.extend(generate_usages(caller, span, qname))

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


def is_access_chain(node: cst.CSTNode) -> bool:
    match node:
        case cst.Attribute(lhs, cst.Name()):
            return is_access_chain(lhs)
        case cst.Name():
            return True
        case _:
            return False
