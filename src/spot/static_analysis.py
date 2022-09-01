# -----------------------------------------------------------
# project-level static analysis

from ast import Str
from collections import defaultdict
from functools import lru_cache
from posixpath import islink
import enum

from .utils import *
from .type_env import AnnotInfo, AnnotPath, CodePathManager, collect_user_annotations
from libcst.metadata import (
    CodeRange,
    QualifiedNameProvider,
    QualifiedName,
    QualifiedNameSource,
    PositionProvider,
)

ModuleName = str
ModulePath = str


class ProjectPath(NamedTuple):
    """The path of a top-level function or method in a project."""

    module: ModuleName
    path: ModulePath

    def __str__(self) -> str:
        return f"{self.module}/{self.path}"

    def __repr__(self) -> str:
        return f"proj'{str(self)}'"

    def append(self, path: ModulePath) -> "ProjectPath":
        return ProjectPath(self.module, self.path + "." + path)

    def pop(self) -> "ProjectPath":
        p1 = ".".join(self.path.split(".")[:-1])
        return ProjectPath(self.module, p1)

    @staticmethod
    def from_str(s: str) -> "ProjectPath":
        module, path = s.split("/")
        return ProjectPath(module, path)


ProjNamespace = dict[str, ProjectPath]

# @dataclass
# class ProjectNamespace:
#     children: "PMap[str, ProjectNamespace | ProjectPath]"

#     def add_member(self, name: str, path: ProjectPath) -> "ProjectNamespace":
#         return ProjectNamespace(self.children.set(name, path))


@dataclass
class PythonFunction:
    name: str
    path: ProjectPath
    parent_class: ProjectPath | None
    tree: cst.FunctionDef

    def __repr__(self):
        return f"PythonFunction(path={self.path})"

    def in_class(self) -> bool:
        return self.parent_class is not None


@dataclass
class PythonVariable:
    name: str
    path: ProjectPath
    parent_class: ProjectPath | None
    assignments: list[
        cst.Assign | cst.AnnAssign
    ]  # only record assignments outside of functions

    def __repr__(self):
        return f"PythonAttribute(path={self.path})"

    def in_class(self) -> bool:
        return self.parent_class is not None


PythonElem = PythonFunction | PythonVariable

from functools import cached_property


@dataclass
class PythonClass:
    name: str
    path: ProjectPath
    attributes: dict[str, PythonVariable]
    methods: dict[str, PythonFunction]
    tree: cst.ClassDef
    superclasses: list[QualifiedName] | None = None

    def __repr__(self):
        return f"PythonClass(path={self.path}, n_attrs={len(self.attributes)}, n_methods={len(self.methods)})"

    @cached_property
    def is_dataclass(self) -> bool:
        for dec in self.tree.decorators:
            match dec.decorator:
                case cst.Name(value="dataclass") | cst.Call(
                    func=cst.Name(value="dataclass")
                ):
                    return True
        if self.superclasses:
            for sc in self.superclasses:
                if "NamedTuple" in sc.name.split("."):
                    return True
        return False


@dataclass
class PythonModule:
    functions: list[PythonFunction]
    global_vars: list[PythonVariable]
    classes: list[PythonClass]
    name: str
    imported_modules: set[ModuleName]
    defined_symbols: dict[str, ProjectPath]
    tree: cst.Module

    @staticmethod
    def from_cst(module: cst.Module, name: str) -> "PythonModule":
        builder = PythonModuleBuilder(name)
        module.visit(builder)
        return builder.get_module()

    def __repr__(self):
        return f"PythonModule(n_functions={len(self.functions)}, n_classes={len(self.classes)})"

    def all_funcs(self) -> Generator[PythonFunction, None, None]:
        yield from self.functions
        for c in self.classes:
            yield from c.methods.values()

    def all_vars(self) -> Generator[PythonVariable, None, None]:
        yield from self.global_vars
        for c in self.classes:
            yield from c.attributes.values()

    def all_elements(self) -> Generator[PythonElem, None, None]:
        yield from self.global_vars
        yield from self.functions
        for c in self.classes:
            yield from c.attributes.values()
            yield from c.methods.values()


@dataclass
class PythonProject:
    modules: dict[ModuleName, PythonModule]
    symlinks: dict[ModuleName, ModuleName]

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

    def all_vars(self) -> Generator[PythonVariable, None, None]:
        for module in self.modules.values():
            yield from module.all_vars()

    def all_elems(self) -> Generator[PythonElem, None, None]:
        for module in self.modules.values():
            yield from module.all_elements()

    @staticmethod
    def rel_path_to_module_name(rel_path: Path) -> ModuleName:
        parts = rel_path.parts
        assert parts[-1].endswith(".py")
        if parts[0] == "src":
            parts = parts[1:]
        if parts[-1] == "__init__.py":
            return ".".join(parts[:-1])
        else:
            # also remove the .py extension
            return ".".join([*parts[:-1], parts[-1][:-3]])


def to_abs_import_path(current_mod: ModuleName, path: str) -> ModuleName:
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
class ProjectUsage:
    user: ProjectPath
    used: ProjectPath
    call_site: CodeRange
    is_certain: bool  # some usage might not be certain, e.g. if it's a method call on an expression

    def __str__(self):
        return (
            f"{self.user} {'' if self.is_certain else 'potentially '}uses {self.used}"
        )


class ModuleHierarchy:
    def __init__(self):
        self.children = dict[str, "ModuleHierarchy"]()

    def __repr__(self):
        return f"ModuleNamespace({self.children})"

    def add_module(self, segs: list[str]) -> None:
        namespace = self
        for s in segs:
            if s in namespace.children:
                namespace = namespace.children[s]
            else:
                namespace.children[s] = ModuleHierarchy()
                namespace = namespace.children[s]

    def resolve_path(self, segs: list[str]) -> ProjectPath | None:
        if len(segs) < 2:
            return None
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
    def from_modules(modules: Iterable[str]) -> "ModuleHierarchy":
        root = ModuleHierarchy()
        for m in modules:
            root.add_module(split_import_path(m))
        return root


def sort_modules_by_imports(project: PythonProject) -> list[str]:
    "Sort modules topologically according to imports"
    sorted_modules = list[str]()
    visited = set[str]()

    def visit(m: str) -> None:
        if m in visited or m not in project.modules:
            return
        visited.add(m)
        if m in project.modules:
            for m2 in project.modules[m].imported_modules:
                visit(m2)
        sorted_modules.append(m)

    for m in project.modules:
        visit(m)
    return sorted_modules


def build_project_namespaces(
    project: PythonProject,
) -> dict[ModuleName, ProjNamespace]:
    """Return the visible project-defined symbols in each module."""
    sorted_modules = sort_modules_by_imports(project)
    result = dict[ModuleName, ProjNamespace]()
    for mod in sorted_modules:
        mv = _NsBuilder(mod, result)
        project.modules[mod].tree.visit(mv)
        new_ns = mv.namespace
        new_ns.update(project.modules[mod].defined_symbols)
        result[mod] = new_ns
    return result


class _NsBuilder(cst.CSTVisitor):
    def __init__(self, module_path: str, module2ns: Mapping[ModuleName, ProjNamespace]):
        self.module_path = module_path
        self.module2ns = module2ns
        self.namespace: ProjNamespace = dict()

    # todo: handle imported modules and renamed modules

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        src_mod = ".".join(
            parse_module_path(node.module, self.module_path, len(node.relative))
        )
        if src_mod not in self.module2ns:
            return
        src_ns = self.module2ns[src_mod]
        match node.names:
            case cst.ImportStar():
                self.namespace.update(src_ns)
            case _:
                for name in node.names:
                    match name.name:
                        case cst.Name(value=n1) if n1 in src_ns:
                            n2 = (
                                name.asname.value
                                if isinstance(name.asname, cst.Name)
                                else n1
                            )
                            self.namespace[n2] = src_ns[n1]


class UsageAnalysis:
    all_usages: list[ProjectUsage]
    path2elem: dict[ProjectPath, PythonElem]
    user2used: dict[ProjectPath, list[ProjectUsage]]
    used2user: dict[ProjectPath, list[ProjectUsage]]

    def get_var(self, path: ProjectPath) -> PythonVariable:
        v = self.path2elem[path]
        assert isinstance(v, PythonVariable)
        return v

    def get_func(self, path: ProjectPath) -> PythonFunction:
        v = self.path2elem[path]
        assert isinstance(v, PythonFunction)
        return v

    def __init__(self, project: PythonProject):
        self.project = project
        self.ns_hier = ModuleHierarchy.from_modules(project.modules.keys())
        module2ns = build_project_namespaces(project)
        self.sorted_modules = list(module2ns.keys())

        self.path2elem = {v.path: v for v in project.all_elems()}
        self.path2class = {
            cls.path: cls for mod in project.modules.values() for cls in mod.classes
        }

        # add mapping for star imports
        for mname, ns in module2ns.items():
            for s, p in ns.items():
                if p in self.path2class:
                    cls = self.path2class[p]
                    self.path2class.setdefault(ProjectPath(mname, s), cls)
                    if "__init__" in cls.methods:
                        self.path2elem.setdefault(
                            ProjectPath(mname, s),
                            self.path2class[p].methods["__init__"],
                        )
                elif p in self.path2elem:
                    self.path2elem.setdefault(ProjectPath(mname, s), self.path2elem[p])

        # resolve subtyping relations using `compute_module_usages`
        mod2usages = {
            mname: compute_module_usages(project.modules[mname])
            for mname in self.sorted_modules
        }
        # add elements from parents to subclass
        for mname in self.sorted_modules:
            mod = project.modules[mname]
            for cls in mod.classes:
                path_prefix = ProjectPath(mname, cls.name)
                bases = not_none(cls.superclasses)
                parents = [
                    x for p in bases if (x := self.find_class(mname, p)) is not None
                ]
                for parent in parents:
                    for a_name, a in parent.attributes.items():
                        self.path2elem[path_prefix.append(a_name)] = a
                    for m_name, m in parent.methods.items():
                        self.path2elem[path_prefix.append(m_name)] = m
                for m in cls.methods.values():
                    self.path2elem[m.path] = m
                for a in cls.attributes.values():
                    self.path2elem[a.path] = a

        # add path mapping for class constructors
        for f in list(self.path2elem.values()):
            if f.in_class and f.name == "__init__":
                cons_p = ProjectPath(f.path.module, f.path.path[: -len(".__init__")])
                self.path2elem[cons_p] = f

        all_class_members = {x.path for x in project.all_elems() if x.in_class}
        self.name2class_member = groupby(
            [self.path2elem[p] for p in all_class_members], lambda e: e.name
        )

        best_usages = dict[tuple[ProjectPath, ProjectPath], ProjectUsage]()
        for mname, usages in mod2usages.items():
            for caller, span, qname, is_call in usages:
                for u in self.generate_usages(mname, caller, span, qname, is_call):
                    up = u.user, u.used
                    if (
                        up not in best_usages
                        or u.is_certain > best_usages[up].is_certain
                    ):
                        best_usages[up] = u
        all_usages = list(best_usages.values())
        self.all_usages = all_usages
        self.user2used = groupby(all_usages, lambda u: u.user)
        self.used2user = groupby(all_usages, lambda u: u.used)

    def find_class(self, mname: ModuleName, qname: QualifiedName) -> PythonClass | None:
        cls_path = None
        match qname.source:
            case QualifiedNameSource.IMPORT:
                if len(segs := to_abs_import_path(mname, qname.name).split(".")) >= 2:
                    cls_path = self.ns_hier.resolve_path(segs)
            case QualifiedNameSource.LOCAL:
                cls_path = ProjectPath(mname, qname.name)

        if cls_path in self.path2class:
            return self.path2class[cls_path]
        return None

    def generate_usages(
        self,
        mname: ModuleName,
        caller: ProjectPath,
        span: CodeRange,
        qname: QualifiedName,
        is_call: bool,
    ):
        def gen_class_usages(member_name: str):
            if member_name.startswith("__") and member_name.endswith("__"):
                # skip common methods like __init__
                return
            for e in self.name2class_member.get(member_name, []):
                yield ProjectUsage(caller, e.path, span, is_certain=False)

        def gen_constructor_usages(path: ProjectPath):
            if not is_call or (cls := self.path2class.get(path)) is None:
                return
            used_elems = list[ProjectPath]()
            if cls.is_dataclass:
                for v in cls.attributes.values():
                    used_elems.append(v.path)
            elif "__init__" in cls.methods:
                used_elems.append(cls.methods["__init__"].path)
            for u in used_elems:
                yield ProjectUsage(
                    caller, self.path2elem[u].path, span, is_certain=True
                )

        match qname.source:
            case QualifiedNameSource.IMPORT:
                segs = to_abs_import_path(mname, qname.name).split(".")
                callee = self.ns_hier.resolve_path(segs)
                if callee is None:
                    return
                if callee in self.path2class:
                    yield from gen_constructor_usages(callee)
                elif callee in self.path2elem:
                    yield ProjectUsage(
                        caller, self.path2elem[callee].path, span, is_certain=True
                    )
            case QualifiedNameSource.LOCAL:
                segs = qname.name.split(".")
                match segs:
                    case ["<attr>", m]:
                        # method fuzzy match case 1
                        yield from gen_class_usages(m)
                        return
                    case [cls, _, "<locals>", "self", m]:
                        segs = [cls, m]

                callee = ProjectPath(mname, ".".join(segs))
                if callee in self.path2class:
                    yield from gen_constructor_usages(callee)
                elif callee in self.path2elem:
                    yield ProjectUsage(
                        caller, self.path2elem[callee].path, span, is_certain=True
                    )
                elif len(segs) >= 2 and segs[-2] != "<locals>":
                    # method fuzzy match case 3
                    yield from gen_class_usages(segs[-1])

    def assert_usages(self, caller: str, *callees: tuple[str, bool]) -> None:
        caller_p = ProjectPath.from_str(caller)
        expect = set()
        for callee, certain in callees:
            callee_p = ProjectPath.from_str(callee)
            expect.add((callee_p, certain))

        actual = {(u.used, u.is_certain) for u in self.user2used.get(caller_p, list())}

        try:
            assert_eq(actual, expect)
        except:
            usages = compute_module_usages(self.project.modules[caller_p.module])
            usages = groupby(usages, lambda x: x[0]).get(caller_p, [])
            print(f"Raw callees:")
            for u in usages:
                print("\t", u[2])
            raise


def compute_module_usages(mod: PythonModule):
    """
    Compute a mapping from each method/function to the methods and functions they use.
    Also resolve the qualified name of superclasses.
    """
    wrapper = cst.MetadataWrapper(mod.tree, unsafe_skip_copy=True)
    name_map = wrapper.resolve(QualifiedNameProvider)
    pos_map = wrapper.resolve(PositionProvider)

    recorder = UsageRecorder(name_map, pos_map)
    result = list[tuple[ProjectPath, CodeRange, QualifiedName, bool]]()

    for e in mod.all_elements():
        match e:
            case PythonFunction():
                e.tree.body.visit(recorder)
            case PythonVariable():
                for a in e.assignments:
                    if a.value:
                        a.value.visit(recorder)
        # we only keep the first occurance of each qualified name to save space
        best_callee = dict[QualifiedName, tuple[bool, CodeRange]]()
        for span, qn, is_call in recorder.usages:
            if qn not in best_callee or int(is_call) > int(best_callee[qn][0]):
                best_callee[qn] = (is_call, span)
        for qn, (is_call, span) in best_callee.items():
            result.append((e.path, span, qn, is_call))
        recorder.usages.clear()

    for cls in mod.classes:
        cls.superclasses = list()
        for b in cls.tree.bases:
            if b.value in name_map and name_map[b.value]:
                cls.superclasses.extend(name_map[b.value])
            elif isinstance(b.value, cst.Name):
                # unresovled parent class is treated as local for later processing
                cls.superclasses.append(
                    QualifiedName(b.value.value, QualifiedNameSource.LOCAL)
                )

    return result


# -----------------------------------------------------------
# utilities for static analysis


class _VisitType(enum.Enum):
    Root = enum.auto()
    Class = enum.auto()
    Function = enum.auto()


class PythonModuleBuilder(cst.CSTVisitor):
    """Construct a `PythonModule` from a `cst.Module`.
    If multiple definitions of the same name are found, only the last one is kept."""

    def __init__(self, module_name: str):
        super().__init__()

        self.functions = dict[str, PythonFunction]()
        self.global_vars = dict[str, PythonVariable]()
        self.classes = dict[str, PythonClass]()
        self.current_class: PythonClass | None = None
        self.visit_stack = [_VisitType.Root]
        self.module = None
        self.imported_modules = set[str]()
        self.defined_symbols = dict[str, ProjectPath]()

        self.module_name = module_name

    def _record_elem(self, e: PythonElem, cls: PythonClass | None):
        vars = cls.attributes if cls else self.global_vars
        funcs = cls.methods if cls else self.functions
        if cls is None:
            self.classes.pop(e.name, None)
            self.defined_symbols[e.name] = e.path
        match e:
            case PythonFunction(n):
                vars.pop(n, None)
                funcs[n] = e
            case PythonVariable(n, assignments=assignments):
                funcs.pop(n, None)
                if n in vars:
                    assert_eq(vars[n].path, e.path)
                    vars[n].assignments.extend(assignments)
                else:
                    vars[n] = e

    def get_module(self) -> PythonModule:
        assert self.module is not None, "Must visit a module first"
        return PythonModule(
            global_vars=list(self.global_vars.values()),
            functions=list(self.functions.values()),
            classes=list(self.classes.values()),
            name=self.module_name,
            imported_modules=self.imported_modules,
            defined_symbols=self.defined_symbols,
            tree=self.module,
        )

    def visit_FunctionDef(self, node: cst.FunctionDef):
        parent_type = self.visit_stack[-1]
        self.visit_stack.append(_VisitType.Function)
        if parent_type == _VisitType.Function:
            # skip inner functions
            return False
        for dec in node.decorators:
            match dec.decorator:
                case cst.Name("overload") | cst.Attribute(attr=cst.Name("overload")):
                    # ignore overload signatures
                    return False

        name = node.name.value
        path = self.current_class.name + "." + name if self.current_class else name
        func = PythonFunction(
            name=node.name.value,
            path=ProjectPath(self.module_name, path),
            tree=node,
            parent_class=self.current_class.path if self.current_class else None,
        )
        self._record_elem(func, self.current_class)

    def leave_FunctionDef(self, node) -> None:
        assert self.visit_stack[-1] == _VisitType.Function
        self.visit_stack.pop()

    def visit_ClassDef(self, node: cst.ClassDef):
        self.visit_stack.append(_VisitType.Class)
        if self.current_class is not None:
            return False  # skip inner classes all together
        cls = PythonClass(
            name=node.name.value,
            path=ProjectPath(self.module_name, node.name.value),
            attributes=dict(),
            methods=dict(),
            tree=node,
        )
        self.current_class = cls
        self.global_vars.pop(cls.name, None)
        self.functions.pop(cls.name, None)
        self.classes[cls.name] = cls
        self.defined_symbols[cls.name] = cls.path

    def leave_ClassDef(self, node: cst.ClassDef):
        assert self.visit_stack[-1] == _VisitType.Class
        self.current_class = None
        self.visit_stack.pop()

    # record global_vars and class attributes
    def visit_AnnAssign(self, node: cst.AnnAssign):
        cls = self.current_class
        cls_path = cls.path if cls else None
        var = None
        match self.visit_stack[-1], node.target:
            case (_VisitType.Root, cst.Name(value=n)):
                # global var assignment
                var = PythonVariable(n, ProjectPath(self.module_name, n), None, [node])
            case (_VisitType.Class, cst.Name(value=n)) if cls:
                # initialized outside of methods
                var = PythonVariable(n, cls.path.append(n), cls_path, [node])
            case (
                _VisitType.Function,
                cst.Attribute(value=cst.Name(value="self"), attr=cst.Name(value=n)),
            ) if cls:
                # initialized/accessed inside methods
                # TODO: figure out how to move the annotation to class scope
                var = PythonVariable(n, cls.path.append(n), cls_path, [])
        if var is not None:
            self._record_elem(var, cls)
        return None

    # record global_vars and class attributes
    def visit_Assign(self, node: cst.Assign) -> Optional[bool]:
        cls = self.current_class
        cls_path = cls.path if cls else None

        # class member declaration
        for target in node.targets:
            var = None
            match self.visit_stack[-1], target.target:
                case (_VisitType.Root, cst.Name(value=n)):
                    # global var assignment
                    var = PythonVariable(
                        n, ProjectPath(self.module_name, n), None, [node]
                    )
                case (_VisitType.Class, cst.Name(value=n)) if cls:
                    # initialized outside of methods
                    var = PythonVariable(n, cls.path.append(n), cls_path, [node])
                case (
                    _VisitType.Function,
                    cst.Attribute(value=cst.Name(value="self"), attr=cst.Name(value=n)),
                ) if cls:
                    # initialized/accessed inside methods
                    var = PythonVariable(n, cls.path.append(n), cls_path, [])
            if var is not None:
                self._record_elem(var, cls)

    def visit_Import(self, node: cst.Import):
        for alias in node.names:
            self.imported_modules.add(
                ".".join(parse_module_path(alias.name, self.module_name, 0))
            )

    def visit_ImportFrom(self, node: cst.ImportFrom):
        self.imported_modules.add(
            ".".join(
                parse_module_path(node.module, self.module_name, len(node.relative))
            )
        )

    def visit_Module(self, node: cst.Module):
        self.module = node


def parse_module_path(
    path_ex: cst.Attribute | cst.Name | None, cur_mod: str, dots: int
) -> list[str]:
    result = list[str]() if dots == 0 else cur_mod.split(".")[:-dots]

    def rec(ex):
        match ex:
            case None:
                pass
            case cst.Name(value=name):
                result.append(name)
            case cst.Attribute(value=attr, attr=cst.Name(value=name)):
                rec(attr)
                result.append(name)
            case _:
                raise ValueError(f"Cannot parse {ex} as module path")

    rec(path_ex)
    return result


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
        self.parents = list[cst.CSTNode]()
        self.usages = list[tuple[CodeRange, QualifiedName, bool]]()

    def _resolve(self, name: cst.CSTNode):
        if is_access_chain(name) and name in self.name_mapping:
            srcs = self.name_mapping[name]
            if len(srcs) == 0 and isinstance(name, cst.Name):
                # unresolved symbols are put into the 'LOCAL' category for later processing
                # due to star imports.
                return [QualifiedName(name.value, QualifiedNameSource.LOCAL)]
            else:
                return [s for s in srcs if s.name != "builtins.None"]
        return []

    def is_call_name(self) -> bool:
        return len(self.parents) > 0 and isinstance(self.parents[-1], cst.Call)

    def record_name_use(self, name: cst.CSTNode):
        for src in self._resolve(name):
            self.usages.append((self.span_mapping[name], src, self.is_call_name()))

    def visit_Attribute(self, node: cst.Attribute):
        if not self._resolve(node):
            # if the access cannot be resolved (e.g., is an expression), we record
            # the usage as potential method access.
            qname = QualifiedName(
                f"<attr>.{node.attr.value}", QualifiedNameSource.LOCAL
            )
            span = self.span_mapping[node]
            self.usages.append((span, qname, self.is_call_name()))
            return True
        else:
            # if this access is resolved, do not record prefixes as usages
            return False

    def on_visit(self, node: cst.CSTNode) -> Optional[bool]:
        self.record_name_use(node)
        self.parents.append(node)
        return super().on_visit(node)

    def on_leave(self, node: cst.CSTNode) -> Optional[bool]:
        self.parents.pop()

    # avoid visiting the following nodes
    def visit_Decorator(self, node: cst.Decorator):
        # do not count the calls in decorators
        return False

    # avoid using type annotation to track usages
    def visit_Annotation(self, node: cst.Annotation):
        return False


@lru_cache(maxsize=128)
def is_access_chain(node: cst.CSTNode) -> bool:
    """Return whether the node is an access chain of the form `a.b.c`. A simple name
    is also considered an access chain."""
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
        self.nest_level = 0

    def register_elem(self, name: str, declared: bool):
        if self.ns_stack:
            s = self.ns_stack[-1]
            s.all_elems.add(name)
            if declared:
                s.declared_elems.add(name)

    def visit_ClassDef(self, node: cst.ClassDef):
        self.nest_level += 1
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
        self.nest_level -= 1
        return updated

    def visit_FunctionDef(self, node):
        self.nest_level += 1

    def leave_FunctionDef(self, node, updated: cst.FunctionDef):
        self.register_elem(updated.name.value, True)
        self.nest_level -= 1
        return updated.with_changes(body=StubGenerator.OMIT, returns=None)

    def leave_Annotation(self, node, updated: cst.Annotation):
        return updated.with_changes(annotation=cst.Ellipsis())

    def leave_Param(self, node, updated: cst.Param):
        # remove parameter type annotation and default value
        if updated.default is not None:
            updated = updated.with_changes(default=cst.Ellipsis())
        return updated.with_changes(annotation=None)

    def leave_AnnAssign(self, node, updated: cst.AnnAssign):
        if self.nest_level == 0:
            return updated
        # omit rhs of annotated assignments (if any)
        if updated.value is not None:
            updated = updated.with_changes(value=cst.Ellipsis())
        return updated

    def leave_Assign(self, node, updated: cst.AnnAssign):
        if self.nest_level == 0:
            return updated
        # omit rhs of assignments unless it's a type variable
        return updated.with_changes(value=cst.Ellipsis())
        # match updated.value:
        #     case cst.Call(func=cst.Name("TypeVar")) | cst.Call(
        #         func=cst.Attribute(attr=cst.Name("TypeVar"))
        #     ) | None:
        #         return updated
        #     case _:
        #         return updated.with_changes(value=cst.Ellipsis())

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
