import enum
import time
from dataclasses import dataclass, field
from posixpath import dirname, realpath
from typing import *

from libcst.metadata import CodeRange, PositionProvider
from pyrsistent import plist
from pyrsistent.typing import PList

from .type_check import *
from .utils import *


@enum.unique
class AnnotCat(enum.Enum):
    """The category of an annotation."""

    FuncArg = enum.auto()
    FuncReturn = enum.auto()
    ClassAtribute = enum.auto()
    GlobalVar = enum.auto()
    LocalVar = enum.auto()


@dataclass(order=True, unsafe_hash=True)
class AnnotPath:
    """The unique path of a type annoation."""

    value: PList[str]

    def __repr__(self):
        return f"AnnotPath({self.__str__()})"

    def __str__(self):
        return f"'{'.'.join(map(str, self.value.reverse()))}'"

    def append(self, seg: str, id: int) -> "AnnotPath":
        seg = seg if id == 0 else f"{seg}[{id}]"
        return AnnotPath(self.value.cons(seg))

    def pop(self) -> "AnnotPath":
        return AnnotPath(self.value.rest)


def annot_path(*segs: str) -> AnnotPath:
    return AnnotPath(plist(segs, reverse=True))


@dataclass
class AnnotInfo:
    path: AnnotPath
    cat: AnnotCat
    annot: Optional[cst.Annotation]
    annot_range: Optional[CodeRange]

    def __str__(self):
        if self.annot is not None:
            expr = cst.Module([]).code_for_node(self.annot.annotation)
        else:
            expr = None
        return f"AnnotInfo({str(self.path)}, {self.cat.name}, range={self.annot_range}, expr={expr})"


def collect_annots_info(code: cst.Module | cst.MetadataWrapper) -> list[AnnotInfo]:
    """Collect all AnnotInfo in the given source code.
    Note: the order of the paths are not guaranteed to follow the order of the source code.
    Check the `annot_range` of the returned AnnotInfo to get the correct order."""
    collector = AnnotCollector()
    if not isinstance(code, cst.MetadataWrapper):
        code = cst.MetadataWrapper(code)
    code.visit(collector)
    return collector.annots_info


def collect_user_annotations(
    code: cst.Module | cst.MetadataWrapper,
) -> tuple[list[AnnotInfo], list["PythonType"]]:
    """Collect all user-added type annotations in the given source code. Unlike `collect_annots_info`,
    The order of the returned annotations is gurated to follow the order of the source code."""

    def as_tuple(pos: CodePosition):
        return pos.line, pos.column

    annots = collect_annots_info(code)
    m = code if isinstance(code, cst.Module) else code.module

    types: list[PythonType] = []
    annots_info: list[AnnotInfo] = []
    labels_pos: list[tuple[int, int]] = []

    for info in annots:
        if info.annot is None:
            continue
        ty = parse_type_expr(m, info.annot.annotation, silent=True)
        if ty is None:
            continue
        types.append(ty)
        annots_info.append(info)
        labels_pos.append(as_tuple(not_none(info.annot_range).start))

    reorder = sorted(range(len(labels_pos)), key=lambda i: labels_pos[i])
    types = [types[i] for i in reorder]
    annots_info = [annots_info[i] for i in reorder]
    return annots_info, types


def apply_annotations(
    code: cst.Module,
    annots: dict[AnnotPath, cst.Annotation],
):
    try:
        return code.visit(AnnotApplier(annots))
    except Exception as e:
        raise RuntimeError(
            f"Failed to apply annotations for the following code:\n {code.code}"
        ) from e


def add_stmts(
    code: cst.Module,
    stmts: Sequence[cst.BaseStatement],
):
    return code.visit(StmtsAdder(stmts))


class CodePathManager:
    def __init__(self):
        super().__init__()
        self.path: AnnotPath = annot_path()
        self.path_name_counts: dict[AnnotPath, dict[str, int]] = {}

    def get_path(self, name: str) -> AnnotPath:
        counts = self.path_name_counts.get(self.path, {})
        self.path_name_counts[self.path] = counts
        id = counts.get(name, 0)
        counts[name] = id + 1
        return self.path.append(name, id)


class AnnotCollector(cst.CSTVisitor, CodePathManager):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        CodePathManager.__init__(self)
        # store the type annotations
        self.annots_info: list[AnnotInfo] = []
        self.cat_stack = [AnnotCat.GlobalVar]

    def on_visit(self, node):
        match node:
            case cst.FunctionDef():
                self.path = self.get_path(node.name.value)
                self.cat_stack.append(AnnotCat.LocalVar)
            case cst.ClassDef():
                self.path = self.get_path(node.name.value)
                self.cat_stack.append(AnnotCat.ClassAtribute)
            case cst.Lambda():
                self.path = self.get_path(SpecialNames.Lambda)
                self.cat_stack.append(AnnotCat.LocalVar)
        return super().on_visit(node)

    def on_leave(self, node):
        r = super().on_leave(node)
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.path = self.path.pop()
                self.cat_stack.pop()
        return r

    def _record_annot(
        self,
        name: str,
        cat: AnnotCat,
        annot: cst.Annotation | None,
    ) -> None:
        def fix_pos(pos: CodePosition):
            return CodePosition(pos.line, pos.column + 1)

        path = self.get_path(name)
        crange = None
        if annot:
            crange = self.get_metadata(PositionProvider, annot)
            # use 1-based indexing for columns
            crange = CodeRange(fix_pos(crange.start), fix_pos(crange.end))
        self.annots_info.append(AnnotInfo(path, cat, annot, crange))

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self._record_annot(SpecialNames.Return, AnnotCat.FuncReturn, node.returns)

    def visit_Param(self, node: cst.Param):
        if (name := node.name.value) != "self":
            self._record_annot(name, AnnotCat.FuncArg, node.annotation)

    def visit_AnnAssign(self, node: cst.AnnAssign):
        match node.target:
            case cst.Name(value=name):
                self._record_annot(name, self.cat_stack[-1], node.annotation)
            case cst.Attribute(value=cst.Name(value=l), attr=cst.Name(value=r)):
                self._record_annot(l + "." + r, AnnotCat.ClassAtribute, node.annotation)


class AnnotApplier(cst.CSTTransformer, CodePathManager):
    """Apply the annotations to the code. Note that each AnnotPath will be
    applied at most once."""

    def __init__(self, annots: Dict[AnnotPath, cst.Annotation]):
        super().__init__()
        CodePathManager.__init__(self)

        self.annots = annots.copy()

        # store the target prefixes
        self.prefixes: Set[AnnotPath] = {annot_path()}
        for path in annots.keys():
            while bool(path.value):
                self.prefixes.add(path)
                path = path.pop()

    def on_visit(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.path = self.get_path(node.name.value)
            case cst.Lambda():
                self.path = self.get_path(SpecialNames.Lambda)
        if self.path not in self.prefixes:
            return False
        return super().on_visit(node)

    def on_leave(self, node, updated):
        r = super().on_leave(node, updated)
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.path = self.path.pop()
        return r

    def leave_FunctionDef(
        self, node: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        path = self.get_path(SpecialNames.Return)
        if path in self.annots:
            rep = self.annots.pop(path)
            return updated.with_changes(returns=rep)
        else:
            return updated

    def leave_Param(self, node: cst.Param, updated: cst.Param) -> cst.Param:
        path = self.get_path(updated.name.value)
        if path in self.annots:
            rep = self.annots.pop(path)
            return updated.with_changes(annotation=rep)
        else:
            return updated

    def leave_AnnAssign(
        self, node: cst.AnnAssign, updated: cst.AnnAssign
    ) -> cst.AnnAssign:
        match updated.target:
            case cst.Name(value=name):
                key_name = name
            case cst.Attribute(value=cst.Name(value=l), attr=cst.Name(value=r)):
                key_name = l + "." + r
            case _:
                key_name = SpecialNames.Missing
        path = self.get_path(key_name)

        if path in self.annots:
            rep = self.annots.pop(path)
            return updated.with_changes(annotation=rep)
        else:
            return updated


class StmtsAdder(cst.CSTTransformer):
    def __init__(self, stmts: Sequence[cst.BaseStatement]):
        self.stmts = stmts

    def leave_Module(self, node: cst.Module, updated: cst.Module) -> cst.Module:

        body_type: Any = type(updated.body)
        return updated.with_changes(body=body_type(self.stmts) + updated.body)


AnyAnnot = cst.Annotation(cst.Name("Any"))
# A special annotation to signal that the type annotation is missing.
MissingAnnot = cst.Annotation(cst.Name("Missing"))

TypeExpr = cst.BaseExpression


@dataclass
class TypeInfState:
    """The current (partically annotated) CST"""

    module: cst.Module
    to_annot: dict[AnnotPath, AnnotCat]
    annotated: dict[AnnotPath, TypeExpr]
    num_errors: int

    def next_to_annot(self) -> AnnotPath:
        return next(iter(self.to_annot))

    def __str__(self):
        return f"""
num_errors: {self.num_errors}
num_to_annot: {len(self.to_annot)}
to_annotate: {self.to_annot}
------------------------ code -------------------------------
{self.module.code}
"""


class SelectAnnotations:
    @staticmethod
    def select_annotated(paths: list) -> list[AnnotInfo]:
        "Select all places with an existing type annotation"
        return [info for info in paths if info.annot is not None]

    @staticmethod
    def select_all_paths(paths: list, annotated: dict) -> list[AnnotInfo]:
        "Select all available places"
        return paths


@dataclass
class TypeInfAction:
    """Annotate a location with a type. The type will be converted to `Any` if it would trigger a type error."""

    path: AnnotPath
    type: TypeExpr

    def __repr__(self):
        type_ex = cst.Module([]).code_for_node(self.type)
        return f"TypeInfAction({str(self.path)} : {type_ex})"


class TypeInfEnv:
    """An environment for sequentially annotating a python source file."""

    SpecialComment: str = "# [SPOT Env]"
    Preamble = "from typing import *  # [SPOT Env]\nMissing=Any"

    def __init__(
        self,
        checker: IncrementalChekcer,
        src_file,
        select_annotations: Callable[..., list[AnnotInfo]],
        check_any=False,
        print_mypy_output=False,
    ):
        self.checker = checker
        self.src_file = realpath(src_file)
        self.original_src = read_file(src_file)
        self.check_any = check_any
        self.print_mypy_output = print_mypy_output
        if TypeInfEnv.SpecialComment in self.original_src:
            raise RuntimeError(
                f"The file {src_file} has already been modified by SPOT since it contains the special comment."
            )
        self.select_annotations = select_annotations
        self.state: TypeInfState = None  # type: ignore
        self.preamble = cst.parse_module(TypeInfEnv.Preamble).body

    def restore_file(self) -> None:
        """Restore the python source file to its original state."""
        write_file(self.src_file, self.original_src)

    def init_to_annot(self) -> dict[AnnotPath, AnnotCat]:
        module = cst.parse_module(self.original_src)
        paths, annots = collect_annots_info(module)
        to_annot = self.select_annotations(paths, annots)
        assert isinstance(to_annot, dict)
        return to_annot

    def reset(self) -> None:
        """Reset the environment to the initial state. This will remove some of the type annotations in the original source file."""
        self.restore_file()
        module = cst.parse_module(self.original_src)
        paths_info = collect_annots_info(module)
        to_annot: list[AnnotInfo] = self.select_annotations(paths_info)
        # to_remove = {info.path for info in paths_info if bool(info.annotation)}
        # module = apply_annotations(module, {p: MissingAnnot for p in to_remove})
        # module = add_stmts(module, self.preamble)  # add all the necessary imports
        # write_file(self.src_file, module.code)
        # annotated = {
        #     p: annots[p].annotation for p in annots.keys() if p not in to_remove
        # }
        # num_errors = self.checker.recheck_files(self.src_file).num_errors
        # self.state = TypeInfState(module, to_annot_cats, annotated, num_errors)
        # FIXME

    def step(self, action: TypeInfAction) -> bool:
        state = self.state
        assert state is not None, "Did you forget to call reset()?"
        assert (
            action.path in state.to_annot
        ), f"Invalid action: path {action.path} not in `to_annot`."
        type = action.type
        mod = apply_annotations(state.module, {action.path: cst.Annotation(type)})
        # time.sleep(1.0)
        write_file(self.src_file, mod.code)
        out = self.checker.recheck_files(self.src_file)
        if isinstance(out, str):
            raise RuntimeError(out)

        if self.print_mypy_output:
            print("action: ", action)
            print("mypy output:", out.output_str)

        rejected = out.num_errors > state.num_errors
        if rejected:
            type = cst.parse_expression(f"Annotated[Any, {mod.code_for_node(type)}]")
            mod = apply_annotations(mod, {action.path: cst.Annotation(type)})
            # time.sleep(1.0)
            write_file(self.src_file, mod.code)
            if self.check_any:
                out = self.checker.recheck_files(self.src_file)
                if isinstance(out, str):
                    raise RuntimeError(out)
                assert out.num_errors == state.num_errors, (
                    "Adding Any should not trigger more type errors.\n"
                    f"original errors: {state.num_errors}, new errors: {out.num_errors}\n"
                    f"action: {action}\n"
                    f"mypy output: {out.output_str}\n"
                    f"---------Code---------\n {mod.code}\n"
                )
        state.to_annot.pop(action.path)
        state.annotated[action.path] = type
        state.module = mod
        return rejected


@contextmanager
def type_inf_env(
    checker: IncrementalChekcer,
    src_file,
    select_annotations: Callable = SelectAnnotations.select_annotated,
    check_any=False,
    print_mypy_output=False,
):
    env = TypeInfEnv(
        checker,
        src_file,
        select_annotations,
        check_any=check_any,
        print_mypy_output=print_mypy_output,
    )
    try:

        env.reset()
        yield env
    finally:
        env.restore_file()


def test_inference_performance(src_root, src_files=None, silent=True):
    if src_files is None:
        src_files = list(src_root.glob("**/*.py"))

    iter_f = lambda x: x if silent else tqdm

    with mypy_checker(src_root) as checker:
        n_checks = 0
        t_s = time.time()
        for f in iter_f(src_files):  # type: ignore
            with type_inf_env(checker, f) as env:
                if len(env.init_to_annot()) == 0:
                    continue  # skip files with no annotations
                n_checks += 1
                while len(env.state.to_annot) > 0:
                    n_checks += 1
                    env.step(TypeInfAction(env.state.next_to_annot(), cst.Name("int")))
        t_e = time.time()
        return {"n_checks": n_checks, "time": t_e - t_s}
