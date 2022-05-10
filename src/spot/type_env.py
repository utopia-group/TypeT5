import ast
import enum
import re
import subprocess
import time
from argparse import ArgumentError
from dataclasses import dataclass, field
from distutils.log import error
from logging import warn
from os import PathLike
from posixpath import dirname, realpath
from typing import *
from unittest import case

from libcst.metadata import CodeRange, PositionProvider
from numpy import isin
from pyrsistent import plist
from pyrsistent.typing import PList
from tqdm import tqdm

from .utils import *


@enum.unique
class AnnotCat(enum.Enum):
    """The category of an annotation."""

    FuncArg = enum.auto()
    FuncReturn = enum.auto()
    ClassAtribute = enum.auto()
    GlobalVar = enum.auto()
    LocalVar = enum.auto()


# @dataclass(order=True, unsafe_hash=True)
# class AnnotSegment:
#     """A segment of an annotation path. The id records how many segments of the same name
#     has occurred before this segment."""

#     name: str
#     id: int

#     def __str__(self):
#         return self.name if self.id == 0 else f"{self.name}[{self.id}]"


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


def collect_annotations(
    code: cst.Module | cst.MetadataWrapper,
) -> list[AnnotInfo]:
    """Collect all annotation paths and the corresponding type annotations (if any).
    The order of the paths is the same as the order of the annotations in the code."""
    collector = AnnotCollector()
    if not isinstance(code, cst.MetadataWrapper):
        code = cst.MetadataWrapper(code)
    code.visit(collector)
    return collector.annots_info


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


@dataclass
class MypyResult:
    # total number of errors in all files
    num_errors: int
    # records the errors in each file and their locations
    error_dict: dict[Path, list[tuple[CodePosition, str]]]
    # the original output by mypy
    output_str: str


class MypyChecker:

    TypeCheckFlags = [
        "--follow-imports=skip",
        "--namespace-packages",
        "--allow-untyped-globals",
        "--explicit-package-bases",
        "--ignore-missing-imports",  # a hacky workaround
        "--allow-redefinition",
        "--show-column-numbers",
        "--show-error-codes",
        "--soft-error-limit=-1",
        "--config-file=",
    ]

    Preamble = "from typing import Any # SPOT\n"

    """Run Mypy daemon to (repeatedly) type check given files"""

    def __init__(self, dmypy_path, code_dir, wait_before_check=1.0) -> None:
        self.code_dir = realpath(code_dir)
        self.dmypy_path = realpath(dmypy_path)
        self.wait_before_check = wait_before_check
        # subprocess.run(
        #     ["python", self.dmypy_path, "run", "--", "."],
        #     capture_output=True,
        #     text=True,
        #     cwd=self.code_dir,
        # )
        subprocess.run(
            [
                "python",
                self.dmypy_path,
                "start",
                "--",
                # "--check-untyped-defs",  # turn off to improve performance
                *MypyChecker.TypeCheckFlags,
            ],
            cwd=self.code_dir,
        )
        subprocess.run(
            ["python", self.dmypy_path, "check", self.code_dir],
            cwd=self.code_dir,
            capture_output=True,
        )

    def close(self):
        subprocess.run(
            ["python", self.dmypy_path, "stop"],
            cwd=self.code_dir,
        )

    def check_code_dir(self) -> MypyResult:
        return self._run_mypy(["python", self.dmypy_path, "check", self.code_dir])

    @staticmethod
    def check_project(proj: Path, mypy_path: Path = None):
        if mypy_path is None:
            mypy_path = proj_root() / ".venv/bin/mypy"
        cmd = [
            "python",
            str(mypy_path),
            ".",
            *MypyChecker.TypeCheckFlags,
        ]
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=proj,
        )
        return MypyChecker.parse_mypy_output(out, cmd, cwd=proj)

    @staticmethod
    def parse_mypy_output(
        output: subprocess.CompletedProcess[str],
        cmd: list[str],
        cwd: Path,
    ) -> MypyResult:
        lines = output.stdout.splitlines()
        assert (
            len(lines) > 0
        ), f"mypy failed. Command: `{' '.join(cmd)}`\nError: {output.stderr}"
        error_dict: dict[Path, list[tuple[CodePosition, str]]] = {}
        for l in lines:
            m = re.match(r"(.*\.py):(\d+:\d+): error: (.+) \[[a-z\-]+\]", l)
            if m is not None:
                file = Path(cwd / m.group(1)).resolve()
                line, col = map(int, m.group(2).split(":"))
                error = m.group(3)
                if file not in error_dict:
                    error_dict[file] = []
                error_dict[file].append((CodePosition(line, col), error))

        m = re.match(r"Found (\d+) errors? in", lines[-1])
        if m is None:
            num_errors = 0
        else:
            num_errors = int(m.group(1))

        total_errors = sum(map(len, error_dict.values()))
        # unfortunately, some errors do not have a column number.
        # assert (
        #     num_errors == total_errors
        # ), f"{num_errors} != {total_errors}. errors found: {error_dict.values()}\n mypy output: \n{output.stdout}"
        return MypyResult(num_errors, error_dict, output.stdout)

    def recheck_files(self, *updated_files: str) -> MypyResult:
        # TODO: remove this workaround once (https://github.com/python/mypy/issues/12697) is fixed.
        time.sleep(
            self.wait_before_check
        )  # wait to make sure the type checker sees the file changes

        out = self._run_mypy(
            [
                "python",
                self.dmypy_path,
                "recheck",
                # "--perf-stats-file",
                # "mypy_perf.json",
                "--update",
                *updated_files,
            ]
        )

        # TODO: remove below after fixing the issue
        # subprocess.run(
        #     ["cat", "mypy_perf.json"],
        #     cwd=self.code_dir,
        # )
        # subprocess.run(
        #     [
        #         "python",
        #         self.dmypy_path,
        #         "status",
        #         "--fswatcher-dump-file",
        #         "mypy_fswatcher.json",
        #     ],
        #     cwd=self.code_dir,
        # )
        # subprocess.run(
        #     ["cat", "mypy_fswatcher.json"],
        #     cwd=self.code_dir,
        # )

        return out

    def _run_mypy(self, cmd: list[str]) -> MypyResult:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.code_dir,
        )
        return MypyChecker.parse_mypy_output(result, cmd)


@contextmanager
def mypy_checker(code_dir: Path, dmypy_path: Path = None, wait_before_check=1.0):
    try:
        if dmypy_path is None:
            dmypy_path = proj_root() / ".venv/bin/dmypy"
        yield (
            checker := MypyChecker(
                dmypy_path, code_dir, wait_before_check=wait_before_check
            )
        )
    finally:
        checker.close()


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
        checker: MypyChecker,
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
        paths, annots = collect_annotations(module)
        to_annot = self.select_annotations(paths, annots)
        assert isinstance(to_annot, dict)
        return to_annot

    def reset(self) -> None:
        """Reset the environment to the initial state. This will remove some of the type annotations in the original source file."""
        self.restore_file()
        module = cst.parse_module(self.original_src)
        paths_info = collect_annotations(module)
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
    checker: MypyChecker,
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
        for f in iter_f(src_files):
            with type_inf_env(checker, f) as env:
                if len(env.init_to_annot()) == 0:
                    continue  # skip files with no annotations
                n_checks += 1
                while len(env.state.to_annot) > 0:
                    n_checks += 1
                    env.step(TypeInfAction(env.state.to_annot[0], cst.Name("int")))
        t_e = time.time()
        return {"n_checks": n_checks, "time": t_e - t_s}


@dataclass(unsafe_hash=True, order=True)
class PythonType:
    head: tuple[str, ...]
    args: tuple["PythonType", ...] = ()

    def __str__(self):
        h = ".".join(self.head)
        if self.args:
            return f"{h}[{', '.join(map(str, self.args))}]"
        else:
            return h

    def __repr__(self):
        return self.__str__()

    def all_heads(self):
        """Return an iterator of all the type heads."""
        yield self.head
        for arg in self.args:
            yield from arg.all_heads()

    def head_name(self) -> str:
        """Return the last part of the type head."""
        if self.head == ():
            return SpecialNames.Empty
        else:
            return self.head[-1]

    def is_union(self) -> bool:
        """Check whether the type is a union type."""
        return self.head_name() == "Union"

    @staticmethod
    def Any() -> "PythonType":
        return PythonType(("Any",))


_type_name_map = {
    "list": "List",
    "tuple": "Tuple",
    "dict": "Dict",
    "set": "Set",
}


def normalize_type_name(name: str) -> str:
    return _type_name_map.get(name, name)


def normalize_type_head(head: tuple[str, ...]) -> tuple[str, ...]:
    n = len(head)
    if n == 0:
        return head
    return (*head[0 : n - 1], normalize_type_name(head[n - 1]))


def normalize_type(typ: PythonType) -> PythonType:
    n_args = map(normalize_type, typ.args)
    if typ.is_union():
        arg_set = set[PythonType]()
        for arg in n_args:
            if arg.is_union():
                arg_set.update(arg.args)
            else:
                arg_set.add(arg)
        return PythonType(("Union",), tuple(arg_set))
    return PythonType(
        normalize_type_head(typ.head),
        tuple(n_args),
    )


def parse_type_str(typ_str: str) -> PythonType:
    tree = ast.parse(typ_str, mode="eval").body
    ty = parse_type_from_ast(tree)
    assert isinstance(ty, PythonType), f"{ty} is not a PythonType"
    return ty


def parse_type_expr(
    m: cst.Module, annot: cst.BaseExpression, silent=False
) -> PythonType | None:
    code = m.code_for_node(annot)
    code = re.sub(r"#.*\n", "", code).replace("\n", "")
    try:
        return parse_type_str(code)
    except Exception as e:
        if silent:
            return None
        else:
            print(f"Failed to parse type expression: `{code}` in source module:")
            print(m.code)
            raise e


def parse_type_from_ast(tree: ast.expr) -> PythonType:
    match tree:
        case ast.Name() | ast.Attribute():
            return PythonType(parse_qualified_name(tree))
        case ast.Constant(value=str() as s):
            return parse_type_from_ast(ast.parse(s, mode="eval").body)
        case ast.Constant(value=v):
            if v == None:
                return PythonType(("None",))
            elif v == (...):
                return PythonType(("...",))
            else:
                return PythonType((str(v),))
        case ast.List(elts=elts):  # this can happen inside Callable
            args = tuple(map(parse_type_from_ast, elts))
            return PythonType((), args)
        case ast.Subscript(value=(ast.Attribute() | ast.Name()) as v, slice=slice):
            head = parse_qualified_name(v)
            if head[-1] == "Literal":
                return PythonType(head)  # discard the parameters
            match slice:
                case ast.Tuple(elts=elts):
                    args = tuple(map(parse_type_from_ast, elts))
                case _:
                    args = (parse_type_from_ast(slice),)
            return PythonType(head, args)
        case ast.BinOp(left=left, right=right, op=ast.BitOr()):
            return PythonType(
                ("Union",), (parse_type_from_ast(left), parse_type_from_ast(right))
            )
        case ast.Call():
            return PythonType(("[FuncCall]",))
        case ast.Tuple(elts=elts):
            return PythonType(("Tuple",), tuple(map(parse_type_from_ast, elts)))
        case _:
            raise ArgumentError(
                None, f"Unsupported ast type: {ast.dump(tree, include_attributes=True)}"
            )


def parse_qualified_name(tree: ast.Attribute | ast.Name):
    segs = []
    while isinstance(tree, ast.Attribute):
        segs.append(tree.attr)
        tree = tree.value  # type: ignore
    assert isinstance(tree, ast.Name)
    segs.append(tree.id)
    return tuple(reversed(segs))
