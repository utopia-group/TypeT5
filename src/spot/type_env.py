from argparse import ArgumentError
from os import PathLike
from unittest import case
from dataclasses import dataclass, field
from distutils.log import error
from logging import warn
from posixpath import dirname, realpath
import re
from typing import *
from decorator import contextmanager
from numpy import isin
from pyrsistent import plist
from pyrsistent.typing import PList
import time
from tqdm import tqdm
import ast

from libcst import SimpleWhitespace
from .utils import *
import subprocess

# This is supposed to be immutable, but setting `frozen=True` would cause notebook auto-reload
# to fail
@dataclass(order=True, unsafe_hash=True)
class AnnotPath:
    """The unique path of a type annoation."""

    value: PList[str]

    def __repr__(self):
        return f"AnnotPath({self.__str__()})"

    def __str__(self):
        return f"'{'.'.join(self.value.reverse())}'"

    def append(self, seg: str) -> "AnnotPath":
        return AnnotPath(self.value.cons(seg))

    def pop(self) -> "AnnotPath":
        return AnnotPath(self.value.rest)


def annot_path(*segs: str) -> AnnotPath:
    return AnnotPath(plist(segs, reverse=True))


def collect_annotations(
    code: cst.CSTNode,
) -> Tuple[list[AnnotPath], dict[AnnotPath, cst.Annotation]]:
    """Collect all annotation paths and the corresponding type annotations (if any).
    The order of the paths is the same as the order of the annotations in the code."""
    collector = AnnotCollector()
    code.visit(collector)
    return collector.annot_paths, collector.annotations


def apply_annotations(
    code: cst.Module,
    annots: dict[AnnotPath, cst.Annotation],
):
    try:
        return code.visit(AnnotApplier(annots))
    except Exception as e:
        raise RuntimeError(f"Failed to apply annotations for the following code:\n {code.code}") from e


def add_imports(
    code: cst.Module,
    imports: list[Tuple[str, str]],
):
    code = code.visit(ImportsAdder(imports))
    return code


class AnnotCollector(cst.CSTVisitor):
    def __init__(self):
        self.path: AnnotPath = annot_path()
        # store the type annotations
        self.annot_paths: List[AnnotPath] = []
        self.annotations: Dict[AnnotPath, cst.Annotation] = {}

    def on_visit(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.path = self.path.append(node.name.value)
            case cst.Lambda():
                self.path = self.path.append(SpecialNames.Lambda)
        return super().on_visit(node)

    def on_leave(self, node):
        r = super().on_leave(node)
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.path = self.path.pop()
        return r

    def _record_annot(self, name: str, annot: cst.Annotation | None):
        path = self.path.append(name)
        self.annot_paths.append(path)
        if annot is not None:
            self.annotations[path] = annot

    def leave_FunctionDef(self, node: cst.FunctionDef):
        self._record_annot(SpecialNames.Return, node.returns)

    def visit_Param(self, node: cst.Param):
        if (name := node.name.value) != "self":
            self._record_annot(name, node.annotation)

    def visit_AnnAssign(self, node: cst.AnnAssign):
        match node.target:
            case cst.Name(value=name):
                self._record_annot(name, node.annotation)
            case cst.Attribute(value=cst.Name(value=l), attr=cst.Name(value=r)):
                self._record_annot(l + "." + r, node.annotation)


class AnnotApplier(cst.CSTTransformer):
    def __init__(self, annots: Dict[AnnotPath, cst.Annotation]):
        self.annots = annots
        self.path: AnnotPath = annot_path()

        # store the target prefixes
        self.prefixes: Set[AnnotPath] = {annot_path()}
        for path in annots.keys():
            while bool(path.value):
                self.prefixes.add(path)
                path = path.pop()

    def on_visit(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.path = self.path.append(node.name.value)
            case cst.Lambda():
                self.path = self.path.append(SpecialNames.Lambda)
        if self.path not in self.prefixes:
            return False
        return super().on_visit(node)

    def on_leave(self, node, updated):
        r = super().on_leave(node, updated)
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.path = self.path.pop()
        return r

    def _get_annot(self, name: str) -> AnnotPath:
        return self.path.append(name)

    def leave_FunctionDef(
        self, node: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        path = self._get_annot(SpecialNames.Return)
        if path in self.annots:
            return updated.with_changes(returns=self.annots[path])
        else:
            return updated

    def leave_Param(self, node: cst.Param, updated: cst.Param) -> cst.Param:
        path = self._get_annot(updated.name.value)
        if path in self.annots:
            return updated.with_changes(annotation=self.annots[path])
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
        path = self._get_annot(key_name)

        if path in self.annots:
            return updated.with_changes(annotation=self.annots[path])
        else:
            return updated


class ImportsAdder(cst.CSTTransformer):
    SpecialComment: str = "# [added by SPOT]"

    def __init__(self, imports: list[Tuple[str, str]]):
        self.imports = imports

    def leave_Module(self, node: cst.Module, updated: cst.Module) -> cst.Module:
        body_type: Any = type(updated.body)
        import_stmts = body_type(
            cst.SimpleStatementLine(
                [
                    cst.ImportFrom(
                        module=cst.Name(m), names=[cst.ImportAlias(name=cst.Name(v))]
                    )
                ],
                trailing_whitespace=cst.TrailingWhitespace(
                    SimpleWhitespace("  "),
                    comment=cst.Comment(ImportsAdder.SpecialComment),
                ),
            )
            for (m, v) in self.imports
        )
        return updated.with_changes(body=import_stmts + updated.body)


@dataclass
class MypyResult:
    num_errors: int  # total number of errors in all files
    num_error_dict: Dict[str, int]  # records the number of errors in each file
    output_str: str


class MypyChecker:
    """Run Mypy daemon to (repeatedly) type check given files"""

    def __init__(self, dmypy_path, code_dir) -> None:
        self.code_dir = realpath(code_dir)
        self.dmypy_path = realpath(dmypy_path)
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
                "--follow-imports=skip",
                "--namespace-packages",
                "--allow-untyped-globals",
                "--explicit-package-bases",
                "--ignore-missing-imports",  # a hacky workaround
                "--allow-redefinition",
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

    def recheck_files(self, *updated_files: str) -> MypyResult:
        return self._run_mypy(
            ["python", self.dmypy_path, "recheck", "--update", *updated_files]
        )

    def _run_mypy(self, cmd: list[str]) -> MypyResult:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.code_dir,
        )
        lines = result.stdout.splitlines()
        assert (
            len(lines) > 0
        ), f"mypy failed. Command: `{' '.join(cmd)}`\nError: {result.stderr}"
        num_error_dict: dict[str, int] = {}
        for l in lines:
            m = re.match(r"(.*\.py):[\d+:]* error: .+", l)
            if m is not None:
                num_error_dict[m.group(1)] = num_error_dict.get(m.group(1), 0) + 1

        m = re.match(r"Found (\d+) errors? in", lines[-1])
        if m is None:
            num_errors = 0
        else:
            num_errors = int(m.group(1))

        total_errors = sum(num_error_dict.values())
        assert (
            num_errors == total_errors
        ), f"{num_errors} != {total_errors}. mypy output: {result.stdout}"
        return MypyResult(num_errors, num_error_dict, result.stdout)


@contextmanager
def mypy_checker(code_dir: Path, dmypy_path: Path = None):
    if dmypy_path is None:
        dmypy_path = proj_root() / ".venv/bin/dmypy"
    yield (checker := MypyChecker(dmypy_path, code_dir))
    checker.close()


AnyType = cst.Name("Any")
AnyAnnot = cst.Annotation(AnyType)


TypeExpr = cst.BaseExpression


@dataclass
class TypeInfState:
    """The current (partically annotated) CST"""

    module: cst.Module
    to_annot: list[AnnotPath]
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
    def select_annotated(paths, annotated) -> list[AnnotPath]:
        "Select all places with an existing type annotation"
        return list(annotated.keys())

    @staticmethod
    def select_all_paths(paths, annotated) -> list[AnnotPath]:
        "Select all places with an existing type annotation"
        return paths


@dataclass
class TypeInfAction:
    """Annotate a location with a type. The type will be converted to `Any` if it would trigger a type error."""

    path: AnnotPath
    type: TypeExpr


class TypeInfEnv:
    """An environment for sequentially annotating a python source file."""

    def __init__(
        self,
        checker: MypyChecker,
        src_file,
        select_annotations: Callable,
    ):
        self.checker = checker
        self.src_file = realpath(src_file)
        self.original_src = read_file(src_file)
        if ImportsAdder.SpecialComment in self.original_src:
            raise RuntimeError(
                f"The file {src_file} has already been modified by SPOT since it contains the special comment."
            )
        self.select_annotations = select_annotations
        self.state: TypeInfState = None  # type: ignore

    def restore_file(self) -> None:
        """Restore the python source file to its original state."""
        write_file(self.src_file, self.original_src)

    def to_annot(self):
        module = cst.parse_module(self.original_src)
        paths, annots = collect_annotations(module)
        to_annot: list[AnnotPath] = self.select_annotations(paths, annots)
        assert isinstance(to_annot, list)
        return to_annot

    def reset(self) -> None:
        """Reset the environment to the initial state. This will remove some of the type annotations in the original source file."""
        self.restore_file()
        module = cst.parse_module(self.original_src)
        paths, annots = collect_annotations(module)
        to_annot: list[AnnotPath] = self.select_annotations(paths, annots)
        to_remove = {p for p in annots.keys() if p in to_annot}
        module = apply_annotations(module, {p: AnyAnnot for p in to_remove})
        module = add_imports(
            module, [("typing", "Any")]
        )  # add all the necessary imports
        write_file(self.src_file, module.code)
        annotated = {
            p: annots[p].annotation for p in annots.keys() if p not in to_remove
        }
        num_errors = self.checker.recheck_files(self.src_file).num_errors
        self.state = TypeInfState(module, to_annot, annotated, num_errors)

    def step(self, action: TypeInfAction, check_any=False) -> None:
        state = self.state
        assert state is not None, "Did you forget to call reset()?"
        assert (
            action.path in state.to_annot
        ), f"Invalid action: path {action.path} not in `to_annot`."
        type = action.type
        mod = apply_annotations(state.module, {action.path: cst.Annotation(type)})
        write_file(self.src_file, mod.code)
        ne = self.checker.recheck_files(self.src_file).num_errors
        if ne > state.num_errors:
            type = cst.Name("Any")
            mod = apply_annotations(state.module, {action.path: cst.Annotation(type)})
            write_file(self.src_file, mod.code)
            if check_any:
                check_r = self.checker.recheck_files(self.src_file)
                assert check_r.num_errors == state.num_errors, (
                    "Adding Any should not trigger more type errors.\n"
                    f"action: {action}\n"
                    f"mypy output: {check_r.output_str}\n"
                    f"---------code---------\n {mod.code}\n"
                )
        state.to_annot.remove(action.path)
        state.annotated[action.path] = type
        state.module = mod


@contextmanager
def type_inf_env(
    checker: MypyChecker,
    src_file,
    select_annotations: Callable = SelectAnnotations.select_annotated,
):
    env = TypeInfEnv(checker, src_file, select_annotations)
    env.reset()
    yield env
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
                if len(env.to_annot()) == 0:
                    continue  # skip files with no annotations
                n_checks += 1
                while len(env.state.to_annot) > 0:
                    n_checks += 1
                    env.step(TypeInfAction(env.state.to_annot[0], cst.Name("int")))
        t_e = time.time()
        return {"n_checks": n_checks, "time": t_e-t_s}


@dataclass(unsafe_hash=True)
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


def parse_type_expr(
    m: cst.Module, annot: cst.BaseExpression, silent=False
) -> PythonType | None:
    code = m.code_for_node(annot)
    code = re.sub(r"#.*\n", "", code).replace("\n", "")
    try:
        tree = ast.parse(code, mode="eval").body
        return parse_type_from_ast(tree)
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
