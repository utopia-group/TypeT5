from os import PathLike
from unittest import case
from dataclasses import dataclass
from distutils.log import error
from logging import warn
from posixpath import dirname, realpath
import re
from typing import *
from decorator import contextmanager

from libcst import SimpleWhitespace
from .utils import *
import subprocess


# This is supposed to be immutable, but setting `frozen=True` would cause notebook auto-reload
# to fail
@dataclass(order=True, unsafe_hash=True)
class AnnotPath:
    """The unique path of a type annoation."""

    value: Tuple[str, ...]

    def __repr__(self):
        return f"AnnotPath('{'.'.join(self.value)}')"

    def __str__(self):
        return f"'{'.'.join(self.value)}'"


def annot_path(*segs: str) -> AnnotPath:
    return AnnotPath(tuple(segs))


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
    return code.visit(AnnotApplier(annots))


def add_imports(
    code: cst.Module,
    imports: list[Tuple[str, str]],
):
    code = code.visit(ImportsAdder(imports))
    return code


class AnnotCollector(cst.CSTVisitor):
    def __init__(self):
        # stack for storing the canonical name of the current function
        self.stack: List[str] = []
        # store the type annotations
        self.annot_paths: List[AnnotPath] = []
        self.annotations: Dict[AnnotPath, cst.Annotation] = {}

    def on_visit(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.stack.append(node.name.value)
        return super().on_visit(node)

    def on_leave(self, node):
        r = super().on_leave(node)
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.stack.pop()
        return r

    def _current_path(self):
        return AnnotPath(tuple(self.stack))

    def _record_annot(self, name: str, annot: cst.Annotation | None):
        self.stack.append(name)
        path = self._current_path()
        self.stack.pop()
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
        # stack for storing the canonical name of the current function
        self.stack: List[str] = []
        # store the target prefixes
        self.prefixes: Set[Tuple[str, ...]] = set()
        for path in annots.keys():
            self.prefixes.update(path.value[0:i] for i in range(len(path.value) + 1))

    def _current_path(self):
        return AnnotPath(tuple(self.stack))

    def on_visit(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.stack.append(node.name.value)
        if tuple(self.stack) not in self.prefixes:
            return False
        return super().on_visit(node)

    def on_leave(self, node, updated):
        r = super().on_leave(node, updated)
        match node:
            case cst.ClassDef() | cst.FunctionDef():
                self.stack.pop()
        return r

    def _get_annot(self, name: str) -> AnnotPath:
        self.stack.append(name)
        path = self._current_path()
        self.stack.pop()
        return path

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
