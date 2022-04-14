from dataclasses import dataclass
from distutils.log import error
from posixpath import dirname, realpath
import re
from typing import Iterable
from .utils import *
import subprocess


@dataclass(frozen=True)
class AnnotPath:
    """The unique path of a type annoation."""

    value: Tuple[str, ...]

def annot_path(*segs: str) -> AnnotPath:
    return AnnotPath(tuple(segs))

def collect_annotations(code: cst.CSTNode) -> Dict[AnnotPath, Optional[cst.Annotation]]:
    collector = AnnotCollector()
    code.visit(collector)
    return collector.annotations


def apply_annotations(code: cst.CSTNode, annots: Dict[AnnotPath, cst.Annotation]):
    applier = AnnotApplier(annots)
    return code.visit(applier)


class AnnotCollector(cst.CSTVisitor):
    def __init__(self):
        # stack for storing the canonical name of the current function
        self.stack: List[str] = []
        # store the type annotations
        self.annotations: Dict[AnnotPath, Optional[cst.Annotation]] = {}

    def on_visit(self, node):
        if (
            isinstance(node, cst.FunctionDef)
            or isinstance(node, cst.ClassDef)
            or isinstance(node, cst.Param)
        ):
            self.stack.append(node.name.value)
        elif isinstance(node, cst.AnnAssign):
            self.stack.append(node.target.value)
        return super().on_visit(node)

    def on_leave(self, node):
        r = super().on_leave(node)
        if (
            isinstance(node, cst.FunctionDef)
            or isinstance(node, cst.ClassDef)
            or isinstance(node, cst.Param)
            or isinstance(node, cst.AnnAssign)
        ):
            self.stack.pop()
        return r

    def _current_path(self):
        return AnnotPath(tuple(self.stack))

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.stack.append(SpecialNames.Return)
        self.annotations[self._current_path()] = node.returns
        self.stack.pop()

    def visit_Param(self, node: cst.Param):
        self.annotations[self._current_path()] = node.annotation

    def visit_AnnAssign(self, ndoe: cst.AnnAssign):
        self.annotations[self._current_path()] = ndoe.annotation


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
        if (
            isinstance(node, cst.FunctionDef)
            or isinstance(node, cst.ClassDef)
            or isinstance(node, cst.Param)
        ):
            self.stack.append(node.name.value)
        elif isinstance(node, cst.AnnAssign):
            self.stack.append(node.target.value)
        if tuple(self.stack) not in self.prefixes:
            return False
        return super().on_visit(node)

    def on_leave(self, node, updated):
        r = super().on_leave(node, updated)
        if (
            isinstance(node, cst.FunctionDef)
            or isinstance(node, cst.ClassDef)
            or isinstance(node, cst.Param)
            or isinstance(node, cst.AnnAssign)
        ):
            self.stack.pop()
        return r

    def leave_FunctionDef(
        self, node: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.stack.append(SpecialNames.Return)
        patch = self.annots.get(self._current_path())
        self.stack.pop()
        return updated if patch is None else updated.with_changes(returns=patch)

    def leave_Param(self, node: cst.Param, updated: cst.Param) -> cst.Param:
        patch = self.annots.get(self._current_path())
        return updated if patch is None else updated.with_changes(annotation=patch)


class MypyChecker:
    """Run Mypy daemon to (repeatedly) type check given files"""

    def __init__(self, dmypy_path, code_dir) -> None:
        self.code_dir = realpath(code_dir)
        self.dmypy_path = realpath(dmypy_path)
        self.mypy_version = subprocess.run(
            ["python", self.dmypy_path, "-V"],
            capture_output=True,
            text=True,
            cwd=self.code_dir,
        ).stdout

    def stop_daemon(self):
        subprocess.run(
            ["python", self.dmypy_path, "stop"],
            cwd=self.code_dir,
        )

    def __del__(self):
        self.stop_daemon()

    def check_file(self, fpath):
        print("chekcing!")
        cmd = ["python", self.dmypy_path, "run", "--", fpath]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.code_dir,
        )
        # retcode = result.returncode
        lines = result.stdout.splitlines()
        assert len(lines) > 0, f"mypy failed. Error: {result.stderr}"
        num_error_dict = {}
        for l in lines:
            m = re.match(r"(.*\.py):\d+: error: .+", l)
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


@dataclass
class MypyResult:
    num_errors: int  # total number of errors in all files
    num_error_dict: Dict[str, int]  # records the number of errors in each file
    output_str: str
