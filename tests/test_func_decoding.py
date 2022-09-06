from pathlib import Path
from spot.utils import *

from spot.static_analysis import (
    ModuleHierarchy,
    ProjectPath,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    build_project_namespaces,
    cst,
    stub_from_module,
    to_abs_import_path as to_abs,
)
from spot.function_dataset import FunctionSignature

import pytest
import copy

from spot.utils import assert_eq, groupby, not_none, show_string_diff


def test_function_signature():
    ex_code = """
def f(x, y: int=3, *, v=3, **kwargs) -> int:
    u: int
    return 1
"""

    f = cast(cst.FunctionDef, cst.parse_module(ex_code).body[0])
    sig = FunctionSignature.from_function(f)

    new_sig = copy.deepcopy(sig)
    new_sig.annots[0] = cst.Annotation(cst.parse_expression("list[int]"))
    assert (
        "def f(x: list[int], y: int=3, *, v=3, **kwargs) -> int"
        in cst.Module([new_sig.apply(f)]).code
    )

    new_sig = copy.deepcopy(sig)
    new_sig.annots[1] = cst.Annotation(cst.parse_expression("list[int]"))
    assert (
        "def f(x, y: list[int]=3, *, v=3, **kwargs) -> int"
        in cst.Module([new_sig.apply(f)]).code
    )

    new_sig = copy.deepcopy(sig)
    new_sig.annots[2] = cst.Annotation(cst.parse_expression("list[int]"))
    assert (
        "def f(x, y: int=3, *, v: list[int]=3, **kwargs) -> int"
        in cst.Module([new_sig.apply(f)]).code
    )

    new_sig = copy.deepcopy(sig)
    new_sig.annots[3] = cst.Annotation(cst.parse_expression("list[int]"))
    assert (
        "def f(x, y: int=3, *, v=3, **kwargs: list[int]) -> int"
        in cst.Module([new_sig.apply(f)]).code
    )

    new_sig = copy.deepcopy(sig)
    new_sig.annots[4] = cst.Annotation(cst.parse_expression("list[int]"))
    assert (
        "def f(x, y: int=3, *, v=3, **kwargs) -> list[int]"
        in cst.Module([new_sig.apply(f)]).code
    )


def test_method_signature():
    ex_code = """
def f(self, x, y):
    u: int
    return 1
"""

    f = cast(cst.FunctionDef, cst.parse_module(ex_code).body[0])
    sig = FunctionSignature.from_function(f)
    assert len(sig.annots) == 3

    ex_code2 = """
def f(a, x, y):
    u: int
    return 1
"""

    f = cast(cst.FunctionDef, cst.parse_module(ex_code2).body[0])
    sig = FunctionSignature.from_function(f)
    assert len(sig.annots) == 4

    ex_code3 = """
def f(a=lambda x: x):
    return 1
"""

    f = cast(cst.FunctionDef, cst.parse_module(ex_code3).body[0])
    sig = FunctionSignature.from_function(f)
    assert len(sig.annots) == 2
