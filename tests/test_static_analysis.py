import spot
from spot.static_analysis import (
    FunctionUsage,
    ProjectPath,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    cst,
)
import pytest

from spot.utils import assert_eq


to_abs = PythonProject.to_abs_import_path


def test_import_normalization():
    assert to_abs("spot.static_analysis", ".") == "spot"
    assert to_abs("spot.static_analysis", "..") == ""
    assert to_abs("spot.static_analysis", ".utils") == "spot.utils"
    assert to_abs("spot.static_analysis", ".utils.a") == "spot.utils.a"
    assert to_abs("spot.static_analysis", "foo.bar") == "foo.bar"

    with pytest.raises(AssertionError):
        to_abs("spot.static_analysis", "...")


def test_usage_analysis():
    code1 = """
# root.file1

# global function
def gf(x):
    return x * x

# with inner function
def gf_with_inner(x):
    def inner(y):
        return y * y
    return inner(x)

# class
class C:
    def __init__(self, x):
        self.x = x
    
    def foo(self, y):
        return self.x + y
    
"""
    code2 = """
# root.file2
from .file1 import gf
from root.file1 import gf_with_inner
import root.file1
import root.file1 as f1

def usage1(x):
    gf(x) + root.file1.C(5)
    foo(5)

def usage2(x):
    def inner():
        1 + gf_with_inner(x)
    return inner()

def usage_method1(x):
    x = f1.C(5)
    1 + x.foo(3)

def usage_method2(x):
    (1 + f1.C(5)).foo(3)

def usage_local():
    usage1(3)
    UsageClass(4)

class UsageClass:
    def __init__(self, x):
        self.x = gf_with_inner(x)
        self.foo(5)

    def foo(self, y):
        return usage_local(f1.gf(y))
"""

    project = PythonProject.from_modules(
        [
            PythonModule.from_cst(cst.parse_module(code1), "root.file1"),
            PythonModule.from_cst(cst.parse_module(code2), "root.file2"),
        ]
    )

    analysis = UsageAnalysis(project)

    def assert_usages(caller: str, *callees: tuple[str, bool]):
        caller_p = ProjectPath.from_str(caller)
        expect = set()
        for callee, certain in callees:
            callee_p = ProjectPath.from_str(callee)
            expect.add((callee_p, certain))

        actual = {
            (u.callee, u.is_certain)
            for u in analysis.caller2callees.get(caller_p, list())
        }

        assert_eq(actual, expect)

    assert_usages(
        "root.file2/usage1",
        ("root.file1/gf", True),
        ("root.file1/C.__init__", True),
    )

    assert_usages(
        "root.file2/usage2",
        ("root.file1/gf_with_inner", True),
    )

    assert_usages(
        "root.file2/usage_method1",
        ("root.file1/C.__init__", True),
        ("root.file1/C.foo", False),
        ("root.file2/UsageClass.foo", False),
    )

    assert_usages(
        "root.file2/usage_method2",
        ("root.file1/C.__init__", True),
        ("root.file1/C.foo", False),
        ("root.file2/UsageClass.foo", False),
    )

    assert_usages(
        "root.file2/usage_local",
        ("root.file2/usage1", True),
        ("root.file2/UsageClass.__init__", True),
    )

    assert_usages(
        "root.file2/UsageClass.__init__",
        ("root.file1/gf_with_inner", True),
        ("root.file2/UsageClass.foo", True),
    )

    assert_usages(
        "root.file2/UsageClass.foo",
        ("root.file2/usage_local", True),
        ("root.file1/gf", True),
    )
