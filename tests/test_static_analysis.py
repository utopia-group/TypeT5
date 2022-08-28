from pathlib import Path
import spot
from spot.static_analysis import (
    ModuleHierarchy,
    ProjectPath,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    cst,
    to_abs_import_path as to_abs,
)
import pytest

from spot.utils import assert_eq


def test_path_to_module():
    to_module = PythonProject.rel_path_to_module_name
    assert to_module(Path("a/b.py")) == "a.b"
    assert to_module(Path("a/b/c.py")) == "a.b.c"
    assert to_module(Path("a/__init__.py")) == "a"
    assert to_module(Path("a/b/__init__.py")) == "a.b"
    assert to_module(Path("src/a.py")) == "a"
    assert to_module(Path("src/a/__init__.py")) == "a"


def test_import_normalization():
    assert to_abs("spot.static_analysis", ".") == "spot"
    assert to_abs("spot.static_analysis", "..") == ""
    assert to_abs("spot.static_analysis", ".utils") == "spot.utils"
    assert to_abs("spot.static_analysis", ".utils.a") == "spot.utils.a"
    assert to_abs("spot.static_analysis", "foo.bar") == "foo.bar"

    with pytest.raises(AssertionError):
        to_abs("spot.static_analysis", "...")


def test_namespace_resolution():
    modules = [
        "foo",
        "foo.bar",
        "z",
    ]
    namespace = ModuleHierarchy.from_modules(modules)

    with pytest.raises(AssertionError):
        namespace.resolve_path(["foo"])
    with pytest.raises(AssertionError):
        namespace.resolve_path(["z"])

    assert namespace.resolve_path(["z", "x"]) == ProjectPath("z", "x")
    assert namespace.resolve_path(["foo", "a"]) == ProjectPath("foo", "a")
    assert namespace.resolve_path(["foo", "bar"]) == ProjectPath("foo", "bar")
    assert namespace.resolve_path(["foo", "C", "x"]) == ProjectPath("foo", "C.x")
    assert namespace.resolve_path(["foo", "bar", "C"]) == ProjectPath("foo.bar", "C")
    assert namespace.resolve_path(["foo", "bar", "C", "x"]) == ProjectPath(
        "foo.bar", "C.x"
    )


def test_module_imports():
    import_code = """
import A
import B.C
from D import a, b as c
from .utils import x
from ..top import *
from infer.type import *   
"""
    mod = PythonModule.from_cst(cst.parse_module(import_code), "root.file1")
    assert mod.imported_modules == {"A", "B.C", "D", "root.utils", "top", "infer.type"}


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

    @staticmethod
    def s_method(x):
        return x + 1
    
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

@f1.C(1)
def usage_dec():
    pass

class UsageClass:
    def __init__(self, x):
        self.x = gf_with_inner(x)
        self.y = self.foo(5)

    def foo(self, y):
        return usage_local(f1.gf(y))

    @staticmethod
    def s_method(x):
        return x

class SubClass(UsageClass):
    def use(self):
        self.foo(5)
        f1.C.s_method(5)

from root.file1 import nonexist

def use_nonexist():
    nonexist(5)
    nonexist.use(5)

def use_nonexist2():
    nonexist().use(5)

def dual():
    gf(5)

def dual():
    # this should override the previous one
    pass

@overload
def dual():
    # this one should be ignored as well
    gf_with_inner(5)
"""
    code3 = """
# root.file3
from .file1 import *

def usage1(x):
    gf(5)
    C(5)
    
"""

    code4 = """
# root.file4
from .file3 import *
from root.file2 import *

def usage4():
    dual()  # from file2
    usage1(5)  # from file2, which shadows file3
    C(5)  # from file1

"""

    project = PythonProject.from_modules(
        [
            PythonModule.from_cst(cst.parse_module(code1), "root.file1"),
            PythonModule.from_cst(cst.parse_module(code2), "root.file2"),
            PythonModule.from_cst(cst.parse_module(code3), "root.file3"),
            PythonModule.from_cst(cst.parse_module(code4), "root.file4"),
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
            (u.used, u.is_certain) for u in analysis.user2used.get(caller_p, list())
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

    assert_usages(
        "root.file2/SubClass.use",
        ("root.file2/UsageClass.foo", False),
        ("root.file1/C.foo", False),
        ("root.file1/C.s_method", True),
    )

    # We should not cound decorator as a usage to avoid blowing up
    assert_usages(
        "root.file2/usage_dec",
    )

    assert_usages(
        "root.file2/use_nonexist",
    )

    assert_usages(
        "root.file2/use_nonexist2",
        ("root.file2/SubClass.use", False),
    )

    assert_usages(
        "root.file2/dual",
    )

    # test star imports
    assert_usages(
        "root.file3/usage1",
        ("root.file1/gf", True),
        ("root.file1/C.__init__", True),
    )

    assert_usages(
        "root.file4/usage4",
        ("root.file2/dual", True),
        ("root.file2/usage1", True),
        ("root.file1/C.__init__", True),
    )


@pytest.mark.skip(reason="not implemented yet.")
def test_attribute_analysis():
    code1 = """
# root.file1
def bernouli():
    return random.random() > 0.5

class A:
    x: int
    y: str = "y_init"
    z: bool = bernouli()

    def __init__(self):
        self.u: bool = bernouli()

    def foo(self):
        return self.x + 1

def bar():
    return A().y
"""

    project = PythonProject.from_modules(
        [PythonModule.from_cst(cst.parse_module(code1), "root.file1")]
    )
    analysis = UsageAnalysis(project)

    def check_attr(attr_path: str, has_initializer: bool, *usages: tuple[str, bool]):
        attr_p = ProjectPath.from_str(attr_path)
        expect = set()
        for caller, certain in usages:
            caller_p = ProjectPath.from_str(caller)
            expect.add((caller_p, certain))

        actual = {
            (u.user, u.is_certain) for u in analysis.used2user.get(attr_p, list())
        }

        assert_eq(actual, expect)

    check_attr(
        "root.file1/A.x",
        False,
        ("root.file1/A.foo", True),
    )

    check_attr(
        "root.file1/A.y",
        True,
        ("root.file1/bar", False),
    )

    check_attr(
        "root.file1/A.z",
        True,
    )

    check_attr(
        "root.file1/A.u",
        False,
        ("root.file1/A.__init__", True),
    )
