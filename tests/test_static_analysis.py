from pathlib import Path
from spot.static_analysis import (
    ModuleHierarchy,
    ProjectPath,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    build_project_namespaces,
    cst,
    compute_module_usages,
    to_abs_import_path as to_abs,
)
import pytest

from spot.utils import assert_eq, groupby, not_none


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
        return not y
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
    return  {x: x.foo(3)}

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
        gf_with_inner(x)
        self.foo(5)

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
    weird(1).use(5)

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
    nonexist()

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

    analysis.assert_usages(
        "root.file2/usage1",
        ("root.file1/gf", True),
        ("root.file1/C.__init__", True),
    )

    analysis.assert_usages(
        "root.file2/usage2",
        ("root.file1/gf_with_inner", True),
    )

    analysis.assert_usages(
        "root.file2/usage_method1",
        ("root.file1/C.__init__", True),
        ("root.file1/C.foo", False),
        ("root.file2/UsageClass.foo", False),
    )

    analysis.assert_usages(
        "root.file2/usage_method2",
        ("root.file1/C.__init__", True),
        ("root.file1/C.foo", False),
        ("root.file2/UsageClass.foo", False),
    )

    analysis.assert_usages(
        "root.file2/usage_local",
        ("root.file2/usage1", True),
        ("root.file2/UsageClass.__init__", True),
    )

    analysis.assert_usages(
        "root.file2/UsageClass.__init__",
        ("root.file1/gf_with_inner", True),
        ("root.file2/UsageClass.foo", True),
    )

    analysis.assert_usages(
        "root.file2/UsageClass.foo",
        ("root.file2/usage_local", True),
        ("root.file1/gf", True),
    )

    analysis.assert_usages(
        "root.file2/SubClass.use",
        ("root.file2/UsageClass.foo", True),
        ("root.file1/C.s_method", True),
    )

    # We should not cound decorator as a usage to avoid blowing up
    analysis.assert_usages(
        "root.file2/usage_dec",
    )

    analysis.assert_usages(
        "root.file2/use_nonexist",
    )

    analysis.assert_usages(
        "root.file2/use_nonexist2",
        ("root.file2/SubClass.use", False),
    )

    analysis.assert_usages(
        "root.file2/dual",
    )

    # test star imports
    analysis.assert_usages(
        "root.file3/usage1",
        ("root.file1/gf", True),
        ("root.file1/C.__init__", True),
    )

    analysis.assert_usages(
        "root.file4/usage4",
        ("root.file2/dual", True),
        ("root.file2/usage1", True),
        ("root.file1/C.__init__", True),
    )


def test_attribute_analysis():
    code1 = """
# root.file1
def bernouli():
    return random.random() > 0.5

Count = 1
Count = bernouli()

class A:
    x: int
    y: str = "y_init"
    z: bool = bernouli()
    s = 1

    def __init__(self):
        self.u: bool = bernouli()
        self.v = self.foo()

    def foo(self):
        return {self.x: self.undefined}

def inc():
    global Count
    Count += 1

def list():
    return [Count]

def loop():
    for x in Count:
        print(x)

def bar():
    return A().y.x
"""

    project = PythonProject.from_modules(
        [PythonModule.from_cst(cst.parse_module(code1), "root.file1")],
    )
    analysis = UsageAnalysis(project)

    A_cls = project.modules["root.file1"].classes[0]
    A_attrs = set(A_cls.attributes.keys())
    assert_eq(A_attrs, {"x", "y", "z", "s", "u", "v"})

    def check_var(attr_path: str, n_initializers: int):
        attr_p = ProjectPath.from_str(attr_path)
        assert_eq(len(analysis.get_var(attr_p).assignments), n_initializers)

    check_var("root.file1/Count", 2)
    check_var("root.file1/A.x", 1)
    check_var("root.file1/A.y", 1)
    check_var("root.file1/A.z", 1)
    check_var("root.file1/A.s", 1)
    check_var("root.file1/A.u", 0)
    check_var("root.file1/A.v", 0)

    analysis.assert_usages("root.file1/inc", ("root.file1/Count", True))
    analysis.assert_usages("root.file1/list", ("root.file1/Count", True))
    analysis.assert_usages("root.file1/loop", ("root.file1/Count", True))
    analysis.assert_usages(
        "root.file1/bar",
        ("root.file1/A.__init__", True),
        ("root.file1/A.y", False),
        ("root.file1/A.x", False),
    )

    analysis.assert_usages(
        "root.file1/A.__init__",
        ("root.file1/A.u", True),
        ("root.file1/A.v", True),
        ("root.file1/A.foo", True),
        ("root.file1/bernouli", True),
    )

    analysis.assert_usages(
        "root.file1/A.foo",
        ("root.file1/A.x", True),
    )

    # initializer usages
    analysis.assert_usages("root.file1/Count", ("root.file1/bernouli", True))
    analysis.assert_usages("root.file1/A.x")
    analysis.assert_usages("root.file1/A.y")
    analysis.assert_usages("root.file1/A.z", ("root.file1/bernouli", True))
    analysis.assert_usages("root.file1/A.s")
    analysis.assert_usages("root.file1/A.u")
    analysis.assert_usages("root.file1/A.v")

    # test inheritance
    code2 = """
from root.file1 import A as Parent
from root.file1 import *

class B(Parent):
    new_mem1: bool
    def fly(self):
        self.new_mem2 = self.foo()
        return self.x

class C():
    y = "y_C"


class D(A, C):
    def __init__(self):
        self.y += 1  # should use y from C
        self.z * 2 # should use z from A

def test_annot():
    x: D = undefined

"""

    project = PythonProject.from_modules(
        [
            PythonModule.from_cst(cst.parse_module(code1), "root.file1"),
            PythonModule.from_cst(cst.parse_module(code2), "root.file2"),
        ],
    )
    analysis = UsageAnalysis(project)

    B_cls = project.modules["root.file2"].classes[0]
    assert {n.name for n in not_none(B_cls.superclasses)} == {"root.file1.A"}

    # test star import of classes
    assert build_project_namespaces(project)["root.file2"]["A"] == A_cls.path

    analysis.assert_usages(
        "root.file2/B.fly",
        ("root.file1/A.x", True),
        ("root.file1/A.foo", True),
        ("root.file2/B.new_mem2", True),
    )

    analysis.assert_usages(
        "root.file2/D.__init__",
        ("root.file2/C.y", True),
        ("root.file1/A.z", True),
    )

    analysis.assert_usages(
        "root.file2/test_annot",
    )
