from pathlib import Path

from numpy import False_
from spot.static_analysis import (
    ModuleHierarchy,
    ModuleName,
    ProjectPath,
    PythonModule,
    PythonProject,
    UsageAnalysis,
    build_project_namespaces,
    cst,
    stub_from_module,
    to_abs_import_path,
)
import pytest

from spot.utils import assert_eq, groupby, not_none, show_string_diff


def project_from_code(name2code: dict[ModuleName, str]):
    modules = [
        PythonModule.from_cst(cst.parse_module(code), name)
        for name, code in name2code.items()
    ]
    return PythonProject.from_modules(Path("[test project]"), modules)


def full_analysis(project: PythonProject) -> UsageAnalysis:
    return UsageAnalysis(project, add_override_usages=False)


def test_path_to_module():
    to_module = PythonProject.rel_path_to_module_name
    assert to_module(Path("a/b.py")) == "a.b"
    assert to_module(Path("a/b/c.py")) == "a.b.c"
    assert to_module(Path("a/__init__.py")) == "a"
    assert to_module(Path("a/b/__init__.py")) == "a.b"
    assert to_module(Path("src/a.py")) == "a"
    assert to_module(Path("src/a/__init__.py")) == "a"


def to_abs(module_name: ModuleName, import_name: str):
    return list(to_abs_import_path(module_name, import_name))


def test_import_normalization():
    assert to_abs("spot.static_analysis", ".") == ["spot"]
    assert to_abs("spot.static_analysis", "..") == [""]
    assert to_abs("spot.static_analysis", ".utils") == ["spot.utils"]
    assert to_abs("spot.static_analysis", ".utils.a") == ["spot.utils.a"]
    assert to_abs("spot.static_analysis", "foo.bar") == ["foo.bar", "spot.foo.bar"]

    with pytest.raises(AssertionError):
        to_abs("spot.static_analysis", "...")


def test_namespace_resolution():
    modules = [
        "foo",
        "foo.bar",
        "z",
    ]
    namespace = ModuleHierarchy.from_modules(modules)

    assert namespace.resolve_path(["foo"]) == None
    assert namespace.resolve_path(["z"]) == None

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
from . import E
"""
    mod = PythonModule.from_cst(cst.parse_module(import_code), "root.file1")
    assert mod.imported_modules == {
        "A",
        "B",
        "B.C",
        "D",
        "D.a",
        "D.b",
        "root",
        "root.utils",
        "root.utils.x",
        "top",
        "infer",
        "infer.type",
        "root.E",
    }


def test_implicit_relative_imports():
    # this is a super weird import style, but somehow it was used in
    # some projects and was supported by the IDE...

    code_B = """
# file: root.B
def f():
    pass
"""

    code_A = """
# file: root.A
from B import f  # equivalent to from .B import f

def g():
    f()

"""
    project = project_from_code({"root.B": code_B, "root.A": code_A})
    analysis = UsageAnalysis(project, add_implicit_rel_imports=True)

    analysis.assert_usages(
        "root.A/g",
        ("root.B/f", True),
    )


def test_inner_classes():
    code1 = """
# file1
x = 1

class A:
    x: list
    class B1:
        x: float

        def __init__(self):
            self.x = 1
        

    class B2:
        x: str

        def __init__(self):
            self.x = 2

        class A:
            x: int

            def f(self):
                return A.B1()

"""
    project = project_from_code({"file1": code1})
    m = project.modules["file1"]

    assert set(c.path.path for c in m.all_classes()) == {"A", "A.B1", "A.B2", "A.B2.A"}

    assert sum(1 for _ in m.all_vars()) == 5
    assert sum(1 for _ in m.all_funcs()) == 3

    analysis = full_analysis(project)

    analysis.assert_usages("file1/A.B1.__init__", ("file1/A.B1.x", True))
    analysis.assert_usages("file1/A.B2.__init__", ("file1/A.B2.x", True))
    analysis.assert_usages("file1/A.B2.A.f", ("file1/A.B1.__init__", True))


def test_light_stub_gen():
    code = """
import typing

T1 = typing.TypeVar("T1") # keep
T2 = list[T1] # keep
number = int # keep
Count = 0 # drop

class A(typing.Generic[T1]): # keep
    # drop body
    x = 0
    def __init__(self, x: T1):
        self.x = x
    
def some_f() -> number: # drop
    return 1
"""

    expected = """
import typing
T1 = typing.TypeVar("T1")
T2 = list[T1]
number = int
class A(typing.Generic[T1]):
    ...
"""

    stub = stub_from_module(
        cst.parse_module(code), lightweight=True, rm_comments=True, rm_imports=False
    )
    if stub.code.strip() != expected.strip():
        print(show_string_diff(stub.code.strip(), expected.strip()))
        assert False, "stub code does not match expected."


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

    project = project_from_code(
        {
            "root.file1": code1,
            "root.file2": code2,
            "root.file3": code3,
            "root.file4": code4,
        }
    )
    analysis = full_analysis(project)

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

    code5 = """
# root.file5
from . import file1

def usage5():
    file1.gf(5)
    
"""

    project = project_from_code(
        {
            "root.file1": code1,
            "root.file5": code5,
        }
    )
    analysis = full_analysis(project)
    analysis.assert_usages("root.file5/usage5", ("root.file1/gf", True))

    # test default argument usage
    code6 = """
# root.file6

Count = 1

def inc(x=Count):
    return x + 1
"""

    project = project_from_code(
        {
            "root.file6": code6,
        }
    )
    analysis = full_analysis(project)
    analysis.assert_usages("root.file6/inc", ("root.file6/Count", True))


def test_inheritance_usages():
    methods_usage = """
# root.file1

class A(ABC):
    @abstractmethod
    def foo(self, x: int | float):
        pass

    def bar():
        pass

class B(A):
    def foo(self, x: int):
        # should use A.foo
        return 1

class C(B):
    def foo(self, x: float):
        # should use B.foo
        return 2.0

    def bar(self):
        return 1

"""

    project = project_from_code({"root.file1": methods_usage})
    analysis = UsageAnalysis(project, add_override_usages=True)

    analysis.assert_usages(
        "root.file1/B.foo",
        ("root.file1/A.foo", False),
    )

    analysis.assert_usages(
        "root.file1/C.foo",
        ("root.file1/B.foo", False),
    )

    analysis.assert_usages(
        "root.file1/C.bar",
        ("root.file1/A.bar", False),
    )

    attr_usages = """
# root.file2

class A:
    x = 1
    def __init__(self):
        self.y = 2

class B(A):
    x = 0.5
    def __init__(self):
        pass

class C(B):
    x = 0.25
    y = 0.12
"""

    project = project_from_code({"root.file2": attr_usages})
    analysis = UsageAnalysis(project, add_override_usages=True)

    analysis.assert_usages(
        "root.file2/B.__init__",
        ("root.file2/A.__init__", False),
    )

    analysis.assert_usages(
        "root.file2/B.x",
        ("root.file2/A.x", False),
    )

    analysis.assert_usages(
        "root.file2/C.x",
        ("root.file2/B.x", False),
    )

    analysis.assert_usages(
        "root.file2/C.y",
        ("root.file2/A.y", False),
    )


def test_annotation_suages():
    code1 = """
# root.file1

Count = 0

def annot1(f, x: int):
    return f(x)

def annot2(f):
    return f

@annot2
@annot1(Count)
def usage_f(x):
    return x

"""
    project = project_from_code({"root.file1": code1})
    analysis = full_analysis(project)

    analysis.assert_usages(
        "root.file1/usage_f",
        ("root.file1/Count", True),
        ("root.file1/annot1", True),
        ("root.file1/annot2", True),
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

    project = project_from_code(
        {
            "root.file1": code1,
        }
    )
    analysis = full_analysis(project)

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

    project = project_from_code(
        {
            "root.file1": code1,
            "root.file2": code2,
        }
    )
    analysis = full_analysis(project)

    B_cls = project.modules["root.file2"].classes[0]
    assert {n.name for n in not_none(B_cls.superclasses)} == {"root.file1.A"}

    # test star import of classes
    ns_hier = ModuleHierarchy.from_modules(project.modules.keys())
    assert build_project_namespaces(project, ns_hier)["root.file2"]["A"] == A_cls.path

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

    # now test a chain of accesses
    code3 = """
# root.file3
from lib import Bar
import lib

class Foo:
    x: Bar

    def use1(self):
        self.x.fly()

    def use2(self):
        Bar().x.fly()

    def use3(self):
        lib.x.fly()

"""

    project = project_from_code(
        {
            "root.file3": code3,
        }
    )
    analysis = full_analysis(project)

    # only the certain usages should be tracked
    # other usages are attributed to the fly() method.
    analysis.assert_usages(
        "root.file3/Foo.use1",
        ("root.file3/Foo.x", True),
    )

    analysis.assert_usages(
        "root.file3/Foo.use2",
        ("root.file3/Foo.x", False),
    )

    analysis.assert_usages(
        "root.file3/Foo.use3",
    )


def test_constructors():
    code1 = """
# root.file1
class A:
    x: int

    def __init__(self, x):
        self.x = x

class B(MaybeNamedTuple):
    x: int
    y: int = field(init=False)


@maybe_dataclass
class C:
    u: int
    v: int

# not a dataclass since A has __init__
class D(A):
    y: int

class E(B):
    z: int

def use():
    A(1)
    B(1, 2)
    C(1, 2)

def use2():    
    D(1)
    E(1,2,3)
"""

    project = project_from_code(
        {
            "root.file1": code1,
        }
    )
    analysis = full_analysis(project)

    analysis.assert_usages(
        "root.file1/use",
        ("root.file1/A.__init__", True),
        ("root.file1/B.x", True),
        ("root.file1/B.y", True),
        ("root.file1/C.u", True),
        ("root.file1/C.v", True),
    )

    analysis.assert_usages(
        "root.file1/use2",
        ("root.file1/A.__init__", True),
        ("root.file1/B.x", False),
        ("root.file1/B.y", False),
        ("root.file1/E.z", True),
    )

    code2 = """
# root.file2

@dataclass
class B:
    x: int
    y: int = field(init=False)

def use():
    # these should not trigger constructor usage.
    B
    isinstance(x, B)
    list[B]()

    """

    project = project_from_code(
        {
            "root.file2": code2,
        }
    )
    analysis = full_analysis(project)

    analysis.assert_usages(
        "root.file2/use",
    )


def test_module_symbols():
    code1 = """
# root.file1

A = 1

def A():
    pass

# should shadow the global A
class A:
    pass

"""

    m1 = PythonModule.from_cst(cst.parse_module(code1), "root.file1")

    assert len(m1.global_vars) == 0
    assert len(m1.functions) == 0
    assert len(m1.classes) == 1


def test_fixture_usages():
    code1 = """
# root.test_1
from pytest import fixture
import pytest as pt

@fixture
def fix1():
    return 1

@pt.fixture
def fix2(fix1):
    return fix1 + 1

def test1(fix1, fix2):
    assert fix1 + 1 == fix2
"""

    project = project_from_code({"root.test_1": code1})
    analysis = full_analysis(project)

    analysis.assert_usages(
        "root.test_1/fix2",
        ("root.test_1/fix1", True),
    )

    analysis.assert_usages(
        "root.test_1/test1",
        ("root.test_1/fix1", True),
        ("root.test_1/fix2", True),
    )

    code_conf = """
# root.conftest
import pytest  

@pytest.fixture(scope="session")
def use_gfix():
    return 1
    
"""

    code2 = """
# root.nest.test_2
def test_global_fix(fix1, use_gfix):
    # fix1 should not be visible
    return fix1 + use_gfix
"""

    code_out = """
# test_out    
def test_global_fix(fix1, use_gfix):
    # both fix1 and use_gfix should not be visible
    return fix1 + use_gfix 
"""

    project = project_from_code(
        {
            "root.test_1": code1,
            "root.conftest": code_conf,
            "root.nest.test_2": code2,
            "test_out": code_out,
        }
    )

    analysis = full_analysis(project)

    analysis.assert_usages(
        "root.nest.test_2/test_global_fix",
        ("root.conftest/use_gfix", True),
    )

    analysis.assert_usages(
        "test_out/test_global_fix",
    )

    code_global = """
# conftest
import pytest

@pytest.fixture
def fix1():
    # should be visible everywhere
    return 1
"""

    project = project_from_code(
        {"root.test_1": code1, "conftest": code_global, "test_out": code_out}
    )

    analysis = full_analysis(project)

    analysis.assert_usages(
        "test_out/test_global_fix",
        ("conftest/fix1", True),
    )
