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
        return f"{'.'.join(map(str, self.value.reverse()))}"

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
    The order of the returned annotations is guaranteed to follow the order of the source code."""

    def as_tuple(pos: CodePosition):
        return pos.line, pos.column

    annots = collect_annots_info(code)

    types: list[PythonType] = []
    annots_info: list[AnnotInfo] = []
    labels_pos: list[tuple[int, int]] = []

    for info in annots:
        if info.annot is None:
            continue
        ty = parse_type_expr(info.annot.annotation, silent=True)
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
        self.path: AnnotPath = annot_path()
        self.path_name_counts: dict[AnnotPath, dict[str, int]] = {}

    def get_path(self, name: str) -> AnnotPath:
        counts = self.path_name_counts.get(self.path, {})
        self.path_name_counts[self.path] = counts
        id = counts.get(name, 0)
        counts[name] = id + 1
        return self.path.append(name, id)

    def on_visit(self, node):
        match node:
            case cst.FunctionDef():
                self.path = self.get_path(node.name.value)
            case cst.ClassDef():
                self.path = self.get_path(node.name.value)
            case cst.Lambda():
                self.path = self.get_path(SpecialNames.Lambda)

    def on_leave(self, node):
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.path = self.path.pop()


class AnnotCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        self.pm = CodePathManager()
        # store the type annotations
        self.annots_info: list[AnnotInfo] = []
        self.cat_stack = [AnnotCat.GlobalVar]

    def on_visit(self, node):
        self.pm.on_visit(node)
        match node:
            case cst.FunctionDef() | cst.Lambda():
                self.cat_stack.append(AnnotCat.LocalVar)
            case cst.ClassDef():
                self.cat_stack.append(AnnotCat.ClassAtribute)
        return super().on_visit(node)

    def on_leave(self, node):
        r = super().on_leave(node)
        match node:
            case cst.ClassDef() | cst.FunctionDef() | cst.Lambda():
                self.cat_stack.pop()
        self.pm.on_leave(node)
        return r

    def _record_annot(
        self,
        name: str,
        cat: AnnotCat,
        annot: cst.Annotation | None,
    ) -> None:
        path = self.pm.get_path(name)
        crange = None
        if annot:
            crange = self.get_metadata(PositionProvider, annot)
            crange = CodeRange(fix_code_pos(crange.start), fix_code_pos(crange.end))
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


def fix_code_pos(pos: CodePosition):
    "convert to 1-based indexing for columns"
    return CodePosition(pos.line, pos.column + 1)


class AnnotApplier(cst.CSTTransformer):
    """Apply the annotations to the code. Note that each AnnotPath will be
    applied at most once."""

    def __init__(self, annots: Dict[AnnotPath, cst.Annotation]):
        super().__init__()

        self.annots = annots.copy()
        self.pm = CodePathManager()

        # store the target prefixes
        self.prefixes: Set[AnnotPath] = {annot_path()}
        for path in annots.keys():
            while bool(path.value):
                self.prefixes.add(path)
                path = path.pop()

    def on_visit(self, node):
        self.pm.on_visit(node)
        if self.pm.path not in self.prefixes:
            return False
        return super().on_visit(node)

    def on_leave(self, node, updated):
        r = super().on_leave(node, updated)
        self.pm.on_leave(node)
        return r

    def leave_FunctionDef(
        self, node: cst.FunctionDef, updated: cst.FunctionDef
    ) -> cst.FunctionDef:
        path = self.pm.get_path(SpecialNames.Return)
        if path in self.annots:
            rep = self.annots.pop(path)
            return updated.with_changes(returns=rep)
        else:
            return updated

    def leave_Param(self, node: cst.Param, updated: cst.Param) -> cst.Param:
        path = self.pm.get_path(updated.name.value)
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
        path = self.pm.get_path(key_name)

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


@dataclass
class AccuracyMetric:
    common_type_names: set[str]
    normalize_types: bool = True
    relaxed_equality: bool = True
    filter_none_any: bool = True
    match_base_only: bool = False
    ignore_namespace: bool = True
    ast_depth_limit: int | None = None
    filter_rare: bool = (
        False  # when filter_rare=True and keep_rare=False, only common types are kept
    )
    keep_rare: bool = (
        False  # when filter_rare=True and keep_rare=True, only rare types are kept
    )
    name: str = "acc"

    def process_type(self, t: PythonType) -> PythonType:
        if self.normalize_types:
            t = normalize_type(t)
        if self.relaxed_equality:
            t = remove_top_final(t)
            t = remove_top_optional(t)
        if self.match_base_only:
            t = PythonType(t.head, ())
        if self.ignore_namespace:
            t = remove_type_namespace(t)
        if self.ast_depth_limit is not None:
            t = limit_type_depth(t, self.ast_depth_limit)
        return t

    _NoneOrAny = {"None", "Any"}

    def to_keep_type(self, t: PythonType) -> bool:
        return (not self.filter_none_any or t.head_name() not in self._NoneOrAny) and (
            not self.filter_rare or (self.is_common_type(t) != self.keep_rare)
        )

    def is_common_type(self, t: PythonType) -> bool:
        return t.head_name() in self.common_type_names and all(
            map(self.is_common_type, t.args)
        )

    @staticmethod
    def default_metrics(
        common_type_names: set[str], ast_depth_limit: int | None = None
    ):
        return [
            AccuracyMetric(
                common_type_names,
                relaxed_equality=False,
                filter_none_any=False,
                ignore_namespace=False,
                ast_depth_limit=ast_depth_limit,
                name="full_acc",
            ),
            AccuracyMetric(
                common_type_names,
                relaxed_equality=False,
                filter_none_any=False,
                ignore_namespace=False,
                filter_rare=True,
                ast_depth_limit=ast_depth_limit,
                name="full_acc_common",
            ),
            AccuracyMetric(
                common_type_names,
                relaxed_equality=False,
                filter_none_any=False,
                ignore_namespace=False,
                filter_rare=True,
                keep_rare=True,
                ast_depth_limit=ast_depth_limit,
                name="full_acc_rare",
            ),
            AccuracyMetric(
                common_type_names, ast_depth_limit=ast_depth_limit, name="acc"
            ),
            AccuracyMetric(
                common_type_names,
                ast_depth_limit=ast_depth_limit,
                filter_rare=True,
                name="acc_common",
            ),
            AccuracyMetric(
                common_type_names,
                filter_rare=True,
                keep_rare=True,
                ast_depth_limit=ast_depth_limit,
                name="acc_rare",
            ),
            AccuracyMetric(
                common_type_names,
                match_base_only=True,
                ast_depth_limit=ast_depth_limit,
                name="base_acc",
            ),
            AccuracyMetric(
                common_type_names,
                match_base_only=True,
                filter_rare=True,
                ast_depth_limit=ast_depth_limit,
                name="base_acc_common",
            ),
            AccuracyMetric(
                common_type_names,
                match_base_only=True,
                filter_rare=True,
                keep_rare=True,
                ast_depth_limit=ast_depth_limit,
                name="base_acc_rare",
            ),
        ]


def type_accuracies(
    pred_types: Sequence[PythonType],
    label_types: Sequence[PythonType],
    types_cat: Sequence[AnnotCat],
    metric: AccuracyMetric,
    crash_on_type_mask=True,
    output_incorrect_set: list[int] | None = None,
) -> dict[str, Any]:
    assert_eq(len(pred_types), len(label_types), len(types_cat))

    if crash_on_type_mask:
        if PythonType.from_name(SpecialNames.TypeMask) in label_types:
            raise RuntimeError("TypeMask found in label types.")

    pred_types = list(map(metric.process_type, pred_types))
    label_types = list(map(metric.process_type, label_types))

    if metric.filter_none_any | metric.filter_rare:
        filtered_ids = [i for i, t in enumerate(label_types) if metric.to_keep_type(t)]
        n_filtered = len(label_types) - len(filtered_ids)
        pred_types = [pred_types[i] for i in filtered_ids]
        label_types = [label_types[i] for i in filtered_ids]
    else:
        filtered_ids = range(len(label_types))
        n_filtered = 0

    def ast_size(ty: PythonType) -> int:
        return 1 + sum(ast_size(a) for a in ty.args)

    acc_by_cat = GroupedAccCounter[AnnotCat]()
    # acc_by_pos = GroupedAccCounter[range]()
    # acc_by_common = GroupedAccCounter[bool]()
    acc_by_simple = GroupedAccCounter[bool]()

    for i, p, l, cat in zip(
        range(len(pred_types)),
        pred_types,
        label_types,
        types_cat,
    ):
        is_correct = p == l
        acc_by_cat.count(cat, is_correct, 1)
        acc_by_simple.count(ast_size(p) == 1, is_correct, 1)
        # if metric.common_type_names is not None:
        #     acc_by_common.count(metric.is_common_type(l), is_correct, 1)
        if not is_correct and output_incorrect_set is not None:
            output_incorrect_set.append(filtered_ids[i])

    # acc_by_common = acc_by_common.grouped_accs(key=lambda x: "common" if x else "rare")
    acc_by_simple = acc_by_simple.grouped_accs(
        key=lambda x: "simple" if x else "complex"
    )
    metric_name = metric.name

    extra_stats = (
        dict()
        if metric.match_base_only
        else {
            f"{metric_name}_by_simple": acc_by_simple,
            f"{metric_name}_label_size": safe_div(
                sum(ast_size(l) for l in label_types), len(label_types)
            ),
            f"{metric_name}_pred_size": safe_div(
                sum(ast_size(p) for p in pred_types), len(pred_types)
            ),
        }
    )

    return {
        metric_name: acc_by_cat.overall_acc(),
        # f"{metric_name}_by_common": acc_by_common,
        f"{metric_name}_by_cat": acc_by_cat.grouped_accs(
            key=lambda x: x.name, sort_by=lambda x: x.value
        ),
        **extra_stats,
        # f"{metric_name}_by_pos": acc_by_pos.grouped_accs(sort_by=lambda x: x.start),
        f"{metric_name}_ignored_labels": n_filtered,
    }


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
