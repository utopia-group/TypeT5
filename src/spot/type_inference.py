import time
from typing import Callable
from .type_checking import *
import tqdm

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


def _test_inference_performance(src_root, src_files):
    dmypy_path = proj_root() / ".venv/bin/dmypy"
    with mypy_checker(dmypy_path, src_root) as checker:
        n_checks = 0
        t_s = time.time()
        for f in tqdm.tqdm(src_files):
            with type_inf_env(checker, f) as env:
                if len(env.to_annot()) == 0:
                    continue  # skip files with no annotations
                n_checks += 1
                while len(env.state.to_annot) > 0:
                    n_checks += 1
                    env.step(TypeInfAction(env.state.to_annot[0], cst.Name("int")))

        t_e = time.time()
        print(f"{n_checks} checks in {t_e - t_s} seconds.")
        print(f"{n_checks / (t_e - t_s)} checks/second.")
