import os
import shutil
from pathlib import Path

import pytest

from spot.type_env import (
    AnnotCat,
    AnnotPath,
    AnyAnnot,
    SelectAnnotations,
    TypeInfAction,
    annot_path,
    apply_annotations,
    collect_annots_info,
    mypy_checker,
    normalize_type,
    parse_type_str,
    type_inf_env,
)
from spot.utils import SpecialNames, cst, read_file, write_file

os.chdir(Path(__file__).parent.parent)


def test_annotation_collection():
    parsed = cst.parse_module(read_file("data/code/env_code_2.py"))
    annots = collect_annots_info(parsed)
    annot_paths = [(a.path, a.cat) for a in annots]
    correct_annot_paths: list[tuple[AnnotPath, AnnotCat]] = [
        (annot_path("fib", "n"), AnnotCat.FuncArg),
        (annot_path("fib", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("foo", "bar"), AnnotCat.FuncArg),
        (annot_path("foo", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "z"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "w"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "__init__", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "__init__", "x"), AnnotCat.FuncArg),
        (annot_path("Bar", "__init__", "self.x"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "__init__", "self.y"), AnnotCat.ClassAtribute),
        (annot_path("Bar", "reset", "w0"), AnnotCat.FuncArg),
        (annot_path("Bar", "reset", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("Bar", "foo", "z"), AnnotCat.FuncArg),
        (annot_path("Bar", "foo", SpecialNames.Return), AnnotCat.FuncReturn),
        (annot_path("bar"), AnnotCat.GlobalVar),
    ]
    for pair in correct_annot_paths:
        assert pair in annot_paths
    for pair in annot_paths:
        assert pair in correct_annot_paths


parsed = cst.parse_module(read_file("data/code/bad_code_1.py"))


code_1_patch = {
    annot_path("fib", "n"): cst.Annotation(cst.Name("int")),
    annot_path("fib", SpecialNames.Return): cst.Annotation(cst.Name("int")),
    annot_path("t_add", SpecialNames.Return): cst.Annotation(cst.Name("str")),
    annot_path("bad_y"): AnyAnnot,
}


def test_annotation_applying():
    old_annots = collect_annots_info(parsed)
    old_map = {a.path: a.annot for a in old_annots}
    new_parsed = apply_annotations(parsed, code_1_patch)
    new_annots = collect_annots_info(new_parsed)
    new_map = {a.path: a.annot for a in new_annots}

    for k, v in code_1_patch.items():
        assert old_map[k].annotation.value != new_map[k].annotation.value
        assert new_map[k].annotation.value == v.annotation.value


def test_mypy_checker_1():
    with mypy_checker("data/code", wait_before_check=0.0) as checker:
        check_r = checker.recheck_project()
        assert Path("data/code/bad_code_1.py").resolve() in check_r.error_dict
        assert Path("data/code/bad_code_2.py").resolve() in check_r.error_dict


def test_mypy_checker_2():
    with mypy_checker("data/code_output", wait_before_check=0.0) as checker:
        if Path("data/code_output/bad_code_1.py").exists():
            os.remove("data/code_output/bad_code_1.py")
        oe = checker.recheck_project().num_errors
        write_file("data/code_output/bad_code_1.py", parsed.code)
        assert checker.recheck_project().num_errors > oe
        new_code = apply_annotations(parsed, code_1_patch).code
        write_file(
            "data/code_output/bad_code_1.py",
            new_code,
        )
        c_r = checker.recheck_project()
        assert c_r.num_errors == oe, f"mypy_output: {c_r.output_str}\ncode: {new_code}"


@pytest.mark.skip(reason="Is considering to deprecate incremental type checking.")
def test_type_env():
    # remove `data/temp` if it exists
    inference_dir = "data/code_output/inference"
    if os.path.exists(inference_dir):
        shutil.rmtree(inference_dir)
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)
    write_file(f"{inference_dir}/env_code_1.py", read_file("data/code/env_code_1.py"))
    write_file(f"{inference_dir}/env_code_2.py", read_file("data/code/env_code_2.py"))

    with mypy_checker(inference_dir, wait_before_check=0.0) as checker:
        with type_inf_env(
            checker,
            f"{inference_dir}/env_code_1.py",
            SelectAnnotations.select_all_paths,
        ) as env:
            while len(env.state.to_annot) > 0:
                p = next(iter(env.state.to_annot))
                env.step(TypeInfAction(p, cst.Name("int")))

            assert env.state.num_errors == 0
            assert len(env.state.annotated) == 11
            for k, v in env.state.annotated.items():
                if k == annot_path("int_add", "b"):
                    assert not v.deep_equals(cst.Name("int")), f"{k}:{v}"
                else:
                    assert v.deep_equals(cst.Name("int")), f"{k}:{v}"

        _, annots = collect_annots_info(
            cst.parse_module(read_file(f"{inference_dir}/env_code_2.py"))
        )
        with type_inf_env(
            checker,
            f"{inference_dir}/env_code_2.py",
            SelectAnnotations.select_annotated,
        ) as env:
            assert len(env.state.annotated) == 0
            assert (
                len(env.state.to_annot) == len(annots) == 10
            )  # this should equal to the number of manual annotations
            while len(env.state.to_annot) > 0:
                path = next(iter(env.state.to_annot))
                env.step(TypeInfAction(path, annots[path].annotation))

            assert env.state.num_errors == 0
            assert len(env.state.annotated) == 10


def test_type_normalization():
    equiv_pairs: list[tuple[str, str]] = [
        ("list[int]", "List[int]"),
        ("dict[str, list]", "Dict[str, List]"),
        ("'Foo[int]'", "Foo[int]"),
        ("typing.Union[str, List]", "typing.Union[list, str]"),
        ("typing.Union[str, typing.Union[str, int]]", "str | int"),
        ("typing.Union[str, float, typing.Union[str, int]]", "str | int | float"),
        ("Union[str, float, None]", "Optional[Union[str, float]]"),
        ("str | None", "Optional[str]"),
        ("Any | None", "Optional"),
    ]

    for a, b in equiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) == normalize_type(tb)

    nonequiv_pairs: list[tuple[str, str]] = [
        ("Union[str, int]", "Union[str, list]"),
        ("typing.List[str]", "t.List[str]"),
        ("tuple[str, int]", "tuple[int, str]"),
    ]

    for a, b in nonequiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) != normalize_type(tb)
