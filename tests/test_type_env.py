import os
import shutil
from pathlib import Path

import pytest
from spot.type_check import MypyResult

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
    old_map = {a.path: a.annot for a in old_annots if a.annot is not None}
    new_parsed = apply_annotations(parsed, code_1_patch)
    new_annots = collect_annots_info(new_parsed)
    new_map = {a.path: a.annot for a in new_annots if a.annot is not None}

    for k, v in code_1_patch.items():
        assert old_map[k].annotation.value != new_map[k].annotation.value  # type: ignore
        assert new_map[k].annotation.value == v.annotation.value  # type: ignore


def test_mypy_checker_1():
    with mypy_checker(Path("data/code"), wait_before_check=0.0) as checker:
        check_r = checker.recheck_project()
        assert isinstance(check_r, MypyResult)
        assert Path("data/code/bad_code_1.py").resolve() in check_r.error_dict
        assert Path("data/code/bad_code_2.py").resolve() in check_r.error_dict


def test_mypy_checker_2():
    with mypy_checker(Path("data/code_output"), wait_before_check=0.0) as checker:
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
        ("List[Any]", "List"),
        ("Dict[Any, Any]", "Dict"),
    ]

    for a, b in equiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) == normalize_type(tb)

    nonequiv_pairs: list[tuple[str, str]] = [
        ("Union[str, int]", "Union[str, list]"),
        ("typing.List[str]", "t.List[str]"),
        ("tuple[str, int]", "tuple[int, str]"),
        ("Dict[str, Any]", "Dict"),
    ]

    for a, b in nonequiv_pairs:
        ta = parse_type_str(a)
        tb = parse_type_str(b)
        assert normalize_type(ta) != normalize_type(tb)


import shutil

from spot.data import SrcDataset, type_check_src, type_check_src_in_project
from spot.utils import load_tokenizer_spot, proj_root


def test_mypy_checking():
    simple_dataset = SrcDataset.from_repos(
        proj_root() / "data",
        [proj_root() / "data/code"],
        load_tokenizer_spot(),
        drop_comments=True,
        max_workers=10,
        label_ratio=1.0,
    )

    src_to_check = simple_dataset.get_src_by_file(Path("bad_code_2.py"))
    result_1 = type_check_src(src_to_check, {0: "int"})
    assert len(result_1.feedbacks) == 0

    src_to_check = simple_dataset.get_src_by_file(Path("bad_code_2.py"))
    temp_dir = proj_root() / "mypy_temp/test_dir"
    shutil.rmtree(temp_dir, ignore_errors=True)

    result_2 = type_check_src_in_project(
        src_to_check,
        {0: "int"},
        project_root=(proj_root() / "data/code"),
    )
    assert isinstance(result_2.feedbacks, list) and len(result_2.feedbacks) == 1
    assert (
        'Argument 1 to "fib" has incompatible type "int"; expected "str"'
        in result_2.feedbacks[0].message
    )
