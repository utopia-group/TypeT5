from pathlib import Path
from spot.utils import cst, read_file, write_file, SpecialNames
import os
from spot.type_checking import (
    AnnotPath,
    annot_path,
    collect_annotations,
    apply_annotations,
    MypyChecker,
)

os.chdir(Path(__file__).parent.parent)

parsed = cst.parse_module(read_file("data/code/bad_code_1.py"))


def test_annotation_collection():
    annots = collect_annotations(parsed)
    annot_places: list[AnnotPath] = [
        annot_path("fib", "n"),
        annot_path("fib", SpecialNames.Return),
        annot_path("t_add", "x"),
        annot_path("t_add", "y"),
        annot_path("t_add", SpecialNames.Return),
        annot_path("x"),
    ]
    for p in annot_places:
        assert p in annots


code_1_patch = {
    annot_path("fib", "n"): cst.Annotation(cst.Name("int")),
    annot_path("fib", SpecialNames.Return): cst.Annotation(cst.Name("int")),
    annot_path("t_add", SpecialNames.Return): cst.Annotation(cst.Name("str")),
}


def test_annotation_applying():
    old_annots = collect_annotations(parsed)
    new_parsed = apply_annotations(parsed, code_1_patch)
    new_annots = collect_annotations(new_parsed)

    for k, v in code_1_patch.items():
        assert not old_annots[k].annotation.deep_equals(new_annots[k].annotation)
        assert new_annots[k].annotation == (v.annotation)


def test_mypy_checker_1():
    checker = MypyChecker(".venv/bin/dmypy", "data/code")
    check_r = checker.check_file(".")
    assert "bad_code_1.py" in check_r.num_error_dict
    assert "bad_code_2.py" in check_r.num_error_dict


def test_mypy_checker_2():
    out_checker = MypyChecker(".venv/bin/dmypy", "data/code_output")
    write_file("data/code_output/bad_code_1.py", parsed.code)
    assert out_checker.check_file(".").num_errors > 0
    write_file("data/code_output/bad_code_1.py", apply_annotations(parsed, code_1_patch).code)
    assert out_checker.check_file(".").num_errors == 0
