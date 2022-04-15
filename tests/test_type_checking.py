from pathlib import Path
from spot.utils import cst, read_file, write_file, SpecialNames
import os
from spot.type_checking import (
    AnnotPath,
    annot_path,
    collect_annotations,
    apply_annotations,
    mypy_checker,
)

os.chdir(Path(__file__).parent.parent)

parsed = cst.parse_module(read_file("data/code/bad_code_1.py"))


def test_annotation_collection():
    annot_paths, _ = collect_annotations(parsed)
    annot_places: list[AnnotPath] = [
        annot_path("fib", "n"),
        annot_path("fib", SpecialNames.Return),
        annot_path("t_add", "x"),
        annot_path("t_add", "y"),
        annot_path("t_add", SpecialNames.Return),
        annot_path("x"),
    ]
    for p in annot_places:
        assert p in annot_paths


code_1_patch = {
    annot_path("fib", "n"): cst.Annotation(cst.Name("int")),
    annot_path("fib", SpecialNames.Return): cst.Annotation(cst.Name("int")),
    annot_path("t_add", SpecialNames.Return): cst.Annotation(cst.Name("str")),
}


def test_annotation_applying():
    _, old_annots = collect_annotations(parsed)
    new_parsed = apply_annotations(parsed, code_1_patch)
    _, new_annots = collect_annotations(new_parsed)

    for k, v in code_1_patch.items():
        assert old_annots[k].annotation != new_annots[k].annotation
        assert new_annots[k].annotation == v.annotation


def test_mypy_checker_1():
    with mypy_checker(".venv/bin/dmypy", "data/code") as checker:
        check_r = checker.check_code_dir()
        assert "bad_code_1.py" in check_r.num_error_dict
        assert "bad_code_2.py" in check_r.num_error_dict


def test_mypy_checker_2():
    with mypy_checker(".venv/bin/dmypy", "data/code_output") as checker:
        write_file("data/code_output/bad_code_1.py", parsed.code)
        assert checker.recheck_files("bad_code_1.py").num_errors > 0
        write_file(
            "data/code_output/bad_code_1.py",
            apply_annotations(parsed, code_1_patch).code,
        )
        assert checker.recheck_files("bad_code_1.py").num_errors == 0
