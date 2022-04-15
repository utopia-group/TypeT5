from pathlib import Path
from spot.utils import cst, read_file, write_file, SpecialNames
import os
import shutil
from spot.type_checking import (
    AnnotPath,
    annot_path,
    collect_annotations,
    apply_annotations,
    mypy_checker,
)
from spot.type_inference import *


os.chdir(Path(__file__).parent.parent)

parsed = cst.parse_module(read_file("data/code/bad_code_1.py"))


def test_annotation_collection():
    annot_paths, _ = collect_annotations(parsed)
    correct_annot_paths: list[AnnotPath] = [
        annot_path("fib", "n"),
        annot_path("fib", SpecialNames.Return),
        annot_path("t_add", "x"),
        annot_path("t_add", "y"),
        annot_path("t_add", SpecialNames.Return),
        annot_path("x"),
    ]
    for p in correct_annot_paths:
        assert p in annot_paths
    assert annot_paths == correct_annot_paths


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


def test_type_env():
    # remove `data/temp` if it exists
    inference_dir = "data/code_output/inference"
    if os.path.exists(inference_dir):
        shutil.rmtree(inference_dir)
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)
    write_file(f"{inference_dir}/env_code_1.py", read_file("data/code/env_code_1.py"))

    with mypy_checker(".venv/bin/dmypy", inference_dir) as checker:
        env = TypeInfEnv(checker, f"{inference_dir}/env_code_1.py")
        env.reset()
        while len(env.state.to_annot) > 0:
            env.step(TypeInfAction(env.state.to_annot[0], cst.Name("int")))

        print(env.state)
        assert env.state.num_errors == 0
        assert len(env.state.annotated) == 11
        for k, v in env.state.annotated.items():
            if k == annot_path("int_add", "b"):
                assert v.deep_equals(cst.Name("Any")), f"{k}:{v}"
            else:
                assert v.deep_equals(cst.Name("int")), f"{k}:{v}"