import subprocess

import requests

from typet5.experiments.type4py import Type4PyEvalResult, Type4PySupportedSyntax
from typet5.experiments.utils import SupportedSyntax, remove_newer_syntax
from typet5.function_dataset import SignatureMap
from typet5.static_analysis import (
    ElemSignature,
    FunctionSignature,
    ModuleName,
    ProjectPath,
    PythonProject,
    VariableSignature,
)
from typet5.type_check import normalize_type, parse_type_expr, parse_type_str
from typet5.type_env import AccuracyMetric
from typet5.utils import *

PredList = list[list]  # of the form [[type1, score1], [type2, score2], ...]


def eval_hityper_on_repos(
    repo_roots: list[Path],
    hityper_python: Path,
    work_dir: Path,
    max_workers: int | None = None,
):
    out_dirs = [work_dir / r.name for r in repo_roots]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    model_outputs = pmap(
        run_hityper,
        repo_roots,
        out_dirs,
        [hityper_python] * len(repo_roots),
        desc="Running HiTyper",
        max_workers=max_workers,
    )

    projects = [
        PythonProject.parse_from_root(r, discard_bad_files=True) for r in repo_roots
    ]

    label_signatures: dict[str, SignatureMap] = {
        project.name: {e.path: e.get_signature() for e in project.all_elems()}
        for project in projects
    }

    pred_signatures: dict[str, SignatureMap] = {n: dict() for n in label_signatures}
    for proj, sigmap in zip(projects, model_outputs):
        pred_signatures[proj.name] = sigmap

    return Type4PyEvalResult(
        pred_maps=pred_signatures,
        label_maps=label_signatures,
    )


class HiTyperResponseParser:
    def __init__(self, module: ModuleName):
        self.module = module
        self.assignment: SignatureMap = dict()

    def parse(self, res_json: dict[str, list]) -> SignatureMap:
        self.assignment = dict()

        def parse_var(x: dict) -> tuple[str, cst.Annotation | None]:
            return x["name"], _parse_annot(x["type"])

        for e_name, e_list in res_json.items():
            if e_name == "global@global":
                vars = [v := parse_var(x) for x in e_list if x["category"] == "local"]
                for name, annot in vars:
                    path = ProjectPath(self.module, name)
                    self.assignment[path] = VariableSignature(annot, in_class=False)
            else:
                name, parent = e_name.split("@")
                parent = "" if parent == "global" else parent
                path = ProjectPath(self.module, parent).append(name)
                params = [parse_var(x) for x in e_list if x["category"] == "arg"]
                returns = [parse_var(x) for x in e_list if x["category"] == "return"]
                rt = returns[0][1] if returns else None
                self.assignment[path] = FunctionSignature(
                    {v[0]: v[1] for v in params},
                    rt,
                    in_class=False,
                )

        return self.assignment


def run_hityper(repo: Path, out_dir: Path, python_path: Path) -> SignatureMap:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        [
            python_path,
            "-m",
            "hityper",
            "infer",
            "--type4py",
            "-p",
            repo.resolve(),
            "-d",
            out_dir.resolve(),
        ],
        cwd=out_dir,
        # env={"PYTHONPATH": "src"},
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"HiTyper failed on {repo} with error: {out.stderr.decode()}"
        )
    results = json.loads(read_file(out_dir / "inferred_types.json"))
    sigmap = SignatureMap()
    for fname, mres in results.items():
        mname = PythonProject.rel_path_to_module_name(
            Path(fname).relative_to(repo.resolve())
        )
        parser = HiTyperResponseParser(mname)
        sigmap.update(parser.parse(mres))
    return sigmap


def _parse_annot(ts: list[str]) -> cst.Annotation | None:
    if not ts:
        return None
    try:
        if len(ts) == 1:
            return cst.Annotation(cst.parse_expression(ts[0]))
        else:
            return cst.Annotation(cst.parse_expression(" | ".join(ts)))
    except cst.ParserSyntaxError:
        return None
