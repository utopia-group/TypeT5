import json
import subprocess

from typet5.experiments.utils import SupportedSyntax
from typet5.function_dataset import collect_public_api_labels
from typet5.static_analysis import LabelInfo, ModuleName, PythonProject
from typet5.type_check import PythonType, parse_type_expr, parse_type_str
from typet5.type_env import AccuracyMetric, type_accuracies
from typet5.utils import *

JSON = dict


def eval_typilus_on_repos(
    repo_roots: list[Path],
    metrics: list[AccuracyMetric],
    typilus_path: Path,
    work_dir: Path,
    max_workers: int | None = None,
):
    out_dirs = [work_dir / r.name for r in repo_roots]
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    pmap(
        run_typilus,
        repo_roots,
        out_dirs,
        [typilus_path] * len(repo_roots),
        desc="Running Typilus",
        max_workers=max_workers,
    )

    typilus_outputs = []
    for repo in repo_roots:
        with open(work_dir / repo.name / "predictions.json") as f:
            typilus_outputs.append(json.load(f))
    return analyze_typilus_predictions(
        typilus_outputs,
        repo_roots,
        metrics,
    )


def run_typilus(repo: Path, out_dir: Path, typilus_path: Path) -> None:
    typilus_python = "/home/jiayi/anaconda3/envs/typilus-torch/bin/python"

    out = subprocess.run(
        [
            typilus_python,
            "-m",
            "run_typilus",
            repo.resolve(),
            out_dir.resolve(),
        ],
        cwd=typilus_path,
        env={"PYTHONPATH": "src"},
        capture_output=True,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"Typilus failed on {repo} with error: {out.stderr.decode()}"
        )


def analyze_typilus_predictions(
    typilus_outputs: list[JSON],
    repo_roots: list[Path],
    metrics: list[AccuracyMetric],
    common_labels_only: bool = True,
):
    assert_eq(
        len({r.name for r in repo_roots}),
        len(repo_roots),
        extra_message=lambda: "repo names must be unique",
    )

    # first, build the label map
    label_maps = pmap(collect_public_api_labels, repo_roots, desc="Collecting labels")

    # then, collect the prediction-label pairs
    pred_maps = list[dict[ModuleName, dict[CodePosition, str]]]()
    for out in typilus_outputs:
        pred_map = dict[ModuleName, dict[CodePosition, str]]()
        for file, preds in out.items():
            assert isinstance(file, str)
            assert isinstance(preds, list)
            if file.startswith("/"):
                file = file[1:]
            file_mod = PythonProject.rel_path_to_module_name(Path(file))
            submap = pred_map[file_mod] = dict()
            for pred in preds:
                line, col = pred["location"]
                submap[CodePosition(line, col)] = pred["pred"]
        pred_maps.append(pred_map)

    pred_list, label_list, cat_list = [], [], []
    n_missing = 0
    n_skipped_rare = 0
    for label_map, pred_map in zip(label_maps, pred_maps):
        for mod, labels in label_map.items():
            preds = pred_map.get(mod)
            n_labels = len(labels)
            if preds is None:
                if n_labels > 0:
                    logging.warning(f"Missing {n_labels} predictions for module: {mod}")
                    n_missing += n_labels
                continue
            for pos, linfo in labels.items():
                pred = preds.get(pos)
                if pred is None:
                    n_missing += 1
                    continue

                if (ltype := parse_type_expr(linfo.annot.annotation)) is None:
                    continue
                try:
                    pred = parse_type_str(pred)
                except SyntaxError:
                    pred = PythonType.Any()

                if common_labels_only and not metrics[0].is_common_type(ltype):
                    n_skipped_rare += 1
                    continue
                label_list.append(ltype)
                cat_list.append(linfo.cat)
                pred_list.append(pred)

    accs = dict[str, dict]()
    for m in metrics:
        accs[m.name] = sub_a = type_accuracies(pred_list, label_list, cat_list, m)
        if n_missing > 0:
            sub_a["n_missing"] = n_missing
        if n_skipped_rare > 0:
            sub_a["n_skipped_rare"] = n_skipped_rare
    return accs


TypilusSupportedSyntax = SupportedSyntax(
    pattern_match=False,
    union_types=False,
    basic_types=False,
    named_exprs=False,
)
