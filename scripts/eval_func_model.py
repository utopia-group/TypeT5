# evaluate the model trained on fucntional dataset using
# incremental decoding.

# %%
import asyncio
import os
from shutil import rmtree
from typing import *

import torch
from termcolor import colored
import wandb

from spot.data import GitRepo
from spot.function_dataset import data_project_from_dir
from spot.model import ModelWrapper
from spot.utils import (
    assert_eq,
    get_data_dir,
    get_model_dir,
    pickle_dump,
    pickle_load,
    pmap,
    pretty_show_dict,
    proj_root,
    run_long_task,
    write_file,
)
from spot.visualization import string_to_html

os.chdir(proj_root())

datadir = get_data_dir()
modeldir = get_model_dir()


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


# %%

# experiment configurations

gpu_id = 0
model_name = "model-v4--TrainingConfig(func_only=True, drop_env_types=False, left_margin=1536, preamble_size=768, right_margin=2048)"

print(colored(f"Use GPU: {gpu_id}", "green"))

repos_split: dict[str, list[GitRepo]] = pickle_load(
    proj_root() / "data/repos_split.pkl"
)

wandb.init(
    project="SPOT-eval",
    name=model_name,
    dir=str(datadir),
)

# %%

# load model
model = ModelWrapper.from_pretrained(modeldir / f"checkpoints/lit-saved/{model_name}")
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded to {device}")

# load test projects
repos_dir = datadir / "SPOT-data/repos"
test_repo_paths = [r.repo_dir(repos_dir) for r in repos_split["test"]]
test_projects = pmap(
    data_project_from_dir,
    test_repo_paths,
    desc="Loading test projects",
)

# %%

from spot.function_decoding import (
    DecodingOrders,
    EvalResult,
    PreprocessArgs,
    RolloutCtx,
)

ctx_args = model.args.ctx_args
model.args.sampling_max_tokens = ctx_args.ctx_size
model.args.do_sample = False
model.args.num_beams = 10
model.args.tokens_per_type = 16

rctx = RolloutCtx(model=model)
pre_args = PreprocessArgs()

decode_orders = {
    "non-incr": DecodingOrders.IndependentOrder(),
    "random": DecodingOrders.RandomOrder(),
    "caller2callee": DecodingOrders.Caller2Callee(),
    "double-traversal": DecodingOrders.DoubleTraversal(),
}

with run_long_task("Evaluating different decoding strategy"):
    results_dir = proj_root() / "data/evaluation" / model_name
    if results_dir.exists():
        rmtree(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    print(colored(f"Results will be saved to: {str(results_dir)}", "green"))

    results = dict[str, EvalResult]()
    for oname, order in decode_orders.items():
        evalr = asyncio.run(
            rctx.evaluate_on_projects(
                test_projects,
                pre_args,
                order,
                common_type_names=model.common_type_names,
            )
        )
        results[oname] = evalr
        accs_str = pretty_show_dict(evalr.accuracies)
        write_file(results_dir / f"{oname}.txt", accs_str)
        wandb.log({f"test/{oname}": wandb_string(accs_str)})
        print(f"========== {oname} ===========")
        print(accs_str)

    pickle_dump(results_dir / "results.pkl", results)

import prettytable as pt

# %%
from prettytable import PrettyTable

results_table = PrettyTable()
results_table.field_names = ["order", "full acc", "partial acc"]
results_table.align = "r"
results_table.set_style(pt.SINGLE_BORDER)
results_table.float_format = ".4"

for oname, order in decode_orders.items():
    evalr = results[oname]
    results_table.add_row(
        [oname, evalr.accuracies["full_acc"].acc, evalr.accuracies["partial_acc"].acc]
    )

print(results_table)

acc_counts = {oname: r.accuracies["full_acc"].n_total for oname, r in results.items()}
assert_eq(*acc_counts.values(), extra_message=lambda: f"Accuracy counts: {acc_counts}")
