# evaluate the model trained on fucntional dataset using
# incremental decoding.

# %%
import asyncio
import os
from typing import *

import torch
import wandb
from spot.function_dataset import data_project_from_dir
from spot.model import ModelWrapper
from spot.train import TrainingConfig, PreprocessArgs
from spot.type_env import AccuracyMetric
from spot.utils import (
    assert_eq,
    get_dataset_dir,
    get_eval_dir,
    get_gpu_id,
    get_model_dir,
    pickle_dump,
    pmap,
    pretty_show_dict,
    proj_root,
    run_long_task,
    write_file,
)
from spot.visualization import string_to_html
from termcolor import colored

os.chdir(proj_root())


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


# %%

# experiment configurations

config = TrainingConfig(
    quicktest=False,
    pre_args=PreprocessArgs(
        drop_env_types=False,
        add_override_usages=True,
    ),
    func_only=True,
)

gpu_id = get_gpu_id(0)
# model_name = "model-v5--TrainingConfig(drop_env_types=False)"
# model_name = "model-v6--TrainingConfig(drop_env_types=False)"
model_name = config.get_model_name()
dataset_name = "ManyTypes4Py"
# dataset_name = "SPOT-src"
experiment_name = dataset_name + ": " + model_name

print(colored(f"Use GPU: {gpu_id}", "green"))

# %%

# load model
model = ModelWrapper.from_pretrained(get_model_dir() / model_name)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded to {device}")

# load test projects
repos_dir = get_dataset_dir(dataset_name) / "repos" / "test"
test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]
test_projects = pmap(
    data_project_from_dir,
    test_repo_paths,
    desc="Loading test projects",
)
assert len(test_projects) > 0

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

decode_orders = {
    "double-traversal": DecodingOrders.DoubleTraversal(),
    "non-incr": DecodingOrders.IndependentOrder(),
    "random": DecodingOrders.RandomOrder(),
    "no-neighbors": DecodingOrders.IndependentOrder(),
    # "callee2caller": DecodingOrders.Callee2Caller(),
    # "random-twice": DecodingOrders.RandomTwice(),
    # "caller2callee": DecodingOrders.Caller2Callee(),
}

metrics = AccuracyMetric.default_metrics(model.common_type_names)
with run_long_task("Evaluating different decoding strategy"):
    results_dir = get_eval_dir(dataset_name, model_name)
    results_dir.mkdir(exist_ok=True, parents=True)
    print(colored(f"Results will be saved to: {str(results_dir)}", "green"))

    wandb.init(
        project="SPOT-eval",
        name=experiment_name,
        dir=str(results_dir),
    )

    evals = dict[str, EvalResult]()
    for oname, order in decode_orders.items():
        print(f"Evaluating decoding strategy: {oname}")
        pre_args = PreprocessArgs()
        if oname == "no-neighbors":
            pre_args.max_callers = 0
            pre_args.max_callees = 0
        evalr = asyncio.run(
            rctx.evaluate_on_projects(
                test_projects,
                pre_args,
                order,
            )
        )
        pickle_dump(results_dir / f"{oname}-EvalResult.pkl", evalr)
        evals[oname] = evalr
        accs = {m.name: evalr.error_analysis(None, m).accuracies for m in metrics}
        accs_str = pretty_show_dict(accs)
        write_file(results_dir / f"{oname}-accuracy.txt", accs_str)
        wandb.log({f"test/{oname}": wandb_string(accs_str)})
        print(f"========== {oname} ===========")
        print(accs_str)


# %%
from prettytable import PrettyTable
import prettytable as pt

common_type_names = ModelWrapper.load_common_type_names(get_model_dir() / model_name)
results_table = PrettyTable()
results_table.field_names = ["order", *(m.name for m in metrics)]
results_table.align = "r"
results_table.set_style(pt.SINGLE_BORDER)
results_table.float_format = ".4"

for oname in decode_orders:
    accs = [
        evals[oname].error_analysis(None, metric).accuracies[metric.name].acc
        for metric in metrics
    ]
    results_table.add_row([oname, *accs])

print(results_table)
write_file(results_dir / "comparison.txt", results_table.get_string())
