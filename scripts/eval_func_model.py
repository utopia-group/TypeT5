# evaluate the model trained on fucntional dataset using
# incremental decoding.

# %%
import asyncio
import copy
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
    get_modified_args,
    pickle_dump,
    pickle_load,
    pmap,
    pretty_show_dict,
    proj_root,
    run_long_task,
    write_file,
)
from spot.visualization import string_to_html
from termcolor import colored
from spot.experiments.typet5 import TypeT5Configs

os.chdir(proj_root())


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


# %%

# experiment configurations

load_results = False
use_oracle = False
gpu_id = get_gpu_id(0)
train_config = TypeT5Configs.NoSequential

model_name = train_config.get_model_name()
# model_name = (
#     "model-v7--TrainingConfig(drop_env_types=False, add_implicit_rel_imports=True)"
# )
# dataset_name = "ManyTypes4Py"
dataset_name = "InferTypes4Py"

test_pre_args = train_config.pre_args
oracle_tag = "(use-oracle) " if use_oracle else ""
# group_tag = "(implicit_imports, new) "
group_tag = "(ablation) "
experiment_name = oracle_tag + group_tag + model_name

print(colored(f"Use GPU: {gpu_id}", "green"))

# %%

# load model
model = ModelWrapper.from_pretrained(get_model_dir() / model_name)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded:", model_name)

# load test projects
repos_dir = get_dataset_dir(dataset_name) / "repos" / "test"
test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]
if not load_results:
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
    RolloutCtx,
)
from spot.experiments.typet5 import accs_as_table_row

ctx_args = model.args.ctx_args
ctx_args.max_labels = 16
model.args.sampling_max_tokens = ctx_args.ctx_size
model.args.do_sample = False
model.args.num_beams = 16
model.args.tokens_per_type = 16

rctx = RolloutCtx(model=model)

decode_orders = {
    # "double-traversal": DecodingOrders.DoubleTraversal(),
    # "reverse-double-traversal": DecodingOrders.Reversed(
    #     DecodingOrders.DoubleTraversal()
    # ),
    "non-incr": DecodingOrders.IndependentOrder(),
    # "random": DecodingOrders.RandomOrder(),
    # "no-neighbors": DecodingOrders.IndependentOrder(),
    # "callee2caller": DecodingOrders.Callee2Caller(),
    # "caller2callee": DecodingOrders.Caller2Callee(),
    # "random-twice": DecodingOrders.RandomTwice(),
}

metrics = AccuracyMetric.default_metrics(model.common_type_names)
with run_long_task("Evaluating different decoding strategy", notify=not load_results):
    results_dir = get_eval_dir(dataset_name, experiment_name)
    results_dir.mkdir(exist_ok=True, parents=True)
    print(colored(f"Results will be saved to: {str(results_dir)}", "green"))

    if not load_results:
        wandb.init(
            project="SPOT-eval",
            name=dataset_name + ": " + experiment_name,
            dir=str(results_dir),
            config=get_modified_args(model.args),
        )

    evals = dict[str, EvalResult]()
    for oname, order in decode_orders.items():
        result_path = results_dir / f"{oname}-EvalResult.pkl"
        if not load_results:
            print(f"Evaluating decoding strategy: {oname}")
            pre_args = copy.deepcopy(test_pre_args)
            if oname == "no-neighbors":
                pre_args.max_callers = 0
                pre_args.max_callees = 0
            evalr = asyncio.run(
                rctx.evaluate_on_projects(
                    test_projects,  # type: ignore
                    pre_args,
                    order,
                    use_oracle=use_oracle,
                )
            )
            pickle_dump(result_path, evalr)
        else:
            if not result_path.exists():
                print(f"Result file not found, skip: {result_path}")
                continue
            evalr = pickle_load(result_path)
        evals[oname] = evalr
        accs = {m.name: evalr.error_analysis(None, m).accuracies for m in metrics}
        accs_str = pretty_show_dict(accs)
        write_file(results_dir / f"{oname}-accuracy.txt", accs_str)
        if not load_results:
            wandb.log({f"test/{oname}": wandb_string(accs_str)})
        print(f"========== {oname} ===========")
        print(accs_str)
        accs_as_table_row(accs)


# %%
from prettytable import PrettyTable
import prettytable as pt

common_type_names = ModelWrapper.load_common_type_names(get_model_dir() / model_name)
results_table = PrettyTable()
results_table.field_names = ["order", *(m.name for m in metrics)]
results_table.align = "r"
results_table.set_style(pt.SINGLE_BORDER)
results_table.float_format = ".4"

for oname in evals:
    accs = [
        evals[oname].error_analysis(None, metric).accuracies[metric.name].acc
        for metric in metrics
    ]
    results_table.add_row([oname, *accs])

print(results_table)
write_file(results_dir / "comparison.txt", results_table.get_string())
