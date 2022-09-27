# %%
import os
from typing import *
from spot.experiments.typet5 import TypeT5Configs

from spot.utils import *

os.chdir(proj_root())

# %%
# experiment configurations

from spot.data import (
    create_tokenized_srcsets,
    get_tk_dataset_name,
    load_tokenized_srcsets,
    TypeCheckSettings,
)
from spot.model import CtxArgs, DecodingArgs, ModelSPOT, ModelWrapper
from spot.train import TrainingConfig, TypeCheckArgs
from spot.tokenized_src import PreprocessArgs
from termcolor import colored

gpu_id = get_gpu_id(0)
eval_only = False
recreate_dataset = False


config = TypeT5Configs.NoUsers

TypeCheckSettings.temp_path = f"GPU-{gpu_id}"
print(colored(f"Use GPU: {gpu_id}", "green"))

if config.quicktest:
    print(colored("Quicktest mode", "red"))
if eval_only:
    print(colored("Model Evaluating Mode", "blue"))

project_name = "test-SPOT" if config.quicktest else "SPOT"
train_ctx_args = config.train_ctx_args()
if train_ctx_args.window_size < 100:
    print(
        colored(
            f"[Warning] window size is very small: {train_ctx_args.window_size}", "red"
        )
    )
tc_args = TypeCheckArgs(check_in_isolation=config.check_in_isolation)

max_tokens_per_file = config.ctx_size
dec_args = DecodingArgs(
    sampling_max_tokens=8 * max_tokens_per_file,
    ctx_args=config.dec_ctx_args(),
)

dataset = config.trained_on
print("Model will be trained on dataset:", colored(dataset, "blue"))

sdata_name = get_tk_dataset_name(
    dataset, config.pre_args, config.func_only, data_reduction=config.data_reduction
)
sdata_path = get_dataroot() / "TokenizedSrcSets" / sdata_name
if recreate_dataset or not sdata_path.exists():
    create_tokenized_srcsets(
        dataset,
        sdata_path,
        func_only=config.func_only,
        pre_args=config.pre_args,
        data_reduction=config.data_reduction,
    )

tk_dataset = load_tokenized_srcsets(
    sdata_path,
    quicktest=config.quicktest,
)
print("Training set stats:")
tk_dataset["train"].print_stats()

model_name = config.get_model_name()
print(colored(f"Training model: {model_name}", "green"))

# %%
# train the model
from spot.train import ModelTrainingArgs, train_spot_model, TypeCheckArgs
from spot.utils import run_long_task
import wandb
import torch

if not eval_only:
    train_args = ModelTrainingArgs(
        train_ctx_args,
        dec_args,
        train_max_tokens=max_tokens_per_file,
        eval_max_tokens=2 * max_tokens_per_file,
        max_epochs=1,
        tc_args=tc_args,
    )

    wandb.init(
        project=project_name,
        name=model_name,
        config=config.as_dict(),
        dir=str(get_dataroot()),
    )

    with run_long_task("Training spot model"):
        wrapper = train_spot_model(
            tk_dataset,
            model_name,
            train_args=train_args,
            gpus=[gpu_id],
            quicktest=config.quicktest,
            use_small_model=config.use_small_model,
            use_early_stop=False,
        )
else:
    wrapper = ModelWrapper.from_pretrained(get_model_dir() / model_name)

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
wrapper.to(device)


# %%
# model evaluation

from spot.utils import PickleCache
from spot.visualization import pretty_print_dict
from spot.type_env import AccuracyMetric

bs_args = DecodingArgs(
    sampling_max_tokens=max_tokens_per_file,
    ctx_args=config.dec_ctx_args(),
    do_sample=False,
    num_beams=16,
)
wrapper.args = bs_args

eval_cache = PickleCache(get_eval_dir(dataset, model_name) / "eval_cache")
# eval_cache.clear()
eval_r = eval_cache.cached(
    "dataset_pred.pkl",
    lambda: wrapper.eval_on_dataset(tk_dataset["test"]),
)
common_names = wrapper.common_type_names
metrics = AccuracyMetric.default_metrics(common_names)
r0_accs = {m.name: eval_r.accuracies(m) for m in metrics}
print("Accuracies on all user annotations:")
pretty_print_dict(r0_accs)


# %%
# close wandb
from spot.utils import pretty_show_dict
from spot.visualization import string_to_html
import wandb


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


if not eval_only:
    wandb.log({f"test/accuracies": wandb_string(pretty_show_dict(r0_accs))})

# %%
# compute accuracies on the public APIs
from spot.function_decoding import PreprocessArgs
from spot.static_analysis import SignatureErrorAnalysis
from spot.function_dataset import data_project_from_dir, sigmap_from_file_predictions

repos_dir = get_dataset_dir(dataset) / "repos" / "test"
test_repo_paths = [f for f in repos_dir.iterdir() if f.is_dir()]
test_projects = pmap(
    data_project_from_dir,
    test_repo_paths,
    desc="Loading test projects",
)

eval_r = eval_r
pred_map, label_map = sigmap_from_file_predictions(eval_r, test_projects, repos_dir)
api_accs = {
    m.name: SignatureErrorAnalysis(pred_map, label_map, m).accuracies
    for m in AccuracyMetric.default_metrics(common_names)
}

print("Accuracies on public APIs:")
pretty_print_dict(api_accs)
if not eval_only:
    wandb.log({f"test/api_accuracies": wandb_string(pretty_show_dict(api_accs))})

# %%
# export the code with inlined predictions as HTML

from spot.visualization import export_preds_on_code, proj_root

export_preds = True

if export_preds:
    max_samples = 500
    sample_every = max(1, len(eval_r.chunks) // max_samples)
    sub_ids = range(0, len(eval_r.chunks), sample_every)
    export_to = proj_root() / "caches" / "model_predictions" / model_name
    export_preds_on_code(
        eval_r.chunks[sub_ids],
        [eval_r.predictions[i] for i in sub_ids],
        export_to=export_to,
        metric=AccuracyMetric(common_names),
    )
    print(f"Model predictions exported to '{export_to}'")
