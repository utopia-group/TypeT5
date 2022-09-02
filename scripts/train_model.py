# %%
import os
from typing import *

from spot.utils import proj_root, get_data_dir

os.chdir(proj_root())

datadir = get_data_dir()

# %%
# experiment configurations

from spot.data import (
    get_dataset_name,
    load_src_datasets,
    TypeCheckSettings,
)
from spot.model import CtxArgs, DecodingArgs, ModelSPOT, ModelWrapper
from spot.train import TrainingConfig, TypeCheckArgs
from spot.tokenized_src import PreprocessArgs
from termcolor import colored

gpu_id = 1
eval_only = False


config = TrainingConfig(
    quicktest=False,
    pre_args=PreprocessArgs(
        drop_env_types=False,
        stub_in_preamble=False,
    ),
    preamble_size=512 + 256,
    left_margin=1024 + 512,
    right_margin=2048,
    func_only=True,
)

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

datasets_name = get_dataset_name(config.pre_args, config.func_only)

src_datasets = load_src_datasets(
    datadir,
    datasets_name,
    data_reduction=config.data_reduction,
    quicktest=config.quicktest,
)
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
        dir=str(datadir),
    )

    with run_long_task("Training spot model"):
        wrapper, r0_extra = train_spot_model(
            src_datasets,
            model_name,
            train_args=train_args,
            record_batches=False,
            gpus=[gpu_id],
            quicktest=config.quicktest,
            use_small_model=config.use_small_model,
            use_early_stop=False,
        )
else:
    wrapper = ModelWrapper.from_pretrained(
        datadir / f"checkpoints/lit-saved/{model_name}"
    )

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
wrapper.to(device)


# %%
# model evaluation

from spot.train import evaluate_model
from spot.utils import PickleCache
from spot.visualization import pretty_print_dict

bs_args = DecodingArgs(
    sampling_max_tokens=max_tokens_per_file,
    ctx_args=config.dec_ctx_args(),
    do_sample=False,
    num_beams=16,
)
wrapper.args = bs_args

eval_cache = PickleCache(datadir / f"checkpoints/lit-saved/{model_name}/eval_cache")
# eval_cache.clear()
r0_eval = eval_cache.cached(
    "dataset_pred.pkl",
    lambda: wrapper.eval_on_dataset(src_datasets["test"]),
)
r0_accs = r0_eval.accuracies
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
    wandb.finish()

# %%
# export the code with inlined predictions as HTML

from spot.visualization import export_preds_on_code, proj_root

export_preds = True

if export_preds:
    max_samples = 1000
    sub_ids = range(0, len(r0_eval.chunks), len(r0_eval.chunks) // max_samples)
    export_preds_on_code(
        r0_eval.chunks[sub_ids],
        [r0_eval.predictions[i] for i in sub_ids],
        export_to=proj_root() / "caches" / "model_predictions" / model_name,
    )
    print("Model predictions exported to 'caches/model_predictions'")
