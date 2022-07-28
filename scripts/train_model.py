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
from termcolor import colored

config = TrainingConfig(
    quicktest=False, all_labels=True, stub_in_preamble=True, preamble_size=1024
)
gpu_id = 0
TypeCheckSettings.temp_path = f"GPU-{gpu_id}"

if config.quicktest:
    print(colored("Quicktest mode", "red"))

project_name = "test-SPOT" if config.quicktest else "SPOT"
train_ctx_args = config.train_ctx_args()
tc_args = TypeCheckArgs(check_in_isolation=config.check_in_isolation)

max_tokens_per_file = config.ctx_size
dec_args = DecodingArgs(
    sampling_max_tokens=8 * max_tokens_per_file,
    ctx_args=config.dec_ctx_args(),
)

datasets_name = get_dataset_name(config.get_preprocess_args())

src_datasets = load_src_datasets(
    datadir,
    datasets_name,
    data_reduction=config.data_reduction,
    quicktest=config.quicktest,
)
model_name = config.get_model_name()


# %%
# train the model
from spot.train import ModelTrainingArgs, train_spot_model, TypeCheckArgs
from spot.utils import run_long_task
import wandb
import torch

train_args = ModelTrainingArgs(
    train_ctx_args,
    dec_args,
    train_max_tokens=max_tokens_per_file,
    eval_max_tokens=2 * max_tokens_per_file,
    max_epochs=2,
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


wandb.log({f"test/accuracies": wandb_string(pretty_show_dict(r0_accs))})
wandb.finish()

# %%
# export the code with inlined predictions as HTML

from spot.visualization import export_preds_on_code, proj_root

export_preds = True

if export_preds:
    sub_ids = range(0, len(r0_eval.chunks), 10)
    export_preds_on_code(
        r0_eval.chunks[sub_ids],
        [r0_eval.predictions[i] for i in sub_ids],
        export_to=proj_root() / "caches" / "model_predictions",
    )
