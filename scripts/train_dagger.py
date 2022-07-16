# %%

import os
import asyncio
from typing import *

from spot.utils import proj_root, get_data_dir

os.chdir(proj_root())

datadir = get_data_dir()

# %%
# experiment configurations

from spot.data import (
    SrcDataset,
    get_dataset_name,
    load_src_datasets,
    TypeCheckSettings,
)
from spot.model import CtxArgs, DecodingArgs, ModelSPOT, ModelWrapper
from spot.train import TrainingConfig, TypeCheckArgs

config = TrainingConfig(
    quicktest=False,
    all_labels=True,
    ctx_size=2048,
    left_margin=1024,
    right_margin=512,
)
gpu_id = 0
TypeCheckSettings.temp_path = f"DAgger-{gpu_id}"

print(f"quicktest={config.quicktest}")

project_name = "test-SPOT" if config.quicktest else "SPOT"
train_ctx_args = config.train_ctx_args()
tc_args = TypeCheckArgs(check_in_isolation=config.check_in_isolation)

dec_args = DecodingArgs(
    sampling_max_tokens=8 * config.ctx_size,
    ctx_args=config.dec_ctx_args(),
)

datasets_name = get_dataset_name(
    drop_comments=config.drop_comments,
    all_labels=config.all_labels,
)

model_name = "DAgger-model--" + config.as_name()

src_datasets = load_src_datasets(
    datadir,
    datasets_name,
    data_reduction=config.data_reduction,
    quicktest=config.quicktest,
)


# %%
# initialize the model
from spot.model import load_model_spot, DefaultTokenizer
from spot.model import ModelWrapper
from spot.dagger import DAggerModel
import torch

model = load_model_spot("Salesforce/codet5-base")
wrapper = ModelWrapper(model, DefaultTokenizer, dec_args)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
wrapper.to(device)
dmodel = DAggerModel(wrapper)


# %%
# pre-train evaluation
from spot.utils import pretty_print_dict

eval_r = asyncio.run(dmodel.eval_on_data(src_datasets["test"][0:50]))
pretty_print_dict(eval_r.accuracies)


# %%
# train the model
from spot.dagger import DAggerArgs
from spot.utils import run_long_task
import wandb

with run_long_task("DAgger training"):
    wandb.init(
        project=project_name,
        name=model_name,
        config=config.as_dict(),
        dir=str(datadir),
    )

    asyncio.run(
        dmodel.train_on_data(
            src_datasets,
            DAggerArgs(config.grad_accum_labels),
            log_fn=lambda t, x: wandb.log(x, step=t),
        )
    )

    save_path = datadir / f"checkpoints/saved/{model_name}"
    print(f"Saving trained model to: {save_path}")
    wrapper.save_pretrained(save_path)

# %%
# post-train full evaluation
from spot.utils import pretty_print_dict, pretty_show_dict
from spot.visualization import string_to_html


eval_r = asyncio.run(dmodel.eval_on_data(src_datasets["test"]))
pretty_print_dict(eval_r.accuracies)


def wandb_string(s: str):
    return wandb.Html(string_to_html(s))


wandb.log({"test/accuracies": wandb_string(pretty_show_dict(eval_r.accuracies))})
