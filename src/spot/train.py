import os
import pickle
from pathlib import Path
from typing import *

import pandas as pd
import plotly.express as px
import torch

import wandb
from spot.data import ChunkedDataset, SrcDataset, get_dataset_name, get_model_name
from spot.model import (
    CtxArgs,
    DecodingArgs,
    ModelSPOT,
    ModelTrainingArgs,
    ModelWrapper,
    TokenizerSPOT,
)
from spot.utils import TaskLoggingMonitor, cst, proj_root, run_long_task, tqdm


def train_r0_model(
    drop_comments: bool,
    ctx_args: CtxArgs,
    train_args: ModelTrainingArgs,
    data_reduction: int = 1,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> ModelWrapper:
    os.chdir(proj_root())

    datadir = Path(os.getenv("datadir", "data"))
    repos_dir = datadir / "SPOT-data/repos"

    src_datasets_path = (
        datadir / f"SPOT-data" / get_dataset_name(drop_comments=drop_comments)
    )
    src_datasets_path.mkdir(parents=True, exist_ok=True)
    src_datasets = dict[str, SrcDataset]()
    for n in ["train", "valid", "test"]:
        with open(src_datasets_path / f"{n}.pkl", "rb") as f:
            src_datasets[n] = pickle.load(f)
            src_datasets[n].repos_root = repos_dir

    tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained("Salesforce/codet5-base")

    model_name = get_model_name(
        drop_comments=drop_comments, ctx_args=ctx_args, data_reduction=data_reduction
    )
    print("R0 model name: ", model_name)

    model_path = "Salesforce/codet5-base"

    model: ModelSPOT = ModelSPOT.from_pretrained(model_path).to(device)
    r0_monitor = TaskLoggingMonitor("R0")
    r0_args = DecodingArgs(
        sampling_batch_size=128,
        ctx_args=ctx_args,
        max_workers=20,
    )
    wrapper = ModelWrapper(model, tokenizer, r0_args, r0_monitor)

    chunks: dict[str, ChunkedDataset] = {}
    with run_long_task("Preparing chunked datasets", notify=False):
        for n in ["valid", "train"]:
            chunks[n] = src_datasets[n].to_chunks(tokenizer, ctx_args, max_workers=20)

    n_train = len(chunks["train"].data) // data_reduction
    chunks["train"] = chunks["train"][:n_train]

    trainer = wrapper.build_trainer(
        datadir / "checkpoints" / model_name,
        train_args,
        dataset=chunks["train"].data,
        eval_dataset=chunks["valid"].data,
    )

    wandb.init(
        project=model_name,
        dir=str(datadir),
        config={"r0_decoding_args": r0_args, "r0_train_args": train_args},
    )

    with run_long_task(f"Training {model_name}"):
        init_perf = trainer.evaluate(max_length=r0_args.generation_max_length)  # type: ignore
        print("initial eval loss:", init_perf)
        trainer.train()

    wandb.log({"time_stats": r0_monitor.timer.total_times()})

    final_perf = trainer.evaluate(max_length=r0_args.generation_max_length)  # type: ignore
    print("final eval loss:", final_perf)
    wandb.finish()

    wrapper.save_pretrained(datadir / "checkpoints/saved" / model_name)
    return wrapper


if __name__ == "__main__":
    train_r0_model(
        drop_comments=True,
        ctx_args=CtxArgs(
            ctx_size=1024,
            left_margin=256 + 128,
            right_margin=256 - 128,
            types_in_ctx=False,
        ),
        train_args=ModelTrainingArgs(
            train_batch_size=8,
            eval_batch_size=64,
            max_epochs=3,
        ),
    )
