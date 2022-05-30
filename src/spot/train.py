import os
import pickle
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import Dataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from regex import D
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

import wandb
from spot.data import (
    ChunkedDataset,
    R1_srcs_from_preds,
    SrcChunkInfo,
    SrcDataset,
    get_dataset_name,
    get_model_name,
    load_src_datasets,
)
from spot.model import (
    CtxArgs,
    DecodingArgs,
    ModelSPOT,
    ModelTrainingArgs,
    ModelWrapper,
    TokenizerSPOT,
    dynamic_dataloader,
)
from spot.type_env import PythonType
from spot.utils import *


def train_spot_model(
    src_datasets: dict[str, SrcDataset],
    model_name: str,
    dec_args: DecodingArgs,
    train_args: ModelTrainingArgs,
    record_batches: bool,
    gpus: list[int] = [0],
    quicktest=False,
    use_small_model=False,
    ckpt_interval: int = 500,
) -> Tuple[ModelWrapper, dict]:
    os.chdir(proj_root())

    datadir = Path(os.getenv("datadir", "data"))

    model_path = (
        "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
    )
    tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained(model_path)
    lit_model = TrainModelWrapper(model_path, dec_args, record_batches=record_batches)
    wrapper = lit_model.wrapper

    chunks: dict[str, ChunkedDataset] = {}
    with run_long_task("Preparing chunked datasets", notify=False):
        for n in ["valid", "test", "train"]:
            src = src_datasets[n]
            chunks[n] = src.to_chunks(tokenizer, dec_args.ctx_args)

    wandb_logger = WandbLogger(
        project=model_name,
        save_dir=str(datadir),
    )
    wandb_logger.log_hyperparams(
        {
            "decoding_args": dec_args,
            "train_args": train_args,
        }
    )

    val_interval = 1 if quicktest else 500

    running_dir = datadir / "checkpoints/lit-running" / model_name
    if running_dir.exists():
        shutil.rmtree(running_dir)
    running_dir.mkdir(parents=True, exist_ok=True)
    ckpt_interval = 1 if quicktest else ckpt_interval

    extra_callbacks = []
    if record_batches:
        extra_callbacks.append(
            ModelCheckpoint(
                dirpath=running_dir / "models",
                save_top_k=-1,
                filename="{step}",
                every_n_train_steps=ckpt_interval,
                save_weights_only=True,
            )
        )

    checkpoint_cb = ModelCheckpoint(
        dirpath=running_dir,
        save_top_k=2,
        monitor="valid/loss",
        mode="min",
        save_on_train_epoch_end=False,
        verbose=quicktest,
    )

    trainer = pl.Trainer(
        default_root_dir=str(running_dir),
        # fast_dev_run=6 if quicktest else False,
        # log_every_n_steps=500,
        accelerator="gpu" if gpus else "cpu",
        gpus=gpus,
        precision=16,
        max_epochs=train_args.max_epochs,
        logger=wandb_logger,
        val_check_interval=val_interval,
        callbacks=[
            *extra_callbacks,
            checkpoint_cb,
            EarlyStopping("valid/loss", mode="min", verbose=quicktest),
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=train_args.accumulate_grad_batches,
    )

    collate_fn = DataCollatorForSeq2Seq(lit_model.tokenizer, lit_model.model)
    train_dataloader = dynamic_dataloader(
        cast(Any, chunks["train"].data),
        max_tokens=train_args.train_max_tokens,
        collate_fn=collate_fn,
        shuffle=True,
    )
    valid_dataloader = dynamic_dataloader(
        cast(Any, chunks["valid"].data),
        max_tokens=train_args.eval_max_tokens,
        collate_fn=collate_fn,
        shuffle=True,  # doesn't hurt
    )

    warnings.filterwarnings("ignore", "The dataloader.*does not have many workers.*")

    with run_long_task(f"Training {model_name}"):
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

    extra = dict[str, Any]()
    save_dir = datadir / "checkpoints/lit-saved" / model_name

    final_eval = trainer.validate(model=lit_model, dataloaders=valid_dataloader)[0]
    wandb_logger.finalize("Finished.")
    wandb.finish()

    try:
        if (
            best_loss := checkpoint_cb.best_model_score
        ) is not None and best_loss < final_eval["valid/loss"]:
            print(
                f"Loading best model with score {best_loss} from: {checkpoint_cb.best_model_path}"
            )
            wrapper = TrainModelWrapper.load_from_checkpoint(
                checkpoint_cb.best_model_path
            ).wrapper
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        wrapper.save_pretrained(save_dir)
        if record_batches:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extra["batch_ids"] = lit_model.batch_ids
            with run_long_task("Generating R1 datasets", notify=False):
                print(f"Generating R1 dataset: train")
                R1_src_datasets = dict[str, SrcDataset]()
                R1_src_datasets["train"] = R1_srcs_from_ckpts(
                    tokenizer,
                    src_datasets["train"],
                    chunks["train"],
                    lit_model.batch_ids,
                    ckpt_dir=running_dir / "models",
                    ckpt_interval=ckpt_interval,
                    max_workers=wrapper.args.max_workers,
                    device=device,
                )
                for n in ["valid", "test"]:
                    print(f"Generating R1 dataset: {n}")
                    preds = wrapper.to(device).predict(chunks[n].data, {})
                    R1_src_datasets[n] = R1_srcs_from_preds(
                        tokenizer,
                        src_datasets[n],
                        chunks[n].chunks_info,
                        chunks[n].files,
                        preds,
                        max_workers=wrapper.args.max_workers,
                    )
            extra["R1-src_datasets"] = R1_src_datasets
            pickle_dump(save_dir / "extra.pkl", extra)

            shutil.rmtree(running_dir)
    except Exception as e:
        logging.error(
            "Error encountered after training, returning partial results... Error:\n", e
        )

    return wrapper, extra


def R1_srcs_from_ckpts(
    tokenizer: TokenizerSPOT,
    r0_src: SrcDataset,
    cdata: ChunkedDataset,
    chunk_ids: list[list[int]],
    ckpt_dir: Path,
    ckpt_interval: int,
    max_workers: int,
    device,
    tqdm_args={},
) -> SrcDataset:
    chunks_info = list[SrcChunkInfo]()
    model_preds = list[list[PythonType]]()
    for i in tqdm(
        range(ckpt_interval, len(chunk_ids), ckpt_interval),
        desc="R1_srcs_from_ckpts",
        **tqdm_args,
    ):
        ids = list(seq_flatten(chunk_ids[i : i + ckpt_interval]))
        wrapper = TrainModelWrapper.load_from_checkpoint(
            str(ckpt_dir / "step={}.ckpt".format(i))
        ).wrapper
        wrapper.to(device)
        try:
            data_sub = cdata[ids]
        except IndexError as e:
            raise IndexError(
                f"ids: {ids},\nchunk_ids: {cdata.data['chunk_id']}\ncdata: {cdata}"
            ) from e
        chunks_info.extend(data_sub.chunks_info)
        preds = wrapper.predict(data_sub.data, tqdm_args=tqdm_args)
        model_preds.extend(preds)
    return R1_srcs_from_preds(
        tokenizer,
        r0_src,
        chunks_info,
        cdata.files,
        model_preds,
        max_workers=max_workers,
        tqdm_args=tqdm_args,
    )


class TrainModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the SPOT model."

    def __init__(
        self,
        model_checkpoint: str | Path,
        args: DecodingArgs,
        *,
        record_batches: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: ModelSPOT = ModelSPOT.from_pretrained(model_checkpoint)
        self.tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained(model_checkpoint)
        self.args = args
        self.wrapper = ModelWrapper(
            self.model, self.tokenizer, args, EmptyLoggingMonitor()
        )
        self.record_batches = record_batches

    def on_fit_start(self):
        # maps chunk id to the initial predictions made for that chunk immediately
        # before the model was trained on it
        if self.record_batches:
            self.batch_ids: list[list[int]] = []

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [
                    p
                    for pn, p in self.model.named_parameters()
                    if not any(n in pn for n in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for pn, p in self.model.named_parameters()
                    if any(n in pn for n in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(grouped_params, lr=2e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.2)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        if self.record_batches and self.current_epoch == 0:
            self.batch_ids.append(batch["chunk_id"].tolist())

        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("train/loss", loss.item())
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("valid/loss", loss.item())


def concat_batches(batches: list[dict], keys: list[str]) -> dict:
    return {k: torch.concat([b[k] for b in batches]) for k in keys}
