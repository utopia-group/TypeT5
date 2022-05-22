import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from regex import D
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

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
from spot.type_env import PythonType
from spot.utils import *


def train_spot_model(
    spot_round: int,
    drop_comments: bool,
    ctx_args: CtxArgs,
    train_args: ModelTrainingArgs,
    data_reduction: int = 1,
    gpus: list[int] = [0],
    collect_init_preds: bool = None,
    quicktest=False,
) -> Tuple[ModelWrapper, dict]:
    os.chdir(proj_root())

    datadir = Path(os.getenv("datadir", "data"))
    repos_dir = datadir / "SPOT-data/repos"

    src_datasets_path = (
        datadir
        / f"SPOT-data"
        / get_dataset_name(drop_comments=drop_comments, spot_round=spot_round)
    )
    src_datasets = dict[str, SrcDataset]()
    for n in ["train", "valid", "test"]:
        with open(src_datasets_path / f"{n}.pkl", "rb") as f:
            src_datasets[n] = pickle.load(f)
            src_datasets[n].repos_root = repos_dir

    tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained("Salesforce/codet5-base")

    model_name = get_model_name(
        spot_round=spot_round,
        drop_comments=drop_comments,
        ctx_args=ctx_args,
        data_reduction=data_reduction,
        quicktest=quicktest,
    )
    print("Model name: ", model_name)

    model_path = "Salesforce/codet5-base"

    dec_args = DecodingArgs(
        sampling_batch_size=128,
        ctx_args=ctx_args,
        max_workers=20,
    )

    if collect_init_preds is None:
        collect_init_preds = spot_round == 0
    lit_model = TrainModelWrapper(
        model_path, dec_args, collect_init_preds=collect_init_preds
    )
    wrapper = lit_model.wrapper

    chunks: dict[str, ChunkedDataset] = {}
    with run_long_task("Preparing chunked datasets", notify=False):
        for n in ["valid", "train"]:
            src = src_datasets[n][:35] if quicktest else src_datasets[n]
            chunks[n] = src.to_chunks(tokenizer, ctx_args, max_workers=20)

    n_train = len(chunks["train"].data) // data_reduction
    chunks["train"] = chunks["train"][:n_train]

    wandb_logger = WandbLogger(
        project=model_name,
        save_dir=str(datadir),
    )
    wandb_logger.log_hyperparams(
        {"r0_decoding_args": dec_args, "r0_train_args": train_args}
    )

    val_interval = 1 if quicktest else 500

    save_dir = datadir / "checkpoints/lit-running" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        save_top_k=2,
        monitor="valid/loss",
        mode="min",
        save_on_train_epoch_end=False,
        verbose=quicktest,
    )

    trainer = pl.Trainer(
        default_root_dir=save_dir,
        # fast_dev_run=6 if quicktest else False,
        # log_every_n_steps=500,
        accelerator="gpu" if gpus else "cpu",
        gpus=gpus,
        precision=16,
        max_epochs=train_args.max_epochs,
        logger=wandb_logger,
        val_check_interval=val_interval,
        callbacks=[
            checkpoint_cb,
            EarlyStopping("valid/loss", mode="min", verbose=quicktest),
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=train_args.accumulate_grad_batches,
    )

    collate_fn = DataCollatorForSeq2Seq(lit_model.tokenizer, lit_model.model)
    train_dataloader = DataLoader(
        cast(Any, chunks["train"].data),
        batch_size=train_args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        cast(Any, chunks["valid"].data),
        batch_size=train_args.eval_batch_size,
        collate_fn=collate_fn,
    )
    warnings.filterwarnings("ignore", "The dataloader.*does not have many workers.*")

    with run_long_task(f"Training {model_name}"):
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

    extra = dict()
    if collect_init_preds:
        extra["init_preds"] = lit_model.init_preds
        pickle_dump(
            datadir / "checkpoints/lit-saved" / model_name / "init_preds.pkl",
            lit_model.init_preds,
        )

    final_eval = trainer.validate(model=lit_model, dataloaders=valid_dataloader)[0]

    try:
        if (
            checkpoint_cb.best_model_score is not None
            and checkpoint_cb.best_model_score < final_eval["valid/loss"]
        ):
            print("Loading best model from: ", checkpoint_cb.best_model_path)
            wrapper = TrainModelWrapper.load_from_checkpoint(
                checkpoint_cb.best_model_path
            ).wrapper

        wandb_logger.finalize("Finished.")
        wrapper.save_pretrained(datadir / "checkpoints/lit-saved" / model_name)
        if quicktest:
            shutil.rmtree(save_dir)
    except Exception as e:
        logging.error("Error encountered during final stages: ", e)

    return wrapper, extra


class TrainModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the SPOT model."

    def __init__(
        self,
        model_checkpoint: str | Path,
        args: DecodingArgs,
        *,
        collect_init_preds: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: ModelSPOT = ModelSPOT.from_pretrained(model_checkpoint)
        self.tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained(model_checkpoint)
        self.args = args
        self.wrapper = ModelWrapper(
            self.model, self.tokenizer, args, EmptyLoggingMonitor()
        )
        self.collect_init_preds = collect_init_preds

    def on_fit_start(self):
        # maps chunk id to the initial predictions made for that chunk immediately
        # before the model was trained on it
        if self.collect_init_preds:
            self.init_preds: dict[int, list[PythonType]] = {}
            self.train_batch_buffer: list[dict] = []
            self.train_batch_buffer_size: int = 0

    def on_train_epoch_end(self):
        if self.collect_init_preds:
            self._process_buffer()

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
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.total_train_steps)
        # scheduler_dict = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.2)
        return [optimizer], [lr_scheduler]

    def _process_buffer(self):
        if self.train_batch_buffer_size == 0:
            return
        shadow_batch = concat_batches(
            self.train_batch_buffer, keys=["input_ids", "chunk_id", "n_labels"]
        )
        self.train_batch_buffer = []
        self.train_batch_buffer_size = 0
        preds = self.wrapper.predict_on_batch(shadow_batch, do_sample=True)
        for id, ps in zip(shadow_batch["chunk_id"].tolist(), preds):
            assert id not in self.init_preds, f"Repeating chunk id: {id}"
            self.init_preds[id] = ps

    def training_step(self, batch, batch_idx):
        if self.collect_init_preds and self.current_epoch == 0:
            preds = self.wrapper.predict_on_batch(batch)
            for id, ps in zip(batch["chunk_id"].tolist(), preds):
                assert id not in self.init_preds, f"Repeating chunk id: {id}"
                self.init_preds[id] = ps

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
        self.log("valid/loss", outputs.loss)


def concat_batches(batches: list[dict], keys: list[str]) -> dict:
    return {k: torch.concat([b[k] for b in batches]) for k in keys}
