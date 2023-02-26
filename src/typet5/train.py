import os
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from transformers import DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput

from .data import ChunkedDataset, TokenizedSrcSet
from .model import (
    CtxArgs,
    DecodingArgs,
    ModelType,
    ModelWrapper,
    TokenizerType,
    dynamic_dataloader,
)
from .tokenized_src import PreprocessArgs
from .type_check import TypeCheckArgs
from .utils import *


@dataclass
class ModelTrainingArgs:
    train_ctx_args: CtxArgs
    dec_args: DecodingArgs
    train_max_tokens: int
    eval_max_tokens: int
    max_epochs: int
    tc_args: TypeCheckArgs
    accumulate_grad_batches: int | dict | None = None


class TrainingConfig(NamedTuple):
    quicktest: bool = False
    func_only: bool = True  # whether to use functional dataset format
    pre_args: PreprocessArgs = PreprocessArgs()
    trained_on: str = "ManyTypes4Py"
    data_reduction: int = 1
    check_in_isolation: bool = False  # DAgger
    inline_prev_gold: bool = False
    ctx_size: int = 4096
    left_margin: int = 2048
    # up to how much of the left_margin to be allocated as preamble
    preamble_size: int = 1000
    right_margin: int = 2048 - 512
    train_max_labels: int = 32
    dec_max_labels: int = 16
    use_small_model: bool = False
    grad_accum_labels = 32  # DAgger
    modifications: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__annotations__}

    def as_name(self) -> str:
        return self.get_model_name()

    def __repr__(self):
        return repr_modified_args(self, flatten=True)

    def get_model_name(self) -> str:
        return "model-v7--" + repr_modified_args(self, flatten=True)

    def train_ctx_args(self) -> CtxArgs:
        return CtxArgs(
            ctx_size=self.ctx_size,
            preamble_size=self.preamble_size,
            left_margin=self.left_margin,
            right_margin=self.right_margin,
            max_labels=self.train_max_labels,
            inline_prev_gold=self.inline_prev_gold,
        )

    def get_preprocess_args(self):
        return self.pre_args

    def dec_ctx_args(self) -> CtxArgs:
        r = self.train_ctx_args()
        r.max_labels = self.dec_max_labels
        return r


def train_spot_model(
    tk_dataset: dict[str, TokenizedSrcSet],
    model_name: str,
    train_args: ModelTrainingArgs,
    gpus: list[int],
    quicktest=False,
    use_early_stop=False,
    use_small_model=False,
) -> ModelWrapper:
    os.chdir(proj_root())
    train_ctx_args = train_args.train_ctx_args
    dec_args = train_args.dec_args

    running_dir = get_model_dir(False) / model_name
    if running_dir.exists():
        shutil.rmtree(running_dir)
    running_dir.mkdir(parents=True, exist_ok=True)

    print("Disk space left:")
    subprocess.run(["df", "-h", str(running_dir)])

    model_path = ModelWrapper.get_codet5_path(use_small_model)
    lit_model = TrainModelWrapper(model_path, model_saving_path=running_dir / "ckpts")
    tokenizer: TokenizerType = lit_model.tokenizer

    common_type_names = tk_dataset["train"].common_type_names()
    wrapper = ModelWrapper(
        lit_model.model, tokenizer, dec_args, common_type_names=common_type_names
    )

    chunks: dict[str, ChunkedDataset] = {}
    with run_long_task("Preparing chunked datasets", notify=False):
        for n in ["valid", "train"]:
            src = tk_dataset[n]
            chunks[n] = src.to_chunks(train_ctx_args)

    wandb_logger = WandbLogger()  # assuming a run has already been initialized

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

    ckpt_interval = max(1, len(train_dataloader) // 10)
    val_interval = 1 if quicktest else max(500, ckpt_interval)

    checkpoint_cb = ModelCheckpoint(
        dirpath=running_dir,
        save_top_k=3,
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
        devices=gpus,
        precision=16,
        max_epochs=train_args.max_epochs,
        logger=wandb_logger,
        val_check_interval=val_interval,
        callbacks=(
            [checkpoint_cb, EarlyStopping("valid/loss", mode="min", verbose=quicktest)]
            if use_early_stop
            else []
        ),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        # track_grad_norm=2,
    )

    warnings.filterwarnings("ignore", "The dataloader.*does not have many workers.*")

    with run_long_task(f"Training {model_name}", notify=False):
        trainer.fit(
            model=lit_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

    save_dir = get_model_dir(True) / model_name

    final_eval = trainer.validate(model=lit_model, dataloaders=valid_dataloader)[0]

    try:
        if (
            use_early_stop
            and (best_loss := checkpoint_cb.best_model_score) is not None
            and best_loss < final_eval["valid/loss"]
        ):
            print(
                f"Loading best model with score {best_loss} from: {checkpoint_cb.best_model_path}"
            )
            wrapper.model = TrainModelWrapper.load_from_checkpoint(
                checkpoint_cb.best_model_path
            ).model
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        wrapper.save(save_dir)
        shutil.rmtree(running_dir)
    except Exception as e:
        logging.error(
            "Error encountered after training, returning partial results... Error:\n", e
        )

    return wrapper


class TrainModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the SPOT model."

    def __init__(
        self, model_checkpoint: str | Path, *, model_saving_path: Path
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: ModelType = load_model_spot(model_checkpoint)
        self.tokenizer: TokenizerType = TokenizerType.from_pretrained(model_checkpoint)
        self.model_saving_path = model_saving_path
        self.model_saving_interval: Optional[int] = None
        self.avg_loss = MovingAvg(alpha=0.01)
        self.labels_trained = 0

    def on_fit_start(self):
        # maps chunk id to the initial predictions made for that chunk immediately
        # before the model was trained on it
        if self.model_saving_interval is not None:
            self.batch_ids: list[list[int]] = []
            self.saving_counter = 0
            self.model.save_pretrained(self.model_saving_path / f"n_batches=0")

    def configure_optimizers(self):
        return _configure_optimizers(self.model)

    def training_step(self, batch, batch_idx):
        if self.model_saving_interval is not None and self.current_epoch == 0:
            self.batch_ids.append(batch["chunk_id"].tolist())
            self.saving_counter += 1
            if self.saving_counter >= self.model_saving_interval:
                self.saving_counter = 0
                # model can be used for `n_batches` and onward.
                self.model.save_pretrained(
                    self.model_saving_path / f"n_batches={len(self.batch_ids)}"
                )

        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        assert isinstance(outputs, Seq2SeqLMOutput)
        loss = not_none(outputs.loss)
        n_labels = batch["n_labels"].sum().item()
        self.labels_trained += n_labels
        self.avg_loss.update(loss.item())
        self.log("train/loss", self.avg_loss.value)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])  # type: ignore
        self.log("train/labels", float(self.labels_trained))
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("valid/loss", loss.item())
        self.log("train/labels", float(self.labels_trained))


def concat_batches(batches: list[dict], keys: list[str]) -> dict:
    return {k: torch.concat([b[k] for b in batches]) for k in keys}


def _configure_optimizers(model: nn.Module, base_lr: float = 2e-5):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [
                p
                for pn, p in model.named_parameters()
                if not any(n in pn for n in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for pn, p in model.named_parameters()
                if any(n in pn for n in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_params, lr=base_lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.2)
    return [optimizer], [lr_scheduler]
