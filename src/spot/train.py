import os
import warnings
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from transformers.modeling_outputs import Seq2SeqLMOutput

from spot.tokenized_src import PreprocessArgs

from .type_check import TypeCheckArgs

from .data import (
    ChunkedDataset,
    CountedAcc,
    R1_srcs_from_preds,
    SrcChunkInfo,
    SrcDataset,
    preds_to_accuracies,
)
from .model import (
    CtxArgs,
    DecodingArgs,
    ModelSPOT,
    ModelWrapper,
    TokenizerSPOT,
    dynamic_dataloader,
    DatasetPredResult,
)
from .type_env import PythonType
from .utils import *
from .visualization import (
    dict_widget,
    string_widget,
    visualize_sequence_tabs,
)


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
    drop_comments: bool = True
    imports_in_preamble: bool = True
    stub_in_preamble: bool = False
    data_reduction: int = 1
    check_in_isolation: bool = False
    all_labels: bool = True
    ctx_size: int = 4096
    left_margin: int = 2048
    # up to how much of the left_margin to be allocated as preamble
    preamble_size: int = 512
    right_margin: int = 1023
    train_max_labels: int = 32
    dec_max_labels: int = 16
    use_small_model: bool = False
    grad_accum_labels = 32
    modifications: str = ""

    def modified_params(self) -> dict[str, Any]:
        default = TrainingConfig()
        changed = dict[str, Any]()
        # collect all attributes that are different from default
        for attr in self.__annotations__:
            if getattr(self, attr) != getattr(default, attr):
                changed[attr] = getattr(self, attr)
        return changed

    def as_dict(self) -> dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__annotations__}

    def as_name(self) -> str:
        if len(self.modified_params()) > 0:
            return "-".join(
                f"{str(k)}={str(v)}" for k, v in self.modified_params().items()
            )
        else:
            return "default"

    def get_model_name(self) -> str:
        return "model-v3--" + self.as_name()

    def train_ctx_args(self) -> CtxArgs:
        return CtxArgs(
            ctx_size=self.ctx_size,
            preamble_size=self.preamble_size,
            left_margin=self.left_margin,
            right_margin=self.right_margin,
            max_labels=self.train_max_labels,
        )

    def get_preprocess_args(self):
        return PreprocessArgs(
            drop_comments=self.drop_comments,
            imports_in_preamble=self.imports_in_preamble,
            stub_in_preamble=self.stub_in_preamble,
        )

    def dec_ctx_args(self) -> CtxArgs:
        r = self.train_ctx_args()
        r.max_labels = self.dec_max_labels
        return r


def train_spot_model(
    src_datasets: dict[str, SrcDataset],
    model_name: str,
    train_args: ModelTrainingArgs,
    record_batches: bool,
    gpus: list[int],
    quicktest=False,
    use_early_stop=True,
    use_small_model=False,
) -> tuple[ModelWrapper, dict]:
    os.chdir(proj_root())
    train_ctx_args = train_args.train_ctx_args
    dec_args = train_args.dec_args

    datadir = Path(os.getenv("datadir", "data"))

    running_dir = datadir / "checkpoints/lit-running" / model_name
    if running_dir.exists():
        shutil.rmtree(running_dir)
    running_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
    )
    lit_model = TrainModelWrapper(model_path, model_saving_path=running_dir / "models")
    tokenizer: TokenizerSPOT = lit_model.tokenizer
    wrapper = ModelWrapper(lit_model.model, tokenizer, dec_args)

    chunks: dict[str, ChunkedDataset] = {}
    with run_long_task("Preparing chunked datasets", notify=False):
        for n in ["valid", "train"]:
            src = src_datasets[n]
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

    if record_batches:
        lit_model.model_saving_interval = ckpt_interval

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
        gpus=gpus,
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

    extra = dict[str, Any]()
    save_dir = datadir / "checkpoints/lit-saved" / model_name

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

        wrapper.save_pretrained(save_dir)
        if record_batches:
            device = torch.device(f"cuda:{gpus[0]}" if gpus else "cpu")
            wrapper = wrapper.to(device)
            extra["batch_ids"] = lit_model.batch_ids
            with run_long_task("Generating R1 datasets", notify=False):
                R1_src_datasets = R1_srcs_from_extra(
                    wrapper,
                    src_datasets,
                    extra,
                    tc_args=train_args.tc_args,
                    ckpt_dir=running_dir / "models",
                    ckpt_interval=ckpt_interval,
                )

            extra["R1-src_datasets"] = R1_src_datasets
            pickle_dump(save_dir / "extra.pkl", extra)

            shutil.rmtree(running_dir)
    except Exception as e:
        logging.error(
            "Error encountered after training, returning partial results... Error:\n", e
        )

    return wrapper, extra


def R1_srcs_from_extra(
    wrapper: ModelWrapper,
    src_datasets: dict[str, SrcDataset],
    extra: dict[str, Any],
    tc_args: TypeCheckArgs,
    ckpt_dir: Optional[Path] = None,
    ckpt_interval: Optional[int] = None,
) -> dict[str, SrcDataset]:
    """
    Generate the R1 dataset using the extra info recorded during training.
    The training set is generated using the predictions recorded during training
    (or loaded from the different model checkpoints).
    """

    tokenizer = wrapper.tokenizer
    batch_ids = extra["batch_ids"]
    print(f"Generating R1 dataset: train")

    chunk_datasets = dict[str, ChunkedDataset]()
    for n in ["test", "valid", "train"]:
        src = src_datasets[n]
        chunk_datasets[n] = src.to_chunks(wrapper.args.ctx_args)

    R1_src_datasets = dict[str, SrcDataset]()
    if ckpt_dir is None or ckpt_interval is None:
        chunks_info = extra["chunks_info"]
        model_preds = extra["model_preds"]
        R1_src_datasets["train"] = R1_srcs_from_preds(
            src_datasets["train"],
            chunks_info,
            chunk_datasets["train"].files,
            model_preds,
            tc_args=tc_args,
            max_workers=wrapper.args.max_workers,
        )
    else:
        R1_src_datasets["train"], chunks_info, model_preds = R1_srcs_from_ckpts(
            tokenizer,
            wrapper.args,
            src_datasets["train"],
            chunk_datasets["train"],
            batch_ids,
            tc_args=tc_args,
            ckpt_dir=ckpt_dir,
            ckpt_interval=ckpt_interval,
            max_workers=wrapper.args.max_workers,
            device=wrapper.model.device,
        )
        extra["chunks_info"] = chunks_info
        extra["model_preds"] = model_preds
    for n in ["valid", "test"]:
        print(f"Generating R1 dataset: {n}")
        preds = wrapper.predict(chunk_datasets[n].data, {})
        R1_src_datasets[n] = R1_srcs_from_preds(
            src_datasets[n],
            chunk_datasets[n].chunks_info,
            chunk_datasets[n].files,
            preds,
            tc_args=tc_args,
            max_workers=wrapper.args.max_workers,
        )
    return R1_src_datasets


def R1_srcs_from_model(
    wrapper: ModelWrapper,
    src_datasets: dict[str, SrcDataset],
    tc_args: TypeCheckArgs,
) -> dict[str, SrcDataset]:
    R1_src_datasets = dict[str, SrcDataset]()
    tokenizer = wrapper.tokenizer
    chunk_datasets = {
        n: src_datasets[n].to_chunks(wrapper.args.ctx_args)
        for n in ["train", "valid", "test"]
    }
    for n, cdata in chunk_datasets.items():
        print(f"Generating R1 dataset: {n}")
        preds = wrapper.predict(cdata.data, {})

        R1_src_datasets[n] = R1_srcs_from_preds(
            src_datasets[n],
            chunk_datasets[n].chunks_info,
            chunk_datasets[n].files,
            preds,
            tc_args=tc_args,
            max_workers=wrapper.args.max_workers,
        )
        r0_accs = preds_to_accuracies(preds, cdata)
        R1_src_datasets[n].add_stats({"r0_full_acc": r0_accs["full_acc"]})
    return R1_src_datasets


def R1_srcs_from_ckpts(
    tokenizer: TokenizerSPOT,
    dec_args: DecodingArgs,
    r0_src: SrcDataset,
    cdata: ChunkedDataset,
    chunk_ids: list[list[int]],
    tc_args: TypeCheckArgs,
    ckpt_dir: Path,
    ckpt_interval: int,
    max_workers: int,
    device,
    tqdm_args={},
):
    # TODO: find out why some chunks are missing
    # assert_eq(sum(len(x) for x in chunk_ids), len(cdata.chunks_info))
    if (n_got := sum(len(x) for x in chunk_ids)) != len(cdata.chunks_info):
        logging.warning(
            f"Some chunks are missing. Got {n_got} chunks, but expected {len(cdata.chunks_info)}"
        )
    chunks_info = list[SrcChunkInfo]()
    model_preds = list[list[PythonType]]()
    for i in tqdm(
        range(0, len(chunk_ids), ckpt_interval),
        desc="R1_srcs_from_ckpts",
        **tqdm_args,
    ):
        ids = list(seq_flatten(chunk_ids[i : i + ckpt_interval]))
        model = load_model_spot(ckpt_dir / f"n_batches={i}")
        wrapper = ModelWrapper(model, tokenizer, dec_args)
        wrapper = wrapper.to(device)
        try:
            data_sub = cdata[ids]
        except IndexError as e:
            raise IndexError(
                f"ids: {ids},\nchunk_ids: {cdata.data['chunk_id']}\ncdata: {cdata}"
            ) from e
        chunks_info.extend(data_sub.chunks_info)
        preds = wrapper.predict(data_sub.data, tqdm_args=tqdm_args)
        model_preds.extend(preds)
    srcs = R1_srcs_from_preds(
        r0_src,
        chunks_info,
        cdata.files,
        model_preds,
        tc_args=tc_args,
        max_workers=max_workers,
        tqdm_args=tqdm_args,
    )
    return srcs, chunks_info, model_preds


class TrainModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the SPOT model."

    def __init__(
        self, model_checkpoint: str | Path, *, model_saving_path: Path
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: ModelSPOT = load_model_spot(model_checkpoint)
        self.tokenizer: TokenizerSPOT = TokenizerSPOT.from_pretrained(model_checkpoint)
        self.model_saving_path = model_saving_path
        self.model_saving_interval: Optional[int] = None

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
        self.log("train/loss", loss.item())
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0])  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log("valid/loss", loss.item())


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


def evaluate_model(
    r0_wrapper: ModelWrapper,
    r1_wrapper: Optional[ModelWrapper],
    r0_srcs: SrcDataset,
    tc_args: TypeCheckArgs,
    eval_cache: Optional[PickleCache] = None,
    tqdm_args={},
) -> list[tuple[DecodingArgs, DatasetPredResult]]:
    def cached(name, f):
        if eval_cache is None:
            return f()
        return eval_cache.cached(name, f)

    results = list[tuple[DecodingArgs, DatasetPredResult]]()
    r0_result = cached(
        f"r0_eval-{r0_wrapper.args}.pkl",
        lambda: r0_wrapper.eval_on_dataset(r0_srcs, tqdm_args=tqdm_args),
    )
    results.append((copy.deepcopy(r0_wrapper.args), r0_result))
    if r1_wrapper is None:
        return results

    r1_srcs = cached(
        f"r1_srcs-{r0_wrapper.args}-{tc_args}.pkl",
        lambda: R1_srcs_from_preds(
            r0_srcs,
            r0_result.chunks.chunks_info,
            r0_result.chunks.files,
            r0_result.predictions,
            tc_args=tc_args,
            max_workers=r0_wrapper.args.max_workers,
        ),
    )
    r1_srcs = r1_srcs.inline_predictions(True)

    r1_result = cached(
        f"r1_eval-{r1_wrapper.args}-{tc_args}.pkl",
        lambda: r1_wrapper.eval_on_dataset(r1_srcs, tqdm_args=tqdm_args),
    )
    results.append((copy.deepcopy(r1_wrapper.args), r1_result))

    return results
