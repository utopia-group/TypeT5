import warnings
from datasets import Dataset
from spot.data import ChunkedDataset, CtxArgs, SrcDataset
from spot.type_check import normalize_type
from .model import (
    dynamic_dataloader,
    DataLoader,
)
from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5EncoderModel,
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.configuration_t5 import T5Config
import torch.nn as nn
import torch
from .utils import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score
from .train import _configure_optimizers
import copy


@dataclass
class CriticOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    n_preds: list[int]


class CriticModel(T5PreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"critic_classifier\."]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.", r"lm_head\."]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.t5enc = T5EncoderModel(config)
        self.base_model_prefix = "t5enc"

        self.config = config
        self.critic_dropout = nn.Dropout(config.dropout_rate)
        self.critic_classifier = nn.Linear(config.d_model, 1)
        self.tokenizer = tokenizer = load_tokenizer_spot()
        self.extra_id_min = tokenizer.additional_special_tokens_ids[0]
        self.extra_id_max = tokenizer.additional_special_tokens_ids[-1]

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True
        outputs = self.t5enc.forward(input_ids, **kwargs)
        assert isinstance(outputs, BaseModelOutputWithPastAndCrossAttentions)
        hidden_states = not_none(outputs.hidden_states)[-1]
        assert len(hidden_states.shape) == 3  # of shape (batch_size, seq_len, d_model)

        isextra = (self.extra_id_min <= input_ids).bitwise_and(
            input_ids <= self.extra_id_max
        )
        n_preds = isextra.count_nonzero(dim=1).tolist()
        hidden_states = hidden_states[isextra, :]
        assert len(hidden_states.shape) == 2  # of shape (n_labels, d_model)
        hidden_states = self.critic_dropout(hidden_states)
        logits = self.critic_classifier(hidden_states).reshape(-1)
        loss = None
        if labels is not None:
            loss = torch.binary_cross_entropy_with_logits(
                logits,
                labels.to(dtype=logits.dtype),
            ).mean()

        return CriticOutput(logits, loss, n_preds)

    def eval_on_dataset(
        self,
        src_data: SrcDataset,
        ctx_args: CtxArgs,
        sampling_max_tokens: int,
        tqdm_args: dict = {},
    ) -> tuple[ChunkedDataset, list[list[float]], dict]:
        """Convinient method to preprocess the src according to the model's ctx_args and evaluate the (R0) accuracy."""
        chunks = src_data.to_chunks(self.tokenizer, ctx_args, tqdm_args=tqdm_args)
        collator = CriticCollator(self.tokenizer)
        loader = dynamic_dataloader(
            chunks.data,
            max_tokens=sampling_max_tokens,
            collate_fn=collator,
            shuffle=True,
        )
        device = self.device
        preds = dict[int, list[float]]()
        tqdm_bar = tqdm(total=len(chunks.data), desc="predict", **tqdm_args)
        self.eval()
        with torch.no_grad():
            for batch in loader:
                batch_size = batch["input_ids"].shape[0]
                out = self.forward(input_ids=batch["input_ids"].to(device))
                pred_vec: list[float] = (out.logits.sigmoid()).tolist()
                pred_counter = 0
                for i, c_id in enumerate(batch["chunk_id"]):
                    c_id = int(c_id)
                    pred_counter_next = pred_counter + out.n_preds[i]
                    preds[c_id] = pred_vec[pred_counter:pred_counter_next]
                    pred_counter = pred_counter_next

                tqdm_bar.update(batch_size)
        tqdm_bar.close()
        preds = [preds[int(c_id)] for c_id in chunks.data["chunk_id"]]
        target = list(seq_flatten(to_critic_dataset(chunks)["labels"]))
        pred_bools = [p >= 0.5 for p in seq_flatten(preds)]

        metrics = {
            "Acc": accuracy_score(pred_bools, target),
            "F1": f1_score(pred_bools, target),
        }

        return chunks, preds, metrics


def stack_and_pad(xs: list[list[int]], pad_id: int) -> torch.LongTensor:
    max_len = max(len(x) for x in xs)
    xs = [x + [pad_id] * (max_len - len(x)) for x in xs]
    return torch.LongTensor(xs)


class CriticTrainArgs(NamedTuple):
    ctx_args: CtxArgs
    train_max_tokens: int
    eval_max_tokens: int
    max_epochs: int


def train_critic_model(
    critic_datasets: dict[str, SrcDataset],
    train_args: CriticTrainArgs,
    model_name: str,
    gpus: list[int],
    quicktest=False,
    use_early_stop=True,
    use_small_model=False,
) -> tuple[CriticModel, dict]:
    os.chdir(proj_root())

    datadir = Path(os.getenv("datadir", "data"))

    running_dir = datadir / "checkpoints/lit-running" / model_name
    if running_dir.exists():
        shutil.rmtree(running_dir)
    running_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
    )
    lit_model = TrainCriticModelWrapper(model_path)
    model = lit_model.model
    tokenizer: TokenizerSPOT = lit_model.model.tokenizer

    datasets: dict[str, Dataset] = {}
    with run_long_task("Preparing critic datasets", notify=False):
        for n in ["valid", "test", "train"]:
            cdata = critic_datasets[n].to_chunks(tokenizer, train_args.ctx_args)
            datasets[n] = to_critic_dataset(cdata)

    wandb_logger = WandbLogger()  # assuming a run has already been initialized

    collate_fn = CriticCollator(tokenizer)
    dataloaders = dict[str, DataLoader]()
    for n, data in datasets.items():
        dataloaders[n] = dynamic_dataloader(
            data,
            max_tokens=(
                train_args.train_max_tokens
                if n == "train"
                else train_args.eval_max_tokens
            ),
            collate_fn=collate_fn,
            shuffle=True,
        )

    ckpt_interval = max(1, len(dataloaders["train"]) // 8)
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
        accumulate_grad_batches=None,
    )

    warnings.filterwarnings("ignore", "The dataloader.*does not have many workers.*")

    with run_long_task(f"Training {model_name}"):
        trainer.fit(
            model=lit_model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["valid"],
        )

    extra = dict[str, Any]()
    save_dir = datadir / "checkpoints/lit-saved" / model_name

    final_eval = trainer.validate(model=lit_model, dataloaders=dataloaders["valid"])[0]

    try:
        if (
            use_early_stop
            and (best_loss := checkpoint_cb.best_model_score) is not None
            and best_loss < final_eval["valid/loss"]
        ):
            print(
                f"Loading best model with score {best_loss} from: {checkpoint_cb.best_model_path}"
            )
            model = TrainCriticModelWrapper.load_from_checkpoint(
                checkpoint_cb.best_model_path
            ).model
            lit_model.model = model
            trainer.test(model=lit_model, dataloaders=dataloaders["test"])
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_dir)

        shutil.rmtree(running_dir)
    except Exception as e:
        logging.error(
            "Error encountered after training, returning partial results... Error:\n", e
        )

    return model, extra


def to_critic_dataset(cdata: ChunkedDataset) -> Dataset:
    new_data = dict()
    new_data["input_ids"] = cdata.data["input_ids"]
    new_data["labels"] = labels = list[list[bool]]()
    for info in cdata.chunks_info:
        labels.append(
            [
                normalize_type(p) == normalize_type(l)
                for p, l in zip(not_none(info.prev_types), info.types)
            ]
        )

    new_data["chunk_id"] = cdata.data["chunk_id"]
    return Dataset.from_dict(new_data)


@dataclass
class CriticCollator:
    tokenizer: TokenizerSPOT

    def __call__(self, batch: Sequence[dict]) -> dict:
        pad_id = not_none(self.tokenizer.pad_token_id)
        return {
            "input_ids": stack_and_pad([b["input_ids"] for b in batch], pad_id),
            # "output_ids": stack_and_pad([b["output_ids"] for b in batch], pad_id),
            "labels": torch.BoolTensor([l for chunk in batch for l in chunk["labels"]]),
            "chunk_id": [chunk["chunk_id"] for chunk in batch],
        }


class TrainCriticModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the Critic model."

    def __init__(self, model_checkpoint: str | Path) -> None:
        super().__init__()
        self.save_hyperparameters()
        model = CriticModel.from_pretrained(model_checkpoint)
        assert isinstance(model, CriticModel)
        self.model: CriticModel = model

    def configure_optimizers(self):
        return _configure_optimizers(self.model)

    def training_step(self, batch, batch_idx):
        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        batch_size = batch["input_ids"].shape[0]
        loss = not_none(outputs.loss)
        self.log("train/loss", loss.item(), batch_size=batch_size)
        self.log("train/lr", self.lr_schedulers().get_last_lr()[0], batch_size=batch_size)  # type: ignore
        return loss

    def _eval_step(self, batch, name: str):
        outputs = self.model.forward(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        batch_size = batch["input_ids"].shape[0]
        loss = not_none(outputs.loss)
        self.log(f"{name}/loss", loss.item(), batch_size=batch_size)

        preds = (outputs.logits >= 0).cpu()
        target = batch["labels"].cpu()

        acc = accuracy_score(preds, target)
        self.log(f"{name}/accuracy", as_any(acc), batch_size=batch_size)

        f1 = f1_score(preds, target)
        self.log(f"{name}/f1", as_any(f1), batch_size=batch_size)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")


def evaluate_critic(
    critic: CriticModel,
    args: CtxArgs,
    sampling_max_tokens: int,
    r1_srcs: SrcDataset,
    eval_cache: Optional[PickleCache] = None,
    tqdm_args={},
) -> tuple[ChunkedDataset, list[list[float]], dict]:
    def cached(name, f):
        if eval_cache is None:
            return f()
        return eval_cache.cached(name, f)

    return cached(
        f"critic_eval-{args}.pkl",
        lambda: critic.eval_on_dataset(
            r1_srcs, args, sampling_max_tokens, tqdm_args=tqdm_args
        ),
    )
