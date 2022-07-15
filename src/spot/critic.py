import warnings
from datasets import Dataset
from spot.data import ChunkedDataset, CtxArgs, SrcDataset
from spot.type_check import normalize_type
from .model import (
    dynamic_dataloader,
    DataLoader,
)
from transformers.models.t5.modeling_t5 import (
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from .train import TrainingConfig, _configure_optimizers


@dataclass
class CriticOutput:
    logits: torch.Tensor
    n_preds: list[int]


class CriticModel(nn.Module):
    def __init__(self, t5enc: T5EncoderModel):
        super().__init__()

        self.t5enc = t5enc
        config = t5enc.config

        # self.critic_classifier = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(config.dropout_rate),
        #     nn.Linear(config.d_model, 1),
        # )
        # self.critic_encoder = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.LeakyReLU(),
        # )
        self.critic_classifier = nn.Linear(config.d_model, 1)
        self.tokenizer = load_tokenizer_spot()
        self.labels_trained = 0

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        prediction_spans: list[list[tuple[int, int]]],
        **kwargs,
    ):
        kwargs["return_dict"] = True
        kwargs["output_hidden_states"] = True
        outputs = self.t5enc.forward(input_ids, attention_mask=attention_mask, **kwargs)
        assert isinstance(outputs, BaseModelOutputWithPastAndCrossAttentions)
        hidden_states = not_none(outputs.last_hidden_state)
        assert len(hidden_states.shape) == 3  # of shape (batch_size, seq_len, d_model)

        classifier_inputs = []
        for row, spans in enumerate(prediction_spans):
            for s in spans:
                hs = hidden_states[row, s[0] - 1 : s[0], :]
                classifier_inputs.append(hs)
                # hs = self.critic_encoder(
                #     hidden_states[row, s[0] : s[1], :]
                # )  # (span_len, d_model)
                # classifier_inputs.append(hs.mean(dim=0, keepdim=True))
        c_inputs = torch.cat(classifier_inputs, dim=0)  # (n_labels, d_model)
        assert len(c_inputs.shape) == 2  # of shape (n_labels, d_model)

        # rescale the hidden_states before feeding to the head, as done in the original T5 model
        # hidden_states = hidden_states * (self.config.d_model**-0.5)
        logits = self.critic_classifier.forward(c_inputs).reshape(-1)
        n_preds = [len(x) for x in prediction_spans]
        return CriticOutput(logits, n_preds)

    @property
    def device(self):
        return self.t5enc.device

    def classify_data(
        self,
        dataloader,
        n_examples: int,
        tqdm_args: dict = {},
    ) -> dict[int, list[float]]:
        """Run the critic model on the given dataloader and returns classification
        probability for each prediction span."""
        device = self.device
        chunk2preds = dict[int, list[float]]()
        self.eval()

        with torch.no_grad(), tqdm(
            total=n_examples, desc="classify_data", **tqdm_args
        ) as pbar:
            for batch in dataloader:
                batch_size = batch["input_ids"].shape[0]
                with torch.autocast("cuda"):
                    out = self.forward(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        prediction_spans=batch["prediction_spans"],
                    )
                pred_vec: list[float] = out.logits.sigmoid().tolist()
                pred_counter = 0
                assert_eq(len(out.n_preds), len(batch["input_ids"]))
                for n_preds, c_id in zip(out.n_preds, batch["chunk_id"]):
                    c_id = int(c_id)
                    pred_counter_next = pred_counter + n_preds
                    chunk2preds[c_id] = pred_vec[pred_counter:pred_counter_next]
                    pred_counter = pred_counter_next

                pbar.update(batch_size)

        return chunk2preds

    @staticmethod
    def compute_metrics(pred_scores, label_bools):
        pred_bools = [x >= 0.5 for x in pred_scores]
        avg_error = float(
            np.mean([abs(x - float(y)) for x, y in zip(pred_scores, label_bools)])
        )
        return {
            "accuracy": accuracy_score(label_bools, pred_bools),
            "avg_error": avg_error,
            "F1": f1_score(label_bools, pred_bools),
            "precision": precision_score(label_bools, pred_bools),
            "recall": recall_score(label_bools, pred_bools),
            "pos_rate": sum(pred_bools) / len(pred_bools),
        }

    def eval_on_src_dataset(
        self,
        src_data: SrcDataset,
        ctx_args: CtxArgs,
        sampling_max_tokens: int,
        tqdm_args: dict = {},
    ) -> tuple[ChunkedDataset, list[list[float]], dict]:
        """Convinient method to preprocess the src according to the model's ctx_args and evaluate the (R0) accuracy."""
        chunks = src_data.to_chunks(ctx_args, tqdm_args=tqdm_args)
        collator = CriticCollator()
        dataset = to_critic_dataset(chunks)
        loader = dynamic_dataloader(
            dataset,
            max_tokens=sampling_max_tokens,
            collate_fn=collator,
            shuffle=True,
        )
        n_exs = len(chunks.data)
        chunk2preds = self.classify_data(loader, n_exs, tqdm_args=tqdm_args)
        chunk2labels: dict[int, list[bool]] = dict(
            zip(dataset["chunk_id"], dataset["is_correct"])
        )

        pred_scores = [p for cid in chunk2preds for p in chunk2preds[cid]]
        label_bools = [l for cid in chunk2preds for l in chunk2labels[cid]]

        metrics = CriticModel.compute_metrics(pred_scores, label_bools)

        preds_ordered = [chunk2preds[int(c_id)] for c_id in chunks.data["chunk_id"]]

        return chunks, preds_ordered, metrics

    def save(self, dir: Path):
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(self, dir / "model.pt")

    @staticmethod
    def load(dir: Path, verify_trained: bool = True) -> "CriticModel":
        model = torch.load(dir / "model.pt")
        assert isinstance(model, CriticModel)
        if verify_trained:
            assert model.labels_trained > 0
        return model

    @staticmethod
    def from_code_t5(path: Path | str) -> "CriticModel":
        def msg_filter(record):
            return (
                not "Some weights of the model checkpoint at Salesforce/codet5-base were not used when initializing T5EncoderModel"
                in record.msg
            )

        logging.getLogger("transformers.modeling_utils").addFilter(msg_filter)

        t5_enc = as_any(T5EncoderModel.from_pretrained(path))
        assert isinstance(t5_enc, T5EncoderModel)
        return CriticModel(t5_enc)


def stack_and_pad(xs: list[list[int]], pad_id: int) -> torch.LongTensor:
    max_len = max(len(x) for x in xs)
    xs = [x + [pad_id] * (max_len - len(x)) for x in xs]
    return torch.LongTensor(xs)


def get_critic_name(no_feedback: bool, new_data: bool, config: TrainingConfig) -> str:
    feedback_tag = "no_feedback-" if no_feedback else ""
    data_tag = "new_data-" if new_data else ""
    return "critic-model--" + feedback_tag + data_tag + config.as_name()


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

    datasets: dict[str, Dataset] = {}
    for n in ["valid", "test", "train"]:
        sdata = critic_datasets[n]
        cdata = sdata.to_chunks(train_args.ctx_args)
        datasets[n] = to_critic_dataset(cdata)

    # pos_weight = compute_pos_weight(list(seq_flatten(datasets["train"]["is_correct"])))
    pos_weight = 1.0
    assert math.isfinite(pos_weight), f"pos_weight = {pos_weight}"

    model_path = (
        "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
    )
    model = CriticModel.from_code_t5(model_path)
    lit_model = TrainCriticModelWrapper(model, pos_weight, running_dir)

    wandb_logger = WandbLogger()  # assuming a run has already been initialized
    wandb_logger.log_hyperparams({"pos_weight": pos_weight})
    print(f"pos_weight = {pos_weight}")

    collate_fn = CriticCollator()
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
    val_interval = 1 if quicktest else ckpt_interval

    trainer = pl.Trainer(
        default_root_dir=str(running_dir),
        accelerator="gpu" if gpus else "cpu",
        gpus=gpus,
        precision=16,
        max_epochs=train_args.max_epochs,
        logger=wandb_logger,
        val_check_interval=val_interval,
        callbacks=(
            [EarlyStopping("valid/loss", mode="min", verbose=quicktest)]
            if use_early_stop
            else []
        ),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=None,
        # track_grad_norm=2,
    )

    warnings.filterwarnings("ignore", "The dataloader.*does not have many workers.*")

    trainer.validate(lit_model, dataloaders["valid"])
    trainer.fit(
        lit_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["valid"],
    )
    model = lit_model.model

    extra = dict[str, Any]()
    save_dir = datadir / "checkpoints/lit-saved" / model_name

    try:
        trainer.test(lit_model, dataloaders["test"])
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)

        model.save(save_dir)

        shutil.rmtree(running_dir)
    except Exception as e:
        logging.error(
            "Error encountered after training, returning partial results... Error:\n", e
        )

    return model, extra


def compute_pos_weight(
    labels_train: list[bool], labels_valid: list[bool] | None = None
) -> float:
    if labels_valid is None:
        return labels_train.count(False) / labels_train.count(True)

    return (labels_valid.count(True) * labels_train.count(False)) / (
        labels_valid.count(False) * labels_train.count(True)
    )


def to_critic_dataset(cdata: ChunkedDataset) -> Dataset:
    new_data = dict()
    new_data["input_ids"] = cdata.data["input_ids"]
    new_data["is_correct"] = labels = list[list[bool]]()
    new_data["prediction_spans"] = [
        [(s.start, s.stop) for s in not_none(info.inlined_spans)]
        for info in cdata.chunks_info
    ]
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
    def __call__(self, batch: Sequence[dict]) -> dict:
        pad_id = not_none(DefaultTokenizer.pad_token_id)
        input_ids = stack_and_pad([chunk["input_ids"] for chunk in batch], pad_id)
        attention_mask = (input_ids != pad_id).float()
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # "output_ids": stack_and_pad([b["output_ids"] for b in batch], pad_id),
            "chunk_id": [chunk["chunk_id"] for chunk in batch],
            "prediction_spans": [chunk["prediction_spans"] for chunk in batch],
        }
        if "is_correct" in batch[0]:
            result["is_correct"] = torch.BoolTensor(
                [l for chunk in batch for l in chunk["is_correct"]]
            )
        return result


class TrainCriticModelWrapper(pl.LightningModule):
    "A pytorch lightening module that handles training and evaluation of the Critic model."

    def __init__(
        self, model: CriticModel, pos_weight: float, running_dir: Path
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model: CriticModel = model
        self.pos_weight = pos_weight
        self.running_dir = running_dir
        self.best_loss = None

    def configure_optimizers(self):
        return _configure_optimizers(self.model)

    def training_step(self, batch, batch_idx):
        labels = batch["is_correct"]
        logits = self.model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prediction_spans=batch["prediction_spans"],
        ).logits
        n_labels = batch["is_correct"].shape[0]
        self.model.labels_trained += len(labels)
        loss_f = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.pos_weight, dtype=logits.dtype, device=logits.device
            )
        )
        loss = loss_f.forward(logits, labels.to(dtype=logits.dtype))
        self.log("train/loss", loss.item(), batch_size=n_labels)
        self.log(
            "train/lr",
            as_any(self.lr_schedulers()).get_last_lr()[0],
            batch_size=n_labels,
        )
        pos_rate = (logits >= 0).float().mean().item()
        self.log("train/pos_rate", pos_rate, batch_size=n_labels)
        return loss

    def _eval_step(self, batch, name: str):
        labels = batch["is_correct"]
        logits = self.model.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            prediction_spans=batch["prediction_spans"],
        ).logits
        loss_f = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                self.pos_weight, dtype=logits.dtype, device=logits.device
            )
        )
        loss = loss_f.forward(logits, labels.to(dtype=logits.dtype))
        n_labels = batch["is_correct"].shape[0]
        self.log(f"{name}/loss", loss.item(), batch_size=n_labels)

        scores = torch.sigmoid(logits).cpu()
        targets = batch["is_correct"].cpu()

        return {
            "loss": loss.item(),
            "n_labels": n_labels,
            "scores": scores,
            "targets": targets,
        }

    def _eval_log_metrics(self, outputs, name: str):
        scores = list(seq_flatten(o["scores"] for o in outputs))
        targets = list(seq_flatten(o["targets"] for o in outputs))
        metrics = CriticModel.compute_metrics(scores, targets)
        for metric, value in metrics.items():
            self.log(f"{name}/{metric}", value)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def validation_epoch_end(self, outputs: list[dict]) -> None:
        total_loss = sum(o["loss"] for o in outputs)
        n_labels = sum(o["n_labels"] for o in outputs)
        avg_loss = total_loss / n_labels
        if self.best_loss is None or avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.model.save(self.running_dir / "best_model")

        self._eval_log_metrics(outputs, "valid")

    def test_epoch_end(self, outputs: list[dict]) -> None:
        self._eval_log_metrics(outputs, "test")

    # def on_fit_end(self):
    #     self.model = as_any(
    #         CriticModel.from_pretrained(self.running_dir / "best_model")
    #     )
    #     super().on_fit_end()
