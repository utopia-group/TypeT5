import random
from collections import Counter
from copy import copy, deepcopy
from typing import NamedTuple, overload

import numpy as np
from datasets.arrow_dataset import Dataset
from huggingface_hub import snapshot_download
from mypy_extensions import mypyc_attr
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from transformers.data.data_collator import DataCollatorForSeq2Seq

from .data import (
    ChunkedDataset,
    CtxArgs,
    TokenizedSrcSet,
    output_ids_as_types,
    preds_to_accuracies,
)
from .type_env import AccuracyMetric, PythonType
from .utils import *


@dataclass
class DecodingArgs:
    ctx_args: CtxArgs
    sampling_max_tokens: int
    max_workers: int = DefaultWorkers
    # the maximal prediction length = tokens_per_type * num_types + slack_tokens
    tokens_per_type: int = 16
    slack_tokens: int = 10
    do_sample: bool = False
    top_p: float = 0.9
    num_beams: Optional[int] = None
    num_beam_groups: Optional[int] = None
    length_penalty: float = 1.0
    diversity_penalty: float | None = None

    def scale_ctx_size(self, factor: float) -> "DecodingArgs":
        result = deepcopy(self)
        assert result.ctx_args is not None
        """Scale the context size of the model by the given factor, while keeping the window size the same.
        Also scale down the sampling batch size accordingly."""
        ctx_size = round(self.ctx_args.ctx_size * factor)
        right_margin = round(self.ctx_args.right_margin * factor)
        left_margin = ctx_size - right_margin - self.ctx_args.window_size
        result.ctx_args.ctx_size = ctx_size
        result.ctx_args.left_margin = left_margin
        result.ctx_args.right_margin = right_margin
        result.sampling_max_tokens = round(self.sampling_max_tokens / factor**2)

        return result

    def __repr__(self) -> str:
        return repr_modified_args(self)


@dataclass
class DatasetPredResult(Generic[T1]):
    chunks: ChunkedDataset
    predictions: list[list[PythonType]]
    extra_info: list[T1] = field(default_factory=list)

    def accuracies(self, metric: AccuracyMetric) -> dict:
        return preds_to_accuracies(self.predictions, self.chunks, metric)

    def group_by_repo(self, repos_dir: Path) -> dict[Path, "DatasetPredResult[T1]"]:
        chunk2repo = list[Path]()
        for i, info in enumerate(self.chunks.chunks_info):
            file = repos_dir / info.src_file
            repo = self.chunks.file2repo[file]
            chunk2repo.append(repo)

        group2ids = groupby(range(len(chunk2repo)), lambda i: chunk2repo[i])
        result = dict()
        chunk_ids = self.chunks.data["chunk_id"]
        for repo, ids in group2ids.items():
            result[repo] = DatasetPredResult(
                self.chunks[(chunk_ids[i] for i in ids)],
                [self.predictions[i] for i in ids],
                [self.extra_info[i] for i in ids] if self.extra_info else [],
            )
        return result


@dataclass
class ModelWrapper:
    model: ModelType
    tokenizer: TokenizerType
    args: DecodingArgs
    common_type_names: set[str]
    monitor: TaskMonitor = EmptyLoggingMonitor()

    @staticmethod
    def get_codet5_path(use_small_model: bool = False) -> str:
        return (
            "Salesforce/codet5-small" if use_small_model else "Salesforce/codet5-base"
        )

    def scale_ctx_size(self, factor) -> "ModelWrapper":
        r = copy(self)
        r.args = r.args.scale_ctx_size(factor)
        return r

    def predict_on_batch(
        self,
        batch: dict,
        num_return_sequences: int | None = None,
    ) -> tuple[list[list[PythonType]], Tensor]:
        """Run the model on the given batch and return the predicted types for each row."""
        model = self.model
        args = self.args
        n_labels = batch["n_labels"]
        max_labels = max(n_labels)

        div_pen = args.diversity_penalty
        if args.num_beam_groups is not None:
            assert (
                div_pen is not None and div_pen > 0
            ), "num_beam_groups requires diversity_penalty > 0"

        output_ids = model.generate(
            inputs=batch["input_ids"].to(model.device),
            do_sample=args.do_sample,
            top_p=args.top_p,
            num_beams=args.num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=args.num_beam_groups,
            max_length=args.tokens_per_type * max_labels + args.slack_tokens,
            diversity_penalty=div_pen,
            length_penalty=args.length_penalty,
            renormalize_logits=True,
        ).cpu()  # type: ignore
        assert len(output_ids.shape) == 2

        def decode_row(row, n_labels) -> list[PythonType]:
            return output_ids_as_types(row, n_labels)

        n_rows = output_ids.shape[0]
        if num_return_sequences is not None:
            assert_eq(n_rows, num_return_sequences * len(n_labels))
        else:
            num_return_sequences = 1
        types = [
            decode_row(output_ids[i, :], n_labels[i // num_return_sequences])
            for i in range(n_rows)
        ]
        return types, output_ids

    @overload
    def predict(
        self, dataset: Dataset, tqdm_args: dict = {}, num_return_sequences: None = None
    ) -> list[list[PythonType]]:
        ...

    @overload
    def predict(
        self, dataset: Dataset, tqdm_args: dict, num_return_sequences: int
    ) -> list[list[list[PythonType]]]:
        ...

    def predict(
        self,
        dataset: Dataset,
        tqdm_args: dict = {},
        num_return_sequences: Optional[int] = None,
    ):
        """Run the  model on the given dataset and return the predicted types
        (or multiple sequences of predicted types if num_return_sequences is not none) for each row."""
        model = self.model
        collator = DataCollatorForSeq2Seq(self.tokenizer, model)
        loader = dynamic_dataloader(
            dataset,  # type: ignore
            max_tokens=self.args.sampling_max_tokens,
            collate_fn=collator,
            shuffle=True,
        )
        device = model.device
        # we use this dict to keep the order of the chunks since it may be permuted by dynamic_dataloader
        pred_types = dict[int, list]()
        with tqdm(
            total=len(dataset), desc="predict", smoothing=0.01, **tqdm_args
        ) as tqdm_bar:
            for batch in loader:
                n_chunks = batch["input_ids"].shape[0]
                batch["input_ids"] = batch["input_ids"].to(device)
                preds, _ = self.predict_on_batch(batch, num_return_sequences)
                for i, c_id in enumerate(batch["chunk_id"]):
                    c_id = int(c_id)
                    if num_return_sequences is None:
                        pred_types[c_id] = preds[i]
                    else:
                        pred_types[c_id] = preds[
                            i * num_return_sequences : (i + 1) * num_return_sequences
                        ]
                tqdm_bar.update(n_chunks)
        return [pred_types[int(c_id)] for c_id in dataset["chunk_id"]]

    def save(self, path: Path):
        """Save the model to the given path along with its tokenizer and args."""
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        pickle_dump(path / "args.pkl", self.args)
        pickle_dump(path / "common_names.pkl", self.common_type_names)

    def to(self, device) -> "ModelWrapper":
        self.model = self.model.to(device)
        return self

    @classmethod
    def load_from_hub(cls, repo_name: str):
        path = snapshot_download(repo_name)
        return cls.load(Path(path))

    @classmethod
    def load(cls, path: Path) -> "ModelWrapper":
        """Load a pretrained model from the given path."""
        model = cast(ModelType, ModelType.from_pretrained(str(path)))
        tokenizer = TokenizerType.from_pretrained(str(path))
        args = pickle_load(path / "args.pkl")
        common_type_names = ModelWrapper.load_common_type_names(path)
        return ModelWrapper(
            model=model,
            tokenizer=tokenizer,
            args=args,
            common_type_names=common_type_names,
            monitor=TaskLoggingMonitor(path.name),
        )

    @classmethod
    def load_common_type_names(cls, model_path: Path) -> set[str]:
        if (model_path / "common_names.pkl").exists():
            return pickle_load(model_path / "common_names.pkl")
        else:
            return set()

    def eval_on_dataset(
        self,
        src_data: TokenizedSrcSet,
        max_labels: Optional[int] = None,
        tqdm_args: dict = {},
    ) -> DatasetPredResult:
        """Convinient method to preprocess the src according to the model's ctx_args and evaluate the (R0) accuracy."""
        ctx_args = self.args.ctx_args
        if max_labels is not None:
            ctx_args = copy(ctx_args)
            ctx_args.max_labels = max_labels

        chunks = src_data.to_chunks(ctx_args, tqdm_args=tqdm_args)
        preds = self.predict(
            chunks.data, num_return_sequences=None, tqdm_args=tqdm_args
        )
        return DatasetPredResult(chunks, preds)


def dynamic_dataloader(
    dataset: Dataset,
    max_tokens: int,
    collate_fn,
    shuffle: bool = False,
):
    ex_sizes = [len(x) for x in dataset["input_ids"]]
    ids = list(range(len(ex_sizes)))
    if shuffle:
        random.shuffle(ids)
    ids.sort(key=lambda x: ex_sizes[x], reverse=True)
    batches = list[list[int]]()
    while len(ids) > 0:
        w = ex_sizes[ids[0]]
        n = max(1, max_tokens // w)
        batches.append(ids[:n])
        ids = ids[n:]
    if shuffle:
        random.shuffle(batches)

    return DataLoader(
        cast(Any, dataset),
        batch_sampler=batches,
        collate_fn=collate_fn,
    )
