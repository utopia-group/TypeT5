import enum
import logging
import random
from collections import Counter
from copy import copy, deepcopy

import numpy as np
import torch
from datasets import Dataset
from mypy_extensions import mypyc_attr
from torch.utils.data import DataLoader, RandomSampler
from transformers import DataCollatorForSeq2Seq
from transformers.trainer import Trainer

from spot.data import (
    ChunkedDataset,
    CtxArgs,
    R1_srcs_from_preds,
    SrcDataset,
    output_ids_as_types,
    patch_code_with_extra,
    preds_to_accuracies,
)
from spot.type_env import PythonType
from spot.utils import *


@dataclass
class DecodingArgs:
    ctx_args: Optional[CtxArgs]
    sampling_batch_size: int
    max_workers: int
    max_tokens_per_type: int = 10
    do_sample: bool = False
    top_p: float = 0.9

    def scale_ctx_size(self, factor: float) -> "DecodingArgs":
        result = deepcopy(self)
        if self.ctx_args is None:
            return result
        assert result.ctx_args is not None
        """Scale the context size of the model by the given factor, while keeping the window size the same.
        Also scale down the sampling batch size accordingly."""
        ctx_size = round(self.ctx_args.ctx_size * factor)
        right_margin = round(self.ctx_args.right_margin * factor)
        left_margin = ctx_size - right_margin - self.ctx_args.window_size
        result.ctx_args.ctx_size = ctx_size
        result.ctx_args.left_margin = left_margin
        result.ctx_args.right_margin = right_margin
        result.sampling_batch_size = round(self.sampling_batch_size / factor**2)

        return result


@dataclass
class ModelTrainingArgs:
    train_batch_size: int
    eval_batch_size: int
    max_epochs: int
    accumulate_grad_batches: int | dict | None = None


@dataclass
class ModelWrapper:
    model: ModelSPOT
    tokenizer: TokenizerSPOT
    args: DecodingArgs
    monitor: TaskMonitor

    def scale_ctx_size(self, factor) -> "ModelWrapper":
        r = copy(self)
        r.args = r.args.scale_ctx_size(factor)
        return r

    def predict_on_batch(
        self,
        batch: dict,
        do_sample=None,
    ) -> list[list[PythonType]]:
        """Run the model on the given batch and return the predicted types for each row."""
        model = self.model
        n_labels = batch["n_labels"]
        max_labels = max(n_labels)

        output_ids = model.generate(
            inputs=batch["input_ids"],
            do_sample=self.args.do_sample if do_sample is None else do_sample,
            top_p=self.args.top_p,
            max_length=self.args.max_tokens_per_type * max_labels,
        ).cpu()  # type: ignore
        assert len(output_ids.shape) == 2

        n_chunks = output_ids.shape[0]
        preds = list[list[PythonType]]()
        for i in range(n_chunks):
            row = output_ids[i, :]
            types = output_ids_as_types(row, self.tokenizer, n_labels[i])
            preds.append(types)
        return preds

    def predict(self, dataset: Dataset, tqdm_args: dict) -> list[list[PythonType]]:
        """Run the  model on the given dataset and return the predicted types for each row."""
        model = self.model
        collator = DataCollatorForSeq2Seq(self.tokenizer, model)
        loader = dynamic_dataloader(
            dataset,  # type: ignore
            batch_size=self.args.sampling_batch_size,
            collate_fn=collator,
            train_per_file=self.args.ctx_args is None,
        )
        device = model.device
        pred_types = dict[int, list[PythonType]]()
        tqdm_bar = tqdm(total=len(dataset), desc="predict", **tqdm_args)
        for batch in loader:
            n_chunks = batch["input_ids"].shape[0]
            batch["input_ids"] = batch["input_ids"].to(device)
            preds = self.predict_on_batch(batch)
            for i, p in zip(batch["chunk_id"], preds):
                pred_types[int(i)] = p
            tqdm_bar.update(n_chunks)
        return [pred_types[int(i)] for i in dataset["chunk_id"]]

    def save_pretrained(self, path: Path):
        """Save the model to the given path along with its tokenizer and args."""
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        with open(path / "args.pkl", "wb") as f:
            pickle.dump(self.args, f)

    def to(self, device) -> "ModelWrapper":
        self.model = self.model.to(device)
        return self

    @staticmethod
    def from_pretrained(path: Path) -> "ModelWrapper":
        """Load a pretrained model from the given path."""
        model = ModelSPOT.from_pretrained(str(path))
        tokenizer = TokenizerSPOT.from_pretrained(str(path))
        with open(path / "args.pkl", "rb") as f:
            args = pickle.load(f)
        return ModelWrapper(
            model=model,
            tokenizer=tokenizer,
            args=args,
            monitor=TaskLoggingMonitor(path.name),
        )

    def eval_on_dataset(
        self, src_data: SrcDataset, tqdm_args={}
    ) -> tuple[dict, ChunkedDataset, list[list[PythonType]]]:
        """Convinient method to preprocess the src according to the model's ctx_args and evaluate the (R0) accuracy."""
        chunks = src_data.to_chunks(
            self.tokenizer,
            self.args.ctx_args,
            max_workers=self.args.max_workers,
            tqdm_args=tqdm_args,
        )
        preds = self.predict(chunks.data, tqdm_args=tqdm_args)
        accs = preds_to_accuracies(preds, chunks, normalize_types=True)
        accs_strict = preds_to_accuracies(preds, chunks, normalize_types=False)
        accs["partial_acc_strict"] = accs_strict["partial_acc"]
        accs["full_acc_strict"] = accs_strict["full_acc"]
        return (accs, chunks, preds)


def dynamic_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn,
    train_per_file: bool,
    shuffle: bool = False,
):

    if not train_per_file:
        return DataLoader(
            cast(Any, dataset),
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )

    ex_sizes = [len(x) for x in dataset["input_ids"]]
    ids = list(range(len(ex_sizes)))
    if shuffle:
        random.shuffle(ids)
    ids.sort(key=lambda x: ex_sizes[x], reverse=True)
    batches = list[list[int]]()
    while len(ids) > 0:
        w = ex_sizes[ids[0]]
        n = max(1, batch_size // w)
        batches.append(ids[:n])
        ids = ids[n:]
    if shuffle:
        random.shuffle(batches)

    return DataLoader(
        cast(Any, dataset),
        batch_sampler=batches,
        collate_fn=collate_fn,
    )


@dataclass
class CombinedModel:
    r0_wrapper: ModelWrapper
    r1_wrapper: ModelWrapper

    def eval_on_dataset(self, dataset: SrcDataset, tqdm_args={}) -> tuple[dict, dict]:
        r0_wrapper, r1_wrapper = self.r0_wrapper, self.r1_wrapper
        r0_accs, r0_chunks, r0_preds = r0_wrapper.eval_on_dataset(
            dataset, tqdm_args={"leave": False}
        )
        r1_srcs = R1_srcs_from_preds(
            r1_wrapper.tokenizer,
            dataset,
            r0_chunks.chunks_info,
            r0_chunks.files,
            r0_preds,
            max_workers=r0_wrapper.args.max_workers,
        )
        r1_accs, _, _ = r1_wrapper.eval_on_dataset(r1_srcs, tqdm_args={"leave": False})
        return r0_accs, r1_accs
