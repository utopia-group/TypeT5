import torch
from datasets import Dataset, interleave_datasets
from numpy import ndarray
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer import Trainer

import wandb
from spot.data import (
    GitRepo,
    LabelMetaInfo,
    SpecialNames,
    TypeInfDataset,
    chunk_masked_code,
    output_ids_as_types,
    patch_code_with_extra,
    repos_to_dataset,
    type_accuracies,
)
from spot.type_env import (
    MypyChecker,
    MypyResult,
    PythonType,
    collect_annotations,
    parse_type_expr,
)

from .model import ModelSPOT, TokenizerSPOT
from .utils import *


@dataclass
class DAggerTrainerArgs:
    output_dir: Path
    max_epochs: int
    repos_group_size: int
    ctx_size: int
    ctx_margin: int
    sampling_batch_size: int
    train_batch_size: int
    max_workers: int
    generation_max_length: int = 128
    generation_num_beams: int = 8
    top_p: float = 0.9


@dataclass
class DAggerTrainer:
    model: ModelSPOT
    tokenizer: TokenizerSPOT
    args: DAggerTrainerArgs
    timer: TimeLogger = field(default_factory=TimeLogger)
    current_task: list[str] = field(default_factory=list)
    current_tqdm: Optional[tqdm] = None

    def train(self, train_repos: Sequence[Path], eval_repos: Sequence[Path]) -> None:
        max_epochs = self.args.max_epochs
        repos_group_size = self.args.repos_group_size
        optimizer = AdamW(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        tqdm_bar = tqdm(range(max_epochs * len(train_repos)), desc="DAgger Training")
        self.current_tqdm = tqdm_bar
        for epoch in range(max_epochs):
            with self.log_task("training"):
                for repos in grouped(train_repos, repos_group_size):
                    super_data, dagger_data, _ = self.generate_data(repos)
                    train_data = interleave_datasets(
                        [super_data.data, dagger_data.data]
                    )
                    with self.log_task("model fitting"):
                        train_r = self.train_on_data(train_data, optimizer)
                    tqdm_bar.update(len(repos))
                    self.log(train_r, step=tqdm_bar.n, epoch=epoch)
            # perform evaluation
            with self.log_task("evaluating"):
                r0_data, r1_data, r0_preds = self.generate_data(eval_repos)
                eval_cats = [i.cat for i in r0_data.meta.annots_info]
                labels = r0_data.meta.types
                r0_stats = {
                    f"R0_{k}": v
                    for k, v in type_accuracies(r0_preds, labels, eval_cats).items()
                }
                self.log(r0_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R0 stats: {r0_stats}")
                r1_stats = {f"R1_{k}": v for k, v in self.eval_on_data(r1_data).items()}
                self.log(r1_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R1 stats: {r1_stats}")
        self.current_tqdm = None

    @contextmanager
    def log_task(self, name: str):
        self.current_task.append(name)
        task_name = " > ".join(self.current_task)
        if self.current_tqdm is not None:
            self.current_tqdm.set_postfix_str(f"Current task: {task_name}")
        with self.timer.log_time(task_name):
            yield
        self.current_task.pop()
        if self.current_tqdm is not None:
            task_name = " > ".join(self.current_task)
            self.current_tqdm.set_postfix_str(f"Current task: {task_name}")

    def generate_data(
        self, repos: Sequence[Path]
    ) -> tuple[TypeInfDataset, TypeInfDataset, list[PythonType]]:
        """Generate two datasets from the given repos. One for training with supervised learning,
        the other for DAgger training, which combines feedback from the type checker."""
        with self.log_task("preparing data"):
            r0_data = repos_to_dataset(
                repos,
                self.tokenizer,
                self.args.max_workers,
                self.args.ctx_margin,
                silent=True,
            )
        # make predictions on the original training set
        with self.log_task("model prediction"):
            pred_types = self.predict(r0_data.data)
        with self.log_task("type checking"):
            new_inputs = self.get_type_checked_inputs(
                repos, pred_types, r0_data.meta, r0_data.files
            )
        with self.log_task("preparing data"):
            dagger_dataset, dager_meta = chunk_masked_code(
                list(new_inputs.values()),
                self.args.ctx_size,
                self.tokenizer,
                self.args.ctx_margin,
                silent=True,
            )
        r1_data = TypeInfDataset(dagger_dataset, dager_meta, r0_data.files)
        return r0_data, r1_data, pred_types

    def predict(self, dataset: Dataset) -> list[PythonType]:
        """Run the current model on the given dataset and return the predicted types."""
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        loader = DataLoader(
            dataset,  # type: ignore
            shuffle=False,
            batch_size=self.args.sampling_batch_size,
            collate_fn=collator,
        )
        device = self.model.device
        pred_types = []
        for batch in loader:
            output_ids = self.model.generate(
                inputs=batch["input_ids"].to(device),
                do_sample=True,
                top_p=self.args.top_p,
                max_length=self.args.generation_max_length,
                # num_beams=self.args.generation_num_beams,
            ).cpu()  # type: ignore
            assert len(output_ids.shape) == 2

            for i in range(output_ids.shape[0]):
                row = output_ids[i, :]
                types = output_ids_as_types(row, self.tokenizer, batch["n_types"][i])
                pred_types.extend(types)
        return pred_types

    def train_on_data(self, dataset: Dataset, optimizer: AdamW) -> dict:
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        loader = DataLoader(
            dataset,  # type: ignore
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=collator,
        )

        device = self.model.device
        losses = []
        for batch in loader:
            outputs = self.model(
                input_ids=batch["input_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
        avg_loss = sum(losses) / len(losses)
        return {"loss": avg_loss}

    def eval_on_data(self, dataset: TypeInfDataset) -> dict:
        with self.log_task("model prediction"):
            pred_types = self.predict(dataset.data)
        cats = [i.cat for i in dataset.meta.annots_info]
        labels = dataset.meta.types
        return type_accuracies(pred_types, labels, cats)

    def get_type_checked_inputs(
        self,
        repo_paths: Sequence[Path],
        pred_types: Sequence[PythonType],
        label_info: LabelMetaInfo,
        parsed_files: Sequence[Path],
    ) -> dict[Path, dict]:
        """Apply the predicted types to the given files and collect the type checker feedback, then restore the
        files to their original contents."""
        file_changes = dict[Path, list[tuple[CodeRange, str]]]()

        assert len(repo_paths) == len(
            set(p.resolve() for p in repo_paths)
        ), "Repo paths must be unique"

        for ty, info, sid in zip(
            pred_types, label_info.annots_info, label_info.src_ids
        ):
            file = parsed_files[sid]
            if file not in file_changes:
                file_changes[file] = []
            assert info.annot_range is not None
            file_changes[file].append((info.annot_range, str(ty)))

        origin_contents = {p: p.read_text() for p in file_changes}
        for f, text in origin_contents.items():
            if MypyChecker.Preamble in text:
                raise RuntimeError(f"{f} is already modified by SPOT.")

        try:
            # commit the changes and get type checker feedback
            for file, changes in file_changes.items():
                start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
                changes.insert(0, (start, MypyChecker.Preamble))
                new_text = replace_strs_by_pos(origin_contents[file], changes)
                file.write_text(new_text)
            file_to_errors = dict[Path, dict[CodePosition, str]]()
            file_to_repo = dict[Path, Path]()
            check_results: list[MypyResult] = process_map(
                MypyChecker.check_project,
                repo_paths,
                max_workers=self.args.max_workers,
                desc="type checking",
                chunk_size=1,
                disable=True,
            )
            for dir, check_r in zip(repo_paths, check_results):
                for file, errors in check_r.error_dict.items():
                    assert (
                        file not in file_to_repo
                    ), f"{file} appears in multiple repos? repo1: {file_to_repo[file]}, repo2: {dir}"
                    file_to_errors[file] = dict(errors)
                    file_to_repo[file] = dir
            new_inputs = dict[Path, dict]()
            for file in file_changes:
                code = file.read_text()
                m = cst.parse_module(code)
                annots = collect_annotations(m)
                preds_map = {
                    a.annot_range: m.code_for_node(a.annot.annotation)
                    for a in annots
                    if a.annot is not None and a.annot_range is not None
                }
                file_errors = file_to_errors.pop(file.resolve(), {})
                new_code = patch_code_with_extra(code, preds_map, file_errors)
                origin_mod = cst.parse_module(origin_contents[file])
                origin_labels = collect_annotations(origin_mod)
                types = [
                    parse_type_expr(origin_mod, info.annot.annotation, silent=True)
                    for info in origin_labels
                    if info.annot is not None
                ]
                new_src = {
                    "code_segs": new_code.split(SpecialNames.TypeMask),
                    "types": types,
                    "annots_info": [
                        info for info in origin_labels if info.annot is not None
                    ],
                }
                new_inputs[file] = new_src

            # files that fail to parse can invalidate this
            # unchanged_files = set(
            #     f.resolve() for f in parsed_files if f not in file_changes
            # )
            # for f in file_to_errors:
            #     assert f in unchanged_files, f"Unhandled errors in file: {f}"

            return new_inputs
        finally:
            # restore the files to their original contents
            for file, content in origin_contents.items():
                file.write_text(content)

    def log(self, metrics: dict, **additional) -> None:
        wandb.log({**metrics, **additional})
