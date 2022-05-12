import logging

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
    AnnotInfo,
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
    # when the eval metric is getting worse, how many more evaluation steps to wait before stopping training
    early_stopping_patience: int = 0


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
        early_stopper = EarlyStopper(self.args.early_stopping_patience)
        best_model_dir = self.args.output_dir / "best_model"
        tqdm_bar = tqdm(range(max_epochs * len(train_repos)), desc="DAgger Training")
        self.current_tqdm = tqdm_bar
        for epoch in range(max_epochs + 1):
            # perform evaluation
            with self.log_task("evaluating"):
                r0_stats, r1_stats = self.eval_on_repos(eval_repos)
                self.log(r0_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R0 stats: {r0_stats}")
                self.log(r1_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R1 stats: {r1_stats}")
                if early_stopper.should_stop(epoch, r1_stats["R1_accuracy_full"]):
                    print(
                        f"[Epoch {epoch}] Early stopping since R1 accuracy has "
                        + f"not improved in {self.args.early_stopping_patience} epochs."
                    )
                    print("Loading best model...")
                    self.model = ModelSPOT.from_pretrained(best_model_dir)
                    break
                elif epoch == early_stopper.best_step():
                    with self.log_task("saving best model"):
                        self.model.save_pretrained(best_model_dir)

            if epoch == max_epochs:
                break

            with self.log_task("training"):
                for repos in grouped(train_repos, repos_group_size):
                    super_data, dagger_data, _ = self.generate_data(repos, silent=True)
                    train_data = interleave_datasets(
                        [super_data.data, dagger_data.data]
                    )
                    with self.log_task("train_on_data"):
                        train_r = self.train_on_data(train_data, optimizer)
                    tqdm_bar.update(len(repos))
                    self.log(train_r, step=tqdm_bar.n, epoch=epoch)
        self.current_tqdm = None

    @contextmanager
    def log_task(self, name: str):
        self.current_task.append(name)
        task_name = " > ".join(self.current_task)
        if self.current_tqdm is not None:
            self.current_tqdm.set_postfix_str(f"Current task: {task_name}")
        try:
            with self.timer.log_time(task_name):
                yield
        finally:
            self.current_task.pop()
            if self.current_tqdm is not None:
                task_name = " > ".join(self.current_task)
                self.current_tqdm.set_postfix_str(f"Current task: {task_name}")

    def generate_data(
        self,
        repos: Sequence[Path],
        silent: bool,
    ) -> tuple[TypeInfDataset, TypeInfDataset, list[PythonType]]:
        """Generate two datasets from the given repos. One for training with supervised learning,
        the other for DAgger training, which combines feedback from the type checker."""
        with self.log_task("repos_to_dataset"):
            r0_data = repos_to_dataset(
                repos,
                self.tokenizer,
                max_workers=self.args.max_workers,
                ctx_margin=self.args.ctx_margin,
                silent=silent,
            )
        # make predictions on the original training set
        with self.log_task("predict"):
            pred_types = self.predict(r0_data.data, silent=silent)
        with self.log_task("get_type_checked_inputs"):
            new_inputs = self.get_type_checked_inputs(
                r0_data, pred_types, repos, silent=silent
            )
        with self.log_task("chunk_masked_code"):
            dagger_dataset, dager_meta = chunk_masked_code(
                list(new_inputs.values()),
                self.args.ctx_size,
                self.tokenizer,
                max_workers=self.args.max_workers,
                ctx_margin=self.args.ctx_margin,
                silent=silent,
            )
        r1_data = TypeInfDataset(
            dagger_dataset, dager_meta, r0_data.files, r0_data.srcs
        )
        return r0_data, r1_data, pred_types

    def predict(self, dataset: Dataset, silent: bool) -> list[PythonType]:
        """Run the current model on the given dataset and return the predicted types."""
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        loader = DataLoader(
            dataset,  # type: ignore
            shuffle=False,
            batch_size=self.args.sampling_batch_size,
            collate_fn=collator,
        )
        device = self.model.device
        pred_types = list[PythonType]()
        tqdm_bar = tqdm(range(len(loader)), desc="predict", disable=silent)
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
            tqdm_bar.update(output_ids.shape[0])
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

    def eval_on_data(self, dataset: TypeInfDataset, silent: bool = True) -> dict:
        with self.log_task("predict"):
            pred_types = self.predict(dataset.data, silent=silent)
        cats = [i.cat for i in dataset.meta.annots_info]
        labels = dataset.meta.types
        return type_accuracies(pred_types, labels, cats)

    def eval_on_repos(
        self, repos: Sequence[Path], silent: bool = True
    ) -> tuple[dict, dict]:
        r0_data, r1_data, r0_preds = self.generate_data(repos, silent=silent)
        eval_cats = [i.cat for i in r0_data.meta.annots_info]
        labels = r0_data.meta.types
        r0_stats = {
            f"R0_{k}": v
            for k, v in type_accuracies(r0_preds, labels, eval_cats).items()
        }
        r1_stats = {
            f"R1_{k}": v for k, v in self.eval_on_data(r1_data, silent=silent).items()
        }
        return r0_stats, r1_stats

    def _t_map(
        self, f: Callable[[T1], T2], xs: Iterable[T1], desc: str, silent=True
    ) -> list[T2]:
        return thread_map(
            f,
            xs,
            max_workers=self.args.max_workers,
            desc=desc,
            disable=silent,
        )

    def get_type_checked_inputs(
        self,
        dataset: TypeInfDataset,
        pred_types: Sequence[PythonType],
        repo_paths: Sequence[Path],
        silent: bool,
    ) -> dict[Path, dict]:
        """Apply the predicted types to the given files and collect the type checker feedback, then restore the
        files to their original contents."""
        file2changes = dict[Path, list[tuple[CodeRange, str]]]()

        assert len(repo_paths) == len(
            set(p.resolve() for p in repo_paths)
        ), "Repo paths must be unique"

        label_info = dataset.meta
        for ty, info, sid in zip(
            pred_types, label_info.annots_info, label_info.src_ids
        ):
            file = dataset.files[sid]
            if file not in file2changes:
                file2changes[file] = []
            assert info.annot_range is not None
            file2changes[file].append((info.annot_range, str(ty)))

        origin_contents = self._t_map(
            read_file,
            file2changes.keys(),
            "reading orginal srcs",
            silent=silent,
        )
        path_to_original = dict[Path, str]()
        for f, text in zip(file2changes.keys(), origin_contents):
            if MypyChecker.Preamble in text:
                raise RuntimeError(f"{f} is already modified by SPOT.")
            path_to_original[f] = text

        max_workers = self.args.max_workers

        try:
            # apply the file changes and get type checker feedback
            for file, changes in tqdm(
                file2changes.items(), desc="apply file changes", disable=silent
            ):
                start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
                changes.insert(0, (start, MypyChecker.Preamble))
                # need this in case libcst does not preserve the original file content
                code_seen = dataset.srcs[file]
                new_text = replace_strs_by_pos(code_seen, changes)
                file.write_text(new_text)
            file2errors = dict[Path, dict[CodePosition, str]]()
            file_to_repo = dict[Path, Path]()

            with self.log_task("Call mypy"):
                check_results: list[MypyResult | str] = thread_map(
                    MypyChecker.check_project,
                    repo_paths,
                    max_workers=max_workers,
                    desc="calling mypy",
                    disable=silent,
                )
            for dir, check_r in zip(repo_paths, check_results):
                if isinstance(check_r, str):
                    logging.warning(f"Mypy errored when checking '{dir}'.")
                    continue
                for file, errors in check_r.error_dict.items():
                    assert (
                        file not in file_to_repo
                    ), f"{file} appears in multiple repos? repo1: {file_to_repo[file]}, repo2: {dir}"
                    file2errors[file] = dict(errors)
                    file_to_repo[file] = dir

            # generate feedback-augmented inputs
            file_errors = [
                file2errors.get(f.resolve(), {}) for f in file2changes.keys()
            ]
            new_inputs = process_map(
                _generate_augmented_inputs,
                file2changes.keys(),
                file_errors,
                origin_contents,
                chunksize=max(len(file2changes) // (8 * max_workers), 1),
                max_workers=max_workers,
                desc="generating augmented inputs",
                disable=silent,
            )
            return dict(zip(file2changes.keys(), new_inputs))
        finally:
            # restore the files to their original contents
            for file, content in path_to_original.items():
                file.write_text(content)

    def log(self, metrics: dict, **additional) -> None:
        wandb.log({**metrics, **additional})


def _generate_augmented_inputs(
    file: Path, file_errors: dict[CodePosition, str], original_src: str
) -> dict:
    origin_mod = cst.parse_module(original_src)
    origin_labels = collect_annotations(origin_mod)
    types = list[PythonType]()
    annots_info = list[AnnotInfo]()
    for info in origin_labels:
        if info.annot is None:
            continue
        ty = parse_type_expr(origin_mod, info.annot.annotation, silent=True)
        if ty is not None:
            types.append(ty)
            annots_info.append(info)
    paths_of_interest = set(info.path for info in annots_info)

    current_code = file.read_text()  # the modifed code with R0 predictions
    try:
        m = cst.parse_module(current_code)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse file: '{file}' with content:\n{current_code}"
        ) from e
    current_code = m.code  # use the code from the CST
    current_annots = collect_annotations(m)
    preds_map = dict[CodeRange, str]()
    for a in current_annots:
        if a.path in paths_of_interest:
            assert (range := a.annot_range) is not None
            assert (annot := a.annot) is not None
            preds_map[range] = m.code_for_node(annot.annotation)
    new_code = patch_code_with_extra(current_code, preds_map, file_errors)

    return {
        "code_segs": new_code.split(SpecialNames.TypeMask),
        "types": types,
        "annots_info": annots_info,
    }


@dataclass
class EarlyStopper:
    patience: int
    current_best: Optional[tuple[int, float]] = None

    def should_stop(self, eval_step: int, score: float) -> bool:
        if self.current_best is None or score > self.current_best[1]:
            self.current_best = (eval_step, score)
            return False
        # score is worse
        return eval_step - self.current_best[0] >= self.patience

    def best_step(self) -> Optional[int]:
        if self.current_best is None:
            return None
        return self.current_best[0]
