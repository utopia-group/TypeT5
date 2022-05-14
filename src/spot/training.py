import logging
from copy import copy

import torch
import wandb
from datasets import Dataset, interleave_datasets
from IPython.display import display
from numpy import ndarray
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers.trainer import Trainer

from spot.data import (
    CtxArgs,
    SpecialNames,
    TypeInfDataset,
    chunk_masked_code,
    output_ids_as_types,
    patch_code_with_extra,
    pretty_print_accuracies,
    repos_to_dataset,
    type_accuracies,
)
from spot.type_env import (
    AnnotInfo,
    AnnotPath,
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
    ctx_args: CtxArgs
    sampling_batch_size: int
    train_batch_size: int
    max_workers: int
    eval_ctx_args: Optional[CtxArgs] = None
    generation_max_length: int = 128
    generation_num_beams: int = 8
    top_p: float = 0.9
    # when the eval metric is getting worse, how many more evaluation steps to wait before stopping training
    early_stopping_patience: int = 0
    skip_first_eval: bool = False
    per_file_feedback: bool = True


@dataclass
class DAggerTrainer:
    """
    Args:
        r0_model: the model used to make the initial predictions, will not be trained.
        r1_model: the model that combines the output of r0_model with the feedback
        from the type checker.
    """

    r0_model: ModelSPOT
    r1_model: ModelSPOT
    tokenizer: TokenizerSPOT
    args: DAggerTrainerArgs
    timer: TimeLogger = field(default_factory=TimeLogger)
    current_task: list[str] = field(default_factory=list)
    current_tqdm: Optional[tqdm] = None

    def train(
        self,
        train_repos: Sequence[Path],
        eval_repos: Sequence[Path],
    ) -> None:
        max_epochs = self.args.max_epochs
        repos_group_size = self.args.repos_group_size
        optimizer = AdamW(self.r1_model.parameters(), lr=5e-5, weight_decay=0.01)
        early_stopper = EarlyStopper(self.args.early_stopping_patience)
        best_model_dir = self.args.output_dir / "best_model"
        tqdm_bar = tqdm(range(max_epochs * len(train_repos)), desc="DAgger Training")
        self.current_tqdm = tqdm_bar

        def eval_step(epoch) -> float:
            with self.log_task("evaluating"):
                r0_r, r1_r = self.eval_on_repos(eval_repos, tqdm_args={"leave": False})
                r0_stats, r1_stats = r0_r[0], r1_r[0]
                self.log(r0_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R0 stats:")
                pretty_print_accuracies(r0_stats, max_show_level=0)
                self.log(r1_stats, step=tqdm_bar.n, epoch=epoch)
                print(f"[Epoch {epoch}] R1 stats:")
                pretty_print_accuracies(r1_stats, max_show_level=0)
                score = r1_stats["R1_full_acc"]
                assert isinstance(score, float)
                return score

        def train_step(epoch) -> None:
            with self.log_task("training"):
                for repos in grouped(train_repos, repos_group_size):
                    r0_data, r1_data, _ = self.generate_r1_inputs(
                        repos,
                        self.args.ctx_args,
                        tqdm_args={"disable": True},
                    )
                    # train_data = interleave_datasets(
                    #     [super_data.data, dagger_data.data]
                    # )
                    with self.log_task("train_on_data"):
                        train_r = self.train_on_data(r1_data.data, optimizer)
                    tqdm_bar.update(len(repos))
                    self.log(train_r, step=tqdm_bar.n, epoch=epoch)

        if not self.args.skip_first_eval:
            assert not early_stopper.should_stop(0, eval_step(0))
        for epoch in range(1, max_epochs + 1):
            train_step(epoch)
            score = eval_step(epoch)
            if early_stopper.should_stop(epoch, score):
                print(
                    f"[Epoch {epoch}] Early stopping since R1 accuracy has "
                    + f"not improved in {self.args.early_stopping_patience} epochs."
                )
                print("Loading best model...")
                self.r1_model = ModelSPOT.from_pretrained(best_model_dir)
                break
            elif epoch == early_stopper.best_step():
                with self.log_task("saving best model"):
                    self.r1_model.save_pretrained(best_model_dir)

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

    def generate_r1_inputs(
        self,
        repos: Sequence[Path],
        ctx_args: CtxArgs,
        tqdm_args: dict,
    ) -> tuple[TypeInfDataset, TypeInfDataset, list[list[PythonType]]]:
        """Generate two datasets from the given repos. One for training with supervised learning,
        the other for DAgger training, which combines feedback from the type checker."""
        ctx_args = copy(ctx_args)
        with self.log_task("repos_to_dataset"):
            ctx_args.types_in_ctx = True
            r0_data = repos_to_dataset(
                repos,
                self.tokenizer,
                ctx_args,
                max_workers=self.args.max_workers,
                tqdm_args=tqdm_args,
            )
        # make predictions on the original training set
        with self.log_task("predict"):
            r0_preds = self.predict(r0_data, self.r0_model, tqdm_args=tqdm_args)
        with self.log_task("get_type_checked_inputs"):
            new_inputs = self.get_type_checked_inputs(
                r0_data, r0_preds, repos, tqdm_args=tqdm_args
            )
            new_files = list(new_inputs.keys())
        with self.log_task("chunk_masked_code"):
            ctx_args.types_in_ctx = False
            r1_dataset, r1_meta = chunk_masked_code(
                list(new_inputs.values()),
                self.tokenizer,
                ctx_args,
                max_workers=self.args.max_workers,
                tqdm_args=tqdm_args,
            )
        r1_data = TypeInfDataset(r1_dataset, r1_meta, new_files, dict())
        return r0_data, r1_data, r0_preds

    def predict(
        self, dataset: TypeInfDataset, model: ModelSPOT, tqdm_args: dict
    ) -> list[list[PythonType]]:
        """Run the current model on the given dataset and return the predicted types."""
        collator = DataCollatorForSeq2Seq(self.tokenizer, model)
        loader = DataLoader(
            dataset.data,  # type: ignore
            shuffle=False,
            batch_size=self.args.sampling_batch_size,
            collate_fn=collator,
        )
        device = model.device
        pred_types = list[list[PythonType]]()
        tqdm_bar = tqdm(total=len(dataset.data), desc="predict", **tqdm_args)
        chunk_id = 0
        for batch in loader:
            output_ids = model.generate(
                inputs=batch["input_ids"].to(device),
                do_sample=True,
                top_p=self.args.top_p,
                max_length=self.args.generation_max_length,
                # num_beams=self.args.generation_num_beams,
            ).cpu()  # type: ignore
            assert len(output_ids.shape) == 2

            n_chunks = output_ids.shape[0]
            for i in range(n_chunks):
                n_annots = len(dataset.chunks_info[i + chunk_id].annots_info)
                row = output_ids[i, :]
                types = output_ids_as_types(row, self.tokenizer, n_annots)
                pred_types.append(types)
            chunk_id += n_chunks
            tqdm_bar.update(output_ids.shape[0])
        return pred_types

    def train_on_data(self, dataset: Dataset, optimizer: AdamW) -> dict:
        model = self.r1_model
        collator = DataCollatorForSeq2Seq(self.tokenizer, model)
        loader = DataLoader(
            dataset,  # type: ignore
            shuffle=True,
            batch_size=self.args.train_batch_size,
            collate_fn=collator,
        )

        device = model.device
        losses = []
        for batch in loader:
            outputs = model(
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

    def eval_on_repos(
        self,
        repos: Sequence[Path],
        ctx_args: Optional[CtxArgs] = None,
        tqdm_args: dict = {"leave": False},
    ) -> tuple[tuple[dict, TypeInfDataset, list[list[PythonType]]], ...]:
        ctx_args = ctx_args or self.args.eval_ctx_args or self.args.ctx_args
        r0_data, r1_data, r0_preds = self.generate_r1_inputs(repos, ctx_args, tqdm_args)
        r0_cats = [an.cat for info in r0_data.chunks_info for an in info.annots_info]
        r0_labels = [ty for info in r0_data.chunks_info for ty in info.types]
        r0_stats = {
            f"R0_{k}": v
            for k, v in type_accuracies(
                list(seq_flatten(r0_preds)), r0_labels, r0_cats
            ).items()
        }

        with self.log_task("predict"):
            r1_preds = self.predict(r1_data, self.r1_model, tqdm_args=tqdm_args)
        r1_cats = [an.cat for info in r1_data.chunks_info for an in info.annots_info]
        r1_labels = [ty for info in r1_data.chunks_info for ty in info.types]

        r1_stats = {
            f"R1_{k}": v
            for k, v in type_accuracies(
                list(seq_flatten(r1_preds)), r1_labels, r1_cats
            ).items()
        }
        return (r0_stats, r0_data, r0_preds), (r1_stats, r1_data, r1_preds)

    def _t_map(
        self,
        f: Callable[[T1], T2],
        xs: Iterable[T1],
        desc: str,
        tqdm_args={"disable": True},
    ) -> list[T2]:
        return thread_map(
            f,
            xs,
            max_workers=self.args.max_workers,
            desc=desc,
            **tqdm_args,
        )

    def get_type_checked_inputs(
        self,
        dataset: TypeInfDataset,
        pred_types: Sequence[Sequence[PythonType]],
        repo_paths: Sequence[Path],
        tqdm_args: dict,
    ) -> dict[Path, dict]:
        if self.args.per_file_feedback:
            return self.get_type_checked_inputs_per_file(
                dataset, pred_types, repo_paths, tqdm_args
            )
        else:
            return self.get_type_checked_inputs_per_project(
                dataset, pred_types, repo_paths, tqdm_args
            )

    def get_type_checked_inputs_per_file(
        self,
        dataset: TypeInfDataset,
        pred_types: Sequence[Sequence[PythonType]],
        repo_paths: Sequence[Path],
        tqdm_args: dict,
    ) -> dict[Path, dict]:
        """Apply the predicted types to each file, collect the corresponding type checker
        feedback (assuming other files have the correct types), then restore the file
        to its original contents."""

        def feedback_for_file(
            file: Path, changes: list[tuple[CodeRange, str]], repo: Path
        ) -> tuple[dict[CodePosition, str], str]:
            with open(file, "r") as f:
                origin_code = f.read()
            if MypyChecker.Preamble in origin_code:
                raise RuntimeError(f"{f} is already modified by SPOT.")
            code_seen = dataset.srcs[file]
            new_text = replace_strs_by_pos(code_seen, [(r, 1, v) for r, v in changes])
            file.write_text(new_text)
            try:
                pass
                check_r = MypyChecker.check_project(repo)
                if isinstance(check_r, str):
                    logging.warning(f"Mypy errored when checking '{dir}'.")
                    errors = {}
                else:
                    errors = dict(check_r.error_dict.get(file.resolve(), []))
            finally:
                file.write_text(origin_code)
            return errors, new_text

        file2changes = dict[Path, list[tuple[CodeRange, str]]]()

        assert len(repo_paths) == len(
            set(p.resolve() for p in repo_paths)
        ), "Repo paths must be unique"

        label_info = dataset.chunks_info
        for chunk_preds, chunk_info in zip(pred_types, label_info):
            for ty, info, sid in zip(
                chunk_preds, chunk_info.annots_info, chunk_info.src_ids
            ):
                file = dataset.files[sid]
                if file not in file2changes:
                    file2changes[file] = []
                assert info.annot_range is not None
                file2changes[file].append((info.annot_range, str(ty)))

        repo2files = dict[Path, list[Path]]()
        for repo in repo_paths:
            files = [f for f in repo.glob("**/*.py") if f in file2changes]
            repo2files[repo] = files

        def feedback_for_repo(
            repo: Path,
        ) -> dict[Path, tuple[dict[CodePosition, str], str]]:
            files = repo2files[repo]
            changes_list = [file2changes[f] for f in files]
            file2errors_content = {}
            for file, changes in zip(files, changes_list):
                start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
                changes.append((start, MypyChecker.Preamble))
                file2errors_content[file.resolve()] = feedback_for_file(
                    file, changes, repo
                )
            return file2errors_content

        repo_paths = list(
            sorted(repo_paths, key=lambda p: len(repo2files[p]), reverse=True)
        )

        feedback_list = self._t_map(
            feedback_for_repo,
            repo_paths,
            "Collect type checker feedback",
            tqdm_args=tqdm_args,
        )
        file2errors = dict[Path, dict[CodePosition, str]]()
        file2contents = dict[Path, str]()
        for ls in feedback_list:
            for k, v in ls.items():
                file2errors[k] = v[0]
                file2contents[k] = v[1]

        # generate feedback-augmented inputs
        changed_files = list(file2changes.keys())
        file_errors = [file2errors.get(f.resolve(), {}) for f in changed_files]
        new_contents = [file2contents[f.resolve()] for f in changed_files]
        origin_contents = [f.read_text() for f in changed_files]
        max_workers = self.args.max_workers
        new_inputs = process_map(
            _generate_augmented_inputs,
            changed_files,
            file_errors,
            origin_contents,
            new_contents,
            chunksize=max(len(file2changes) // (8 * max_workers), 1),
            max_workers=max_workers,
            desc="generating augmented inputs",
            **tqdm_args,
        )
        return dict(zip(changed_files, new_inputs))

    def get_type_checked_inputs_per_project(
        self,
        dataset: TypeInfDataset,
        pred_types: Sequence[Sequence[PythonType]],
        repo_paths: Sequence[Path],
        tqdm_args: dict,
    ) -> dict[Path, dict]:
        """Apply the predicted types to the given files in each project first and then
        collect the type checker feedback. Will always restore the files to
        their original contents afterwards."""
        file2changes = dict[Path, list[tuple[CodeRange, str]]]()

        assert len(repo_paths) == len(
            set(p.resolve() for p in repo_paths)
        ), "Repo paths must be unique"

        label_info = dataset.chunks_info
        for chunk_preds, chunk_info in zip(pred_types, label_info):
            for ty, info, sid in zip(
                chunk_preds, chunk_info.annots_info, chunk_info.src_ids
            ):
                file = dataset.files[sid]
                if file not in file2changes:
                    file2changes[file] = []
                assert info.annot_range is not None
                file2changes[file].append((info.annot_range, str(ty)))

        changed_files = list(file2changes.keys())
        origin_contents = self._t_map(
            read_file,
            changed_files,
            "reading orginal srcs",
            tqdm_args=tqdm_args,
        )
        path_to_original = dict[Path, str]()
        for f, text in zip(changed_files, origin_contents):
            if MypyChecker.Preamble in text:
                raise RuntimeError(f"{f} is already modified by SPOT.")
            path_to_original[f] = text

        max_workers = self.args.max_workers

        try:
            # apply the file changes and get type checker feedback
            current_contents = list[str]()
            for file, changes in file2changes.items():
                start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
                # need this in case libcst does not preserve the original file content
                code_seen = dataset.srcs[file]
                changes.insert(0, (start, MypyChecker.Preamble))
                new_text = replace_strs_by_pos(
                    code_seen, [(r, 1, v) for r, v in changes]
                )
                current_contents.append(new_text)
                file.write_text(new_text)
            file2errors = dict[Path, dict[CodePosition, str]]()
            file_to_repo = dict[Path, Path]()

            with self.log_task("Call mypy"):
                check_results: list[MypyResult | str] = thread_map(
                    MypyChecker.check_project,
                    repo_paths,
                    max_workers=max_workers,
                    desc="calling mypy",
                    **tqdm_args,
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
            file_errors = [file2errors.get(f.resolve(), {}) for f in changed_files]
            new_inputs = process_map(
                _generate_augmented_inputs,
                changed_files,
                file_errors,
                origin_contents,
                current_contents,
                chunksize=max(len(file2changes) // (8 * max_workers), 1),
                max_workers=max_workers,
                desc="generating augmented inputs",
                **tqdm_args,
            )
            return dict(zip(changed_files, new_inputs))
        finally:
            # restore the files to their original contents
            for file, content in path_to_original.items():
                file.write_text(content)

    def log(self, metrics: dict, **additional) -> None:
        wandb.log({**metrics, **additional})


def _generate_augmented_inputs(
    file: Path,
    file_errors: dict[CodePosition, str],
    original_code: str,
    current_code: str,
) -> dict:
    origin_mod = cst.parse_module(original_code)
    origin_labels = collect_annotations(origin_mod)
    path2types = dict[AnnotPath, PythonType]()
    annots_info = list[AnnotInfo]()
    for info in origin_labels:
        if info.annot is None:
            continue
        ty = parse_type_expr(origin_mod, info.annot.annotation, silent=True)
        if ty is not None:
            annots_info.append(info)
            path2types[info.path] = ty

    try:
        m = cst.parse_module(current_code)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse file: '{file}' with content:\n{current_code}"
        ) from e
    m_code = m.code
    assert m_code == current_code, "Code 1:\n<<{}>>\nCode 2:\n<<{}>>".format(
        current_code, m_code
    )
    current_annots = collect_annotations(m)
    preds_map = dict[CodeRange, str]()
    types = list[PythonType]()
    for a in current_annots:
        if a.path in path2types:
            assert (range := a.annot_range) is not None
            assert (annot := a.annot) is not None
            preds_map[range] = m.code_for_node(annot.annotation)
            types.append(path2types[a.path])
    new_code = patch_code_with_extra(current_code, preds_map, file_errors)
    code_segs = new_code.split(SpecialNames.TypeMask)
    assert len(code_segs) == len(types) + 1, f"{len(code_segs)} != {len(types)} + 1"

    return {
        "code_segs": code_segs,
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
