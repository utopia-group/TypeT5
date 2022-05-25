import enum
import logging
from collections import Counter
from copy import deepcopy

import numpy as np
import torch
from datasets import Dataset
from mypy_extensions import mypyc_attr
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer import Trainer

from spot.data import (
    ChunkedDataset,
    CtxArgs,
    SrcChunkInfo,
    SrcDataset,
    TokenizedSrc,
    output_ids_as_types,
    patch_code_with_extra,
    preds_to_accuracies,
)
from spot.type_env import (
    AnnotInfo,
    AnnotPath,
    MypyChecker,
    MypyResult,
    PythonType,
    collect_annots_info,
    collect_user_annotations,
    mypy_checker,
    parse_type_expr,
)
from spot.utils import *


@dataclass
class DecodingArgs:
    ctx_args: CtxArgs
    sampling_batch_size: int
    max_workers: int
    generation_max_length: int = 128
    do_sample: bool = False
    top_p: float = 0.9

    def scale_ctx_size(self, factor: float) -> "DecodingArgs":
        """Scale the context size of the model by the given factor, while keeping the window size the same.
        Also scale down the sampling batch size accordingly."""
        ctx_size = round(self.ctx_args.ctx_size * factor)
        right_margin = round(self.ctx_args.right_margin * factor)
        left_margin = ctx_size - right_margin - self.ctx_args.window_size
        result = deepcopy(self)
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

    def predict_on_batch(
        self,
        batch: dict,
        do_sample=None,
    ) -> list[list[PythonType]]:
        """Run the model on the given batch and return the predicted types for each row."""
        model = self.model
        output_ids = model.generate(
            inputs=batch["input_ids"],
            do_sample=self.args.do_sample if do_sample is None else do_sample,
            top_p=self.args.top_p,
            max_length=self.args.generation_max_length,
        ).cpu()  # type: ignore
        assert len(output_ids.shape) == 2

        n_chunks = output_ids.shape[0]
        preds = list[list[PythonType]]()
        n_labels = batch["n_labels"]
        for i in range(n_chunks):
            row = output_ids[i, :]
            types = output_ids_as_types(row, self.tokenizer, n_labels[i])
            preds.append(types)
        return preds

    def predict(self, dataset: Dataset, tqdm_args: dict) -> list[list[PythonType]]:
        """Run the  model on the given dataset and return the predicted types for each row."""
        model = self.model
        collator = DataCollatorForSeq2Seq(self.tokenizer, model)
        loader = DataLoader(
            dataset,  # type: ignore
            shuffle=False,
            batch_size=self.args.sampling_batch_size,
            collate_fn=collator,
        )
        device = model.device
        pred_types = list[list[PythonType]]()
        tqdm_bar = tqdm(total=len(dataset), desc="predict", **tqdm_args)
        chunk_id = 0
        for batch in loader:
            n_chunks = batch["input_ids"].shape[0]
            batch["input_ids"] = batch["input_ids"].to(device)
            preds = self.predict_on_batch(batch)
            pred_types.extend(preds)
            chunk_id += n_chunks
            tqdm_bar.update(n_chunks)
        return pred_types

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
        accs = preds_to_accuracies(preds, chunks)
        return (accs, chunks, preds)

    def build_trainer(
        self,
        output_dir: Path,
        train_args: ModelTrainingArgs,
        dataset: Dataset,
        eval_dataset: Dataset,
    ) -> Trainer:
        trainer_args = Seq2SeqTrainingArguments(
            str(output_dir),
            evaluation_strategy="steps",  # type: ignore
            eval_steps=500,
            eval_accumulation_steps=5,
            logging_steps=500,
            prediction_loss_only=True,
            save_strategy="steps",  # type: ignore
            save_steps=500,
            save_total_limit=3,
            learning_rate=2e-5,
            per_device_train_batch_size=train_args.train_batch_size,
            per_device_eval_batch_size=train_args.eval_batch_size,
            weight_decay=0.01,
            num_train_epochs=train_args.max_epochs,
            load_best_model_at_end=True,
            fp16=True,
            push_to_hub=False,
            report_to="wandb",  # type: ignore
        )
        model = self.model
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model)

        trainer: Trainer = Seq2SeqTrainer(
            model,
            trainer_args,
            train_dataset=dataset,  # type: ignore
            eval_dataset=eval_dataset,  # type: ignore
            data_collator=data_collator,  # type: ignore
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(3)],
        )

        return trainer

    def type_check_preds_per_file(
        self,
        dataset: ChunkedDataset,
        pred_types: Sequence[Sequence[PythonType]],
        tqdm_args: dict,
    ) -> dict[Path, dict]:
        """Apply the predicted types to each file, collect the corresponding type checker
        feedback (assuming other files have the correct types), then restore the file
        to its original contents."""

        max_workers = self.args.max_workers

        file2labels = dict[Path, dict[AnnotPath, PythonType]]()
        file2changes = dict[Path, list[tuple[CodeRange, str]]]()
        repo_paths = set[Path]()

        file2origin_code = dict[Path, str]()
        label_info = dataset.chunks_info
        for chunk_preds, chunk_info in zip(pred_types, label_info):
            for ty, label, info, sid in zip(
                chunk_preds,
                chunk_info.types,
                chunk_info.annots_info,
                chunk_info.src_ids,
            ):
                file = dataset.files[sid]
                code = dataset.file2src[file]
                file = file.resolve()
                if file not in file2changes:
                    file2changes[file] = []
                if file not in file2labels:
                    file2labels[file] = {}
                file2labels[file][info.path] = label
                file2changes[file].append((not_none(info.annot_range), str(ty)))
                file2origin_code[file] = code
                repo_paths.add(dataset.file2repo[file])

        repo2files = dict[Path, list[Path]]()
        repos_to_check = list(repo_paths)
        repos_to_check.sort(key=lambda p: len(repo2files[p]), reverse=True)

        helper = _TypeCheckingHelper(use_daemon=False)
        command_name = "dmypy" if helper.use_daemon else "mypy"
        change_lists = [
            {f: file2changes[f] for f in repo2files[r]} for r in repos_to_check
        ]
        src_maps = [
            {f: file2origin_code[f] for f in repo2files[r]} for r in repos_to_check
        ]

        with self.monitor.log_task(f"Running {command_name}"):
            feedback_list = process_map(
                helper.feedback_for_repo,
                repos_to_check,
                change_lists,
                src_maps,
                desc=f"Running {command_name}",
                max_workers=max_workers,
                **tqdm_args,
            )
        file2errors = dict[Path, dict[CodePosition, str]]()
        file2contents = dict[Path, str]()
        checking_times = list[float]()
        for ls, time in feedback_list:
            checking_times.append(time)
            for k, v in ls.items():
                file2errors[k] = v[0]
                file2contents[k] = v[1]

        # compute the min, median, and max of the checking times
        mean_time = np.mean(checking_times)
        median_time = np.median(checking_times)
        max_time = max(checking_times)
        logging.info(
            f"Type checked {len(checking_times)} repos.\n"
            + f"Time stats: (median={median_time:.1f}s, mean={mean_time:.1f}s, max={max_time:.1f})s"
        )

        # todo: refactor this part out of the function
        # generate feedback-augmented inputs
        with self.monitor.log_task("Augment inputs"):
            changed_files = list(file2changes.keys())

            labels_list = [file2labels[f] for f in changed_files]
            file_errors = [file2errors.get(f, {}) for f in changed_files]
            new_contents = [file2contents[f] for f in changed_files]
            new_inputs = process_map(
                _generate_augmented_inputs,
                changed_files,
                file_errors,
                labels_list,
                new_contents,
                chunksize=max(len(file2changes) // (8 * max_workers), 1),
                max_workers=max_workers,
                desc="generating augmented inputs",
                **tqdm_args,
            )
        return dict(zip(changed_files, new_inputs))

    def type_check_preds_per_repo(
        self,
        dataset: ChunkedDataset,
        pred_types: Sequence[Sequence[PythonType]],
        tqdm_args: dict,
    ) -> dict[Path, dict]:
        """Apply the predicted types to the given files in each project first and then
        collect the type checker feedback. Will always restore the files to
        their original contents afterwards."""

        max_workers = self.args.max_workers
        file2changes = dict[Path, list[tuple[CodeRange, str]]]()
        file2labels = dict[Path, dict[AnnotPath, PythonType]]()
        repo_paths = set[Path]()

        label_info = dataset.chunks_info
        for chunk_preds, chunk_info in zip(pred_types, label_info):
            for ty, label, info, sid in zip(
                chunk_preds,
                chunk_info.types,
                chunk_info.annots_info,
                chunk_info.src_ids,
            ):
                file = dataset.files[sid]
                if file not in file2changes:
                    file2changes[file] = []
                file2changes[file].append((not_none(info.annot_range), str(ty)))
                if file not in file2labels:
                    file2labels[file] = {}
                file2labels[file][info.path] = label
                repo_paths.add(dataset.file2repo[file])
        repos_to_check = list(repo_paths)

        changed_files = list(file2changes.keys())
        origin_contents = thread_map(
            read_file,
            changed_files,
            desc="reading orginal srcs",
            max_workers=max_workers,
            **tqdm_args,
        )
        path_to_original = dict[Path, str]()
        for f, text in zip(changed_files, origin_contents):
            if MypyChecker.Preamble in text:
                raise RuntimeError(f"{f} is already modified by SPOT:\n{text}")
            path_to_original[f] = text

        try:
            # apply the file changes and get type checker feedback
            current_contents = list[str]()
            for file, changes in file2changes.items():
                start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
                # need this in case libcst does not preserve the original file content
                code_seen = dataset.file2src[file]
                changes.insert(0, (start, MypyChecker.Preamble))
                new_text = replace_strs_by_pos(
                    code_seen, [(r, 1, v) for r, v in changes]
                )
                current_contents.append(new_text)
                file.write_text(new_text)
            file2errors = dict[Path, dict[CodePosition, str]]()

            with self.monitor.log_task("Call mypy"):
                check_results: list[MypyResult | str] = thread_map(
                    MypyChecker.check_project,
                    repos_to_check,
                    max_workers=max_workers,
                    desc="calling mypy",
                    **tqdm_args,
                )
            for dir, check_r in zip(repos_to_check, check_results):
                if isinstance(check_r, str):
                    logging.warning(f"Mypy errored when checking '{dir}'.")
                    continue
                for file, errors in check_r.error_dict.items():
                    file2errors[file] = dict(errors)

            # generate feedback-augmented inputs
            file_errors = [file2errors.get(f.resolve(), {}) for f in changed_files]
            labels_list = [file2labels[f] for f in changed_files]
            new_inputs = process_map(
                _generate_augmented_inputs,
                changed_files,
                file_errors,
                labels_list,
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


# TODO: deprecate
def _generate_augmented_inputs(
    file: Path,
    file_errors: dict[CodePosition, str],
    path2types: dict[AnnotPath, PythonType],
    current_code: str,
) -> dict:
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
    current_annots, _ = collect_user_annotations(m)
    preds_map = dict[CodeRange, str]()
    annots_info = list[AnnotInfo]()
    types = list[PythonType]()
    for a in current_annots:
        if a.path in path2types:
            assert (range := a.annot_range) is not None
            assert (annot := a.annot) is not None
            preds_map[range] = m.code_for_node(annot.annotation)
            types.append(path2types[a.path])
            annots_info.append(a)
    new_code = patch_code_with_extra(current_code, preds_map, file_errors)
    code_segs = new_code.split(SpecialNames.TypeMask)
    assert len(code_segs) == len(types) + 1, f"{len(code_segs)} != {len(types)} + 1"

    return {
        "code_segs": code_segs,
        "types": types,
        "annots_info": annots_info,
    }


@enum.unique
class CheckerFeedbackKind(enum.Enum):
    per_repo = enum.auto()
    per_file = enum.auto()
    per_iso_file = enum.auto()
    per_group = enum.auto()


class _TypeCheckingHelper:
    def __init__(self, use_daemon=True, mypy_path: Optional[Path] = None):
        self.bad_repos: set = set()
        self.use_daemon: bool = use_daemon
        self.mypy_path: Optional[Path] = mypy_path

    def feedback_for_file(
        self,
        file: Path,
        cst_code: Optional[str],
        changes: list[tuple[CodeRange, str]],
        repo_checker: Path | MypyChecker,
    ) -> tuple[dict[CodePosition, str], str]:
        with open(file, "r") as f:
            current_code = f.read()
        if MypyChecker.Preamble in current_code:
            raise RuntimeError(f"{f} is already modified by SPOT.")

        start = CodeRange(CodePosition(1, 1), CodePosition(1, 1))
        replaces = list[tuple[CodeRange, int, str]]()
        replaces.append((start, 1, MypyChecker.Preamble))
        for r, v in changes:
            replaces.append((r, 1, v))

        if cst_code is None:
            cst_code = cst.parse_module(current_code).code
        new_text = replace_strs_by_pos(cst_code, replaces)
        file.write_text(new_text)

        try:
            if isinstance(repo_checker, MypyChecker):
                repo = Path(repo_checker.code_dir)
                check_r = repo_checker.recheck_project()
            else:
                repo = repo_checker
                check_r = MypyChecker.check_project(repo)
            if isinstance(check_r, str):
                if repo not in self.bad_repos:
                    logging.warning(f"Mypy errored when checking '{repo}': {check_r}")
                    self.bad_repos.add(repo)
                errors = {}
            else:
                errors = dict(check_r.error_dict.get(file.resolve(), []))
        finally:
            file.write_text(current_code)
        return errors, new_text

    def feedback_for_repo(
        self,
        repo: Path,
        changes_list: dict[Path, list[tuple[CodeRange, str]]],
        file2origin_code: dict[Path, str],
    ) -> tuple[dict[Path, tuple[dict[CodePosition, str], str]], float]:
        assert changes_list.keys() == file2origin_code.keys()
        file2errors_content = {}
        start = time.time()
        if self.use_daemon:
            with mypy_checker(repo, wait_before_check=1.0) as checker:
                for file, changes in changes_list.items():
                    code = file2origin_code[file]
                    file2errors_content[file.resolve()] = self.feedback_for_file(
                        file,
                        code,
                        changes,
                        checker,
                    )
        else:
            for file, changes in changes_list.items():
                code = file2origin_code[file]
                file2errors_content[file.resolve()] = self.feedback_for_file(
                    file,
                    code,
                    changes,
                    repo,
                )
        time_taken = time.time() - start

        return file2errors_content, time_taken
