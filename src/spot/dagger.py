from typing import *
from spot.data import (
    SrcDataset,
    TypeCheckingEnv,
    chunk_from_src,
    type_check_src_in_project,
)
from spot.model import ModelWrapper
from spot.tokenized_src import TokenizedSrc, feedbacks_to_tokenized_src
from spot.train import _configure_optimizers
from spot.type_check import MypyFeedback, PythonType, TypeCheckArgs, normalize_type
from spot.utils import *
from transformers.modeling_outputs import Seq2SeqLMOutput

import torch
import copy
import random


class DAggerArgs(NamedTuple):
    grad_accum_steps: int = 32


@dataclass
class DAggerTrainingState:
    args: DAggerArgs
    optimizer: torch.optim.Optimizer
    grad_counter: int = 0


@dataclass
class DAggerRunResult:
    type_assignment: dict[int, PythonType] = field(default_factory=dict)
    input_seq: list[str] = field(default_factory=list)
    errors_seq: list[list[MypyFeedback]] = field(default_factory=list)
    loss_seq: list[float] = field(default_factory=list)
    used_expert: list[bool] = field(default_factory=list)


@dataclass
class RunningAvg:
    value: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.value = (self.value * self.count + value * count) / (self.count + count)
        self.count += count

    def __repr__(self) -> str:
        return f"(value={self.value:.4f}, count={self.count})"


@dataclass
class DAggerModel:
    wrapper: ModelWrapper
    t_logger: TimeLogger = field(default_factory=TimeLogger)

    def run_on_src(
        self,
        src: TokenizedSrc,
        typecheck_env: TypeCheckingEnv,
        state: DAggerTrainingState | None = None,
        callback: Callable[[TokenizedSrc], None] = lambda _: None,
        expert_rate: float = 0.0,
    ) -> DAggerRunResult:
        # Start with a file without any type assignments, predict one
        # type at a time using nucleus sampling.
        mr = self.wrapper
        ctx_args = mr.args.ctx_args
        device = mr.model.device
        t_logger = self.t_logger
        new_src = src

        mr.args.do_sample = True
        assignment = dict[int, PythonType]()
        result = DAggerRunResult(assignment)
        for t, label in enumerate(src.types):
            chunk, info = chunk_from_src(new_src, 0, t, ctx_args)
            assert_eq(chunk["n_labels"], 1)
            batch = {
                "input_ids": torch.tensor([chunk["input_ids"]], device=device),
                "labels": torch.tensor([chunk["labels"]], device=device),
                "n_labels": [1],
            }
            result.input_seq.append(decode_tokens(chunk["input_ids"]))
            use_expert = random.random() < expert_rate
            result.used_expert.append(use_expert)
            if use_expert:
                assignment[t] = label
            else:
                with t_logger.timed("predict next type"):
                    with torch.autocast("cuda"):
                        preds, _ = mr.predict_on_batch(batch)
                assignment[t] = preds[0][0]
            if state is not None:
                grad_accum_steps = state.args.grad_accum_steps
                with t_logger.timed("compute gradients"):
                    with torch.autocast("cuda"):
                        outputs = mr.model.forward(
                            input_ids=batch["input_ids"], labels=batch["labels"]
                        )
                    assert isinstance(outputs, Seq2SeqLMOutput)
                    loss = not_none(outputs.loss)
                    result.loss_seq.append(loss.item())
                    (loss / grad_accum_steps).backward()
                    state.grad_counter += 1
                if state.grad_counter == grad_accum_steps:
                    with t_logger.timed("update parameters"):
                        torch.nn.utils.clip_grad_norm_(mr.model.parameters(), 1.0)
                        state.optimizer.step()
                        mr.model.zero_grad()
                        state.grad_counter = 0
            with t_logger.timed("type checking"):
                repo_root = typecheck_env.template_root / src.repo
                check_r = type_check_src_in_project(
                    src, assignment, repo_root, typecheck_env.pre_fdbks[src.file]
                )
            with t_logger.timed("generate new src"):
                errors, current_code = check_r
                errors = [] if isinstance(errors, str) else errors
                result.errors_seq.append(errors)
                new_src = feedbacks_to_tokenized_src(
                    src,
                    current_code,
                    errors,
                    patch_predictions=False,
                )
                new_src.prev_types = assignment
                new_src = new_src.inline_prev_predictions(as_comment=False)
            callback(new_src)
        return result

    def train_on_data(
        self,
        src_datasets: dict[str, SrcDataset],
        dagger_args: DAggerArgs,
    ):
        mr = self.wrapper

        train_set = src_datasets["train"]
        dev_set = src_datasets["valid"]
        all_srcs = train_set.all_srcs + dev_set.all_srcs
        mix_set = SrcDataset(train_set.repos_root, all_srcs)

        train_acc = RunningAvg()
        train_loss = RunningAvg()

        with mix_set.setup_typechecking(all_srcs) as env:
            # training loop
            optimizer = _configure_optimizers(mr.model)[0][0]
            state = DAggerTrainingState(dagger_args, optimizer)

            train_srcs = copy.copy(train_set.all_srcs)
            random.shuffle(train_srcs)
            with tqdm(
                total=sum(len(s.types) for s in train_srcs), desc="Training"
            ) as pbar:
                for src in train_srcs:
                    progress = pbar.n / not_none(pbar.total)
                    assert 0 <= progress <= 1.0
                    r = self.run_on_src(
                        src,
                        env,
                        state,
                        callback=lambda _: pbar.update(),
                        expert_rate=1 - progress,
                    )
                    preds = r.type_assignment
                    for i in range(len(src.types)):
                        if r.used_expert[i]:
                            continue
                        is_correct = normalize_type(src.types[i]) == normalize_type(
                            preds[i]
                        )
                        train_acc.update(int(is_correct))
                    for l in r.loss_seq:
                        train_loss.update(l)

    def eval_on_data(
        self,
        dataset: SrcDataset,
    ):
        correct_seq = []
        loss_seq = []

        with dataset.setup_typechecking(dataset.all_srcs) as env:
            with tqdm(
                total=sum(len(s.types) for s in dataset.all_srcs), desc="Evaluating"
            ) as pbar:
                for src in dataset.all_srcs:
                    r = self.run_on_src(
                        src,
                        env,
                        None,
                        callback=lambda _: pbar.update(),
                        expert_rate=0.0,
                    )
                    preds = r.type_assignment
                    for i in range(len(src.types)):
                        is_correct = normalize_type(src.types[i]) == normalize_type(
                            preds[i]
                        )
                        correct_seq.append(int(is_correct))
                    loss_seq.extend(r.loss_seq)

        return {"loss": float(np.mean(loss_seq)), "acc": float(np.mean(correct_seq))}
