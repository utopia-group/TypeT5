import asyncio
from typing import *
from spot.data import (
    CtxArgs,
    SrcDataset,
    TypeCheckingEnv,
    chunk_from_src,
    src_preds_to_accuracies,
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
    concurrency: int = 12


@dataclass
class DAggerTrainingState:
    args: DAggerArgs
    optimizer: torch.optim.Optimizer
    grad_counter: int = 0


@dataclass
class DAggerRunResult:
    type_assignment: dict[int, PythonType] = field(default_factory=dict)
    src_seq: list[TokenizedSrc] = field(default_factory=list)
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

    async def run_on_src(
        self,
        src: TokenizedSrc,
        typecheck_env: TypeCheckingEnv,
        model_executor: ThreadPoolExecutor,
        cpu_executor: ProcessPoolExecutor,
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
        eloop = asyncio.get_event_loop()

        mr.args.do_sample = True
        assignment = dict[int, PythonType]()
        result = DAggerRunResult(assignment)
        for t, label in enumerate(src.types):
            batch = await eloop.run_in_executor(
                cpu_executor, src_to_batch, new_src, t, ctx_args
            )
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["labels"] = batch["labels"].to(device)
            use_expert = random.random() < expert_rate
            result.used_expert.append(use_expert)
            if use_expert:
                assignment[t] = label
            else:
                with t_logger.timed("predict next type"):
                    preds, _ = await eloop.run_in_executor(
                        model_executor, mr.predict_on_batch, batch
                    )
                assignment[t] = preds[0][0]
            if state is not None:
                loss = await eloop.run_in_executor(
                    model_executor, self.maybe_graident_step, batch, state
                )
                result.loss_seq.append(loss)
            with t_logger.timed("type checking"):
                repo_root = typecheck_env.template_root / src.repo
                check_r = await eloop.run_in_executor(
                    cpu_executor,
                    type_check_src_in_project,
                    src,
                    assignment,
                    repo_root,
                    typecheck_env.pre_fdbks[src.file],
                )
            with t_logger.timed("generate new src"):
                new_src = await eloop.run_in_executor(
                    cpu_executor, get_typechecked_src, src, assignment, check_r
                )
                result.src_seq.append(new_src)
            callback(new_src)
        return result

    def maybe_graident_step(self, batch, state: DAggerTrainingState) -> float:
        grad_accum_steps = state.args.grad_accum_steps
        t_logger = self.t_logger
        mr = self.wrapper
        with t_logger.timed("compute gradients"):
            with torch.autocast("cuda"):
                outputs = mr.model.forward(
                    input_ids=batch["input_ids"], labels=batch["labels"]
                )
            assert isinstance(outputs, Seq2SeqLMOutput)
            loss = not_none(outputs.loss)
            (loss / grad_accum_steps).backward()
            state.grad_counter += 1
        if state.grad_counter == grad_accum_steps:
            with t_logger.timed("update parameters"):
                torch.nn.utils.clip_grad_norm_(mr.model.parameters(), 1.0)
                state.optimizer.step()
                mr.model.zero_grad()
                state.grad_counter = 0
        return loss.item()

    async def train_on_data(
        self,
        src_datasets: dict[str, SrcDataset],
        dagger_args: DAggerArgs,
        log_fn: Callable[[int, dict], None],
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
                total=sum(len(s.types) for s in train_srcs),
                desc="train_on_data",
                smoothing=0.0,
            ) as pbar, ThreadPoolExecutor(1) as model_executor, ProcessPoolExecutor(
                DefaultWorkers
            ) as cpu_executor:

                async def train_step(src):
                    progress = pbar.n / not_none(pbar.total)
                    assert 0 <= progress <= 1.0
                    r = await self.run_on_src(
                        src,
                        env,
                        model_executor,
                        cpu_executor,
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
                    step = train_loss.count
                    log_fn(
                        step,
                        {
                            "train/acc": train_acc.value,
                            "train/loss": train_loss.value,
                        },
                    )

                await throttled_async_run(
                    train_step, train_srcs, dagger_args.concurrency
                )

    async def eval_on_data(
        self,
        dataset: SrcDataset,
        concurrency: int = 10,
    ):
        result = DAggerEvalResult([], [])

        with dataset.setup_typechecking(dataset.all_srcs) as env, tqdm(
            total=sum(len(s.types) for s in dataset.all_srcs),
            desc="eval_on_data",
            smoothing=0.0,
        ) as pbar, ThreadPoolExecutor(1) as model_executor, ProcessPoolExecutor(
            DefaultWorkers
        ) as cpu_executor:

            async def eval_step(src: TokenizedSrc):
                r = await self.run_on_src(
                    src,
                    env,
                    model_executor,
                    cpu_executor,
                    state=None,
                    callback=lambda _: pbar.update(),
                    expert_rate=0.0,
                )
                result.final_srcs.append(r.src_seq[-1])
                preds = [r.type_assignment[i] for i in range(len(src.types))]
                result.final_preds.append(preds)

            await throttled_async_run(eval_step, dataset.all_srcs, concurrency)

        return result


def src_to_batch(src: TokenizedSrc, t: int, ctx_args: CtxArgs):
    chunk, info = chunk_from_src(src, 0, t, ctx_args)
    assert_eq(chunk["n_labels"], 1)
    batch = {
        "input_ids": torch.tensor([chunk["input_ids"]]),
        "labels": torch.tensor([chunk["labels"]]),
        "n_labels": [1],
    }
    return batch


@dataclass
class DAggerEvalResult:
    final_srcs: list[TokenizedSrc]
    final_preds: list[list[PythonType]]

    @property
    def accuracies(self):
        return src_preds_to_accuracies(self.final_preds, self.final_srcs)


async def throttled_async_run(f, xs: Sequence, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def task(x):
        async with sem:
            return await f(x)

    tasks = [task(x) for x in xs]
    return await asyncio.gather(*tasks)


def get_typechecked_src(src: TokenizedSrc, assignment, check_r) -> TokenizedSrc:
    errors, current_code = check_r
    errors = [] if isinstance(errors, str) else errors
    new_src = feedbacks_to_tokenized_src(
        src,
        current_code,
        errors,
        patch_predictions=False,
    )
    new_src.prev_types = assignment
    new_src = new_src.inline_prev_predictions(as_comment=False)
    return new_src
