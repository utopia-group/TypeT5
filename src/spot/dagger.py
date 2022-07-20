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
from collections import deque as Deque

import torch
import random
import threading


class DAggerArgs(NamedTuple):
    save_dir: Path
    grad_accum_steps: int = 32
    concurrency: int = 12
    replay_buffer_size: int = concurrency * 100
    saves_per_epoch: int = 5


@dataclass
class DAggerTrainingState:
    args: DAggerArgs
    optimizer: torch.optim.Optimizer
    prog_bar: tqdm
    log_fn: Callable[[int, dict], None]
    save_every: int
    avg_loss: MovingAvg
    replay_buffer: Deque[dict] = field(default_factory=Deque)
    grad_counter: int = 0
    save_counter: int = 0


@dataclass
class DAggerRunResult:
    type_assignment: dict[int, PythonType] = field(default_factory=dict)
    batch_seq: list[dict] = field(default_factory=list)
    src_seq: list[TokenizedSrc] = field(default_factory=list)
    used_expert: dict[int, bool] = field(default_factory=dict)


class CostModel:
    SAMPLE = 1
    TRAIN = 1


@dataclass
class DAggerModel:
    wrapper: ModelWrapper
    use_type_checker: bool = True
    rollout_length: int = 100
    t_logger: TimeLogger = field(default_factory=TimeLogger)

    async def rollout_on_src(
        self,
        src: TokenizedSrc,
        labels: Sequence[int],
        typecheck_env: TypeCheckingEnv | None,
        model_executor: ThreadPoolExecutor,
        cpu_executor: ProcessPoolExecutor,
        batch_callback: Callable[[dict], Coroutine] | None = None,
        expert_rate: float = 0.0,
    ) -> DAggerRunResult:
        """
        Run the DAgger model on the given source file, predicting one type at a time.
        """

        mr = self.wrapper
        ctx_args = mr.args.ctx_args
        t_logger = self.t_logger
        eloop = asyncio.get_event_loop()

        new_src = src
        result = DAggerRunResult()
        assignment = result.type_assignment

        for t in labels:
            use_expert = random.random() < expert_rate
            result.used_expert[t] = use_expert

            batch = await eloop.run_in_executor(
                cpu_executor, src_to_batch, new_src, t, ctx_args
            )
            result.batch_seq.append(batch)
            if use_expert:
                assignment[t] = src.types[t]
            else:
                with t_logger.timed("predict next type"):
                    preds, _ = await eloop.run_in_executor(
                        model_executor, mr.predict_on_batch, batch
                    )
                assignment[t] = preds[0][0]

            if batch_callback is not None:
                # e.g., perform training here
                cb_future = batch_callback(batch)

            if typecheck_env is not None:
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
            else:
                # skip type checking
                new_src = await eloop.run_in_executor(
                    cpu_executor,
                    TokenizedSrc.inline_predictions,
                    src,
                    False,
                    assignment,
                )
            result.src_seq.append(new_src)

            if batch_callback is not None:
                await cb_future  # type: ignore

        return result

    async def train_on_data(
        self,
        train_set: SrcDataset,
        dagger_args: DAggerArgs,
        log_fn: Callable[[int, dict], None],
    ):
        eloop = asyncio.get_event_loop()
        mr = self.wrapper
        optimizer = _configure_optimizers(mr.model)[0][0]
        n_labels = sum(len(s.types) for s in train_set.all_srcs)
        src_range_list = _to_src_range_list(train_set.all_srcs, self.rollout_length)
        random.shuffle(src_range_list)
        alpha = min(1.0, 100 / (1 + n_labels))
        avg_acc = MovingAvg(alpha=alpha)
        avg_loss = MovingAvg(alpha=alpha)
        log_lock = threading.Lock()

        def log_fn_locked(t, d):
            with log_lock:
                log_fn(t, d)

        with train_set.setup_typechecking(train_set.all_srcs) as env, tqdm(
            total=(CostModel.TRAIN + CostModel.SAMPLE) * n_labels,
            desc="train_on_data",
            smoothing=0.01,
        ) as pbar, ThreadPoolExecutor(1) as model_executor, ProcessPoolExecutor(
            DefaultWorkers
        ) as cpu_executor:
            save_every = n_labels // dagger_args.saves_per_epoch
            state = DAggerTrainingState(
                dagger_args, optimizer, pbar, log_fn_locked, save_every, avg_loss
            )
            labels_counter = 0

            async def batch_callback(batch: dict):
                state.prog_bar.update(CostModel.SAMPLE)

            async def train_step(src_labels: tuple[TokenizedSrc, Sequence[int]]):
                src, labels = src_labels
                # progress = pbar.n / not_none(pbar.total)
                # assert 0 <= progress <= 1.0
                r = await self.rollout_on_src(
                    src,
                    labels,
                    env if self.use_type_checker else None,
                    model_executor,
                    cpu_executor,
                    batch_callback=batch_callback,
                    # expert_rate=1 - progress,
                )
                preds = r.type_assignment
                assert_eq(len(preds), len(labels))
                for t in labels:
                    if r.used_expert[t]:
                        continue
                    norm_pred = normalize_type(preds[t])
                    norm_label = normalize_type(src.types[t])
                    avg_acc.update(int(norm_pred == norm_label))
                nonlocal labels_counter
                labels_counter += len(labels)
                state.log_fn(
                    labels_counter,
                    {"train/acc": avg_acc.value},
                )

                # train on the batches
                for batch in r.batch_seq:
                    await eloop.run_in_executor(
                        model_executor,
                        self._process_batch,
                        batch,
                        state,
                    )

            await throttled_async_run(
                train_step, src_range_list, dagger_args.concurrency
            )
            # train on the remaining batches
            await eloop.run_in_executor(
                model_executor,
                self._empty_buffer,
                state,
            )

    async def eval_on_data(
        self,
        dataset: SrcDataset,
        concurrency: int = DefaultWorkers,
    ):
        result = DAggerEvalResult([], [])

        with dataset.setup_typechecking(dataset.all_srcs) as env, tqdm(
            total=sum(len(s.types) for s in dataset.all_srcs),
            desc="eval_on_data",
            smoothing=0.01,
        ) as pbar, ThreadPoolExecutor(1) as model_executor, ProcessPoolExecutor(
            concurrency
        ) as cpu_executor:

            async def batch_callback(batch):
                pbar.update()

            async def eval_step(src_labels):
                src, labels = src_labels
                r = await self.rollout_on_src(
                    src,
                    labels,
                    env if self.use_type_checker else None,
                    model_executor,
                    cpu_executor,
                    batch_callback=batch_callback,
                    expert_rate=0.0,
                )
                result.final_srcs.append(r.src_seq[-1])
                result.final_preds.append(r.type_assignment)

            src_range_list = _to_src_range_list(dataset.all_srcs, self.rollout_length)
            await throttled_async_run(eval_step, src_range_list, concurrency)

        return result

    def _process_batch(self, batch: dict, state: DAggerTrainingState):
        """
        Add the new batch to the replay buffer and potentially train the model
        by comsuming the buffer.
        Should be called from the model thread to avoid race conditoin.
        """
        buffer_size = state.args.replay_buffer_size
        state.replay_buffer.appendleft(batch)
        if len(state.replay_buffer) > buffer_size:
            assert_eq(len(state.replay_buffer), buffer_size + 1)
            batch = state.replay_buffer.pop()
            self._train_on_batch(batch, state)

    def _empty_buffer(self, state: DAggerTrainingState):
        """
        Empty the replay buffer.
        Should be called from the model thread
        """
        while state.replay_buffer:
            batch = state.replay_buffer.pop()
            self._train_on_batch(batch, state)
        if state.grad_counter > 0:
            self._update_model(state)

    def _train_on_batch(
        self,
        batch: dict,
        state: DAggerTrainingState,
    ):
        t_logger = self.t_logger
        mr = self.wrapper
        accum_steps = state.args.grad_accum_steps
        device = mr.model.device

        with t_logger.timed("compute gradients"):
            with torch.autocast("cuda"):
                outputs = mr.model.forward(
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
            assert isinstance(outputs, Seq2SeqLMOutput)
            loss = not_none(outputs.loss)
            state.avg_loss.update(loss.item())
            (loss / accum_steps).backward()
            state.prog_bar.update(CostModel.TRAIN)
            state.grad_counter += 1

        if state.grad_counter >= accum_steps:
            self._update_model(state)

    def _update_model(self, state: DAggerTrainingState):
        with self.t_logger.timed("update parameters"):
            torch.nn.utils.clip_grad_norm_(self.wrapper.model.parameters(), 1.0)
            state.optimizer.step()
            self.wrapper.model.zero_grad()

        step = state.avg_loss.count
        state.log_fn(
            step,
            {
                "train/loss": state.avg_loss.value,
                "train/replay_buffer": len(state.replay_buffer),
            },
        )

        state.save_counter += state.grad_counter
        if state.save_counter >= state.save_every:
            self.wrapper.save_pretrained(state.args.save_dir / f"step={step}")
            state.save_counter -= state.save_every

        state.grad_counter = 0


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
    final_preds: list[dict[int, PythonType]]

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


def _get_label_ranges(n: int, max_labels: int):
    t = 0
    out = list[range]()
    while n > 0:
        delta = min(n, max_labels)
        out.append(range(t, t + delta))
        t += delta
        n -= delta
    return out


def _to_src_range_list(srcs: Sequence[TokenizedSrc], max_labels: int):
    src_range_list = list[tuple[TokenizedSrc, range]]()
    for s in srcs:
        ranges = _get_label_ranges(len(s.types), max_labels)
        for r in ranges:
            src_range_list.append((s, r))
    return src_range_list
