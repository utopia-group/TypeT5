import ast
import logging
import os
import pickle
import shutil
import time
from abc import ABC, abstractmethod
from asyncio import current_task
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import libcst as cst
import numpy as np
import pandas as pd
from libcst.metadata import CodePosition, CodeRange
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
from transformers import RobertaTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

TokenizerSPOT = RobertaTokenizer
ModelSPOT = T5ForConditionalGeneration


class SpecialNames:
    Return = "<return>"
    Missing = "<missing>"
    Lambda = "<lambda>"
    Empty = "<empty>"
    TypeMask = "SPOT_TYPE_MASK"


def read_file(path) -> str:
    """read file content as string."""
    with open(path, "r") as f:
        return f.read()


def write_file(path, content: str) -> None:
    """write content to file."""
    with open(path, "w") as f:
        f.write(content)


def proj_root() -> Path:
    return Path(__file__).parent.parent.parent


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def not_none(x: Optional[T1]) -> T1:
    assert x is not None
    return x


def seq_flatten(xs: Sequence[Sequence[T1]]) -> Generator[T1, None, None]:
    return (item for sublist in xs for item in sublist)


def join_str(segs: Sequence[str], seps: Sequence[str]) -> str:
    assert len(seps) == len(segs) - 1, f"{len(seps)} != {len(segs) - 1}"
    all_segs = [segs[0]]
    for s, sep in zip(segs[1:], seps):
        all_segs.append(sep)
        all_segs.append(s)
    return "".join(all_segs)


def accuracy_by_labels(
    y_preds: Sequence[T1], y_true: Sequence[T1], top_k: Optional[int] = None
):
    assert len(y_preds) == len(y_true)
    label_counts = Counter(y_true).most_common(top_k)
    label_set = set(l[0] for l in label_counts)
    correct_counts = Counter[T1]()
    for p, l in zip(y_preds, y_true):
        if p == l and l in label_set:
            correct_counts[l] += 1
    return {l: correct_counts[l] / total for l, total in label_counts}


def confusion_matrix_top_k(y_preds, y_true, k):
    labels_counts = Counter(y_true).most_common(k)
    labels = [l[0] for l in labels_counts]
    counts = [l[1] for l in labels_counts]
    cm = confusion_matrix(y_true, y_preds, labels=labels, normalize=None)
    cm = cm / np.array([counts]).T
    return {"labels": labels, "matrix": cm}


def groupby(iterable: Iterable[T1], keyfunc: Callable[[T1], T2]) -> dict[T2, list[T1]]:
    groups = dict[T2, list[T1]]()
    for item in iterable:
        key = keyfunc(item)
        if key not in groups:
            groups[key] = list[T1]()
        groups[key].append(item)
    return groups


def grouped(xs: Sequence[T1], chunk_size: int) -> Iterable[Sequence[T1]]:
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


def issorted(xs: Iterable) -> bool:
    current = None
    for x in xs:
        if current is not None and x < current:
            return False
        current = x
    return True


def replace_strs_by_pos(original: str, replaces: Sequence[tuple[CodeRange, int, str]]):
    """Replace the parts specificed by `replaces` with the given strings.
    Each entry of `replaces` is a tuple of (code_range, priority, new_str)."""

    def as_tuple(p: CodePosition):
        return (p.line, p.column)

    lines = original.split("\n")
    out_segs = list[str]()
    ptr = CodePosition(1, 1)

    def advance_to(target: CodePosition, output: bool):
        nonlocal ptr
        if as_tuple(target) <= as_tuple(ptr):
            return
        assert ptr.line <= target.line, f"ptr: {ptr}, target: {target}"
        if output:
            while ptr.line < target.line:
                out_segs.append(lines[ptr.line - 1][ptr.column - 1 :])
                out_segs.append("\n")
                ptr = CodePosition(ptr.line + 1, 1)
            assert ptr.line == target.line, f"ptr: {ptr}, target: {target}"
            out_segs.append(lines[ptr.line - 1][ptr.column - 1 : target.column - 1])
        ptr = target

    replaces_sorted = sorted(replaces, key=lambda x: (as_tuple(x[0].start), x[1]))
    # for (r1, t1), (r2, t2) in zip(replaces_sorted, replaces_sorted[1:]):
    #     assert as_tuple(r1.end) <= as_tuple(
    #         r2.start
    #     ), f"overlapping ranges:\n   {r1}: {t1}\n   {r2}: {t2}"

    while bool(replaces_sorted):
        r, _, rtext = replaces_sorted.pop(0)
        try:
            advance_to(r.start, True)
        except IndexError:
            raise IndexError(
                f"{r.start} is out of range. Trying to replace with text <<{rtext}>>. Original str:\n<<{original}>>"
            )
        advance_to(r.end, False)
        out_segs.append(rtext)
    last_pos = CodePosition(len(lines), len(lines[-1]) + 1)
    advance_to(last_pos, True)

    return "".join(out_segs)


@contextmanager
def restore_later(file_path: Path):
    """Record the orginal file content and always restore it later."""
    backup = file_path.read_text()
    try:
        yield file_path
    finally:
        file_path.write_text(backup)


class ModuleRemapUnpickler(pickle.Unpickler):
    def __init__(self, file, module_map, **kw_args) -> None:
        self.module_map: Callable[[str], str] = module_map
        super().__init__(file, **kw_args)

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.module_map(module), name)


@dataclass
class TimeLogger:
    times: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def log_time(self, name: str):
        start = time.time()
        yield
        end = time.time()
        if name not in self.times:
            self.times[name] = list[float]()
        self.times[name].append(end - start)

    def as_dataframe(self):
        names = list(self.times.keys())
        total_times = [sum(ts) for ts in self.times.values()]
        counts = [len(ts) for ts in self.times.values()]
        avg_times = [sum(ts) / len(ts) for ts in self.times.values()]

        df = pd.DataFrame(
            {
                "name": names,
                "count": counts,
                "avg_time": avg_times,
                "total_time": total_times,
            }
        )
        df.sort_values(by="total_time", ascending=False, inplace=True)
        return df

    def total_times(self) -> dict[str, float]:
        return {name: sum(ts) for name, ts in self.times.items()}


class TaskMonitor(ABC):
    @abstractmethod
    def log_task(self, name: str):
        pass


@dataclass
class TaskLoggingMonitor(TaskMonitor):
    monitor_name: str
    current_task: list[str] = field(default_factory=list)
    timer: TimeLogger = field(default_factory=TimeLogger)

    @contextmanager
    def log_task(self, name: str):
        self.current_task.append(name)
        task_name = " > ".join(self.current_task)
        # if self.current_tqdm is not None:
        #     self.current_tqdm.set_postfix_str(f"Current task: {task_name}")
        print(f"[{self.monitor_name}] Starting task: '{task_name}'")
        try:
            start = time.time()
            with self.timer.log_time(task_name):
                yield
            end = time.time()
            print(
                f"[{self.monitor_name}] '{task_name}' finished in {end - start} seconds"
            )
        finally:
            self.current_task.pop()
            # if self.current_tqdm is not None:
            #     task_name = " > ".join(self.current_task)
            #     self.current_tqdm.set_postfix_str(f"Current task: {task_name}")


import http.client
import json
import urllib


def pushover_alert(
    title: str, message: str, print_to_console: bool = True, notify: bool = True
) -> None:
    "If notify=False, will only print to console."

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    config_file = proj_root() / "config/pushover.json"
    if print_to_console:
        print(f"Pushover: ({title}) {message}")
    if not config_file.exists():
        print(
            f"No pushover config file found at {config_file}. Not able to push message."
        )
    elif notify:
        config = json.loads(config_file.read_text())
        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(
                {
                    "token": config["token"],
                    "user": config["user"],
                    "title": title,
                    "message": message,
                }
            ),
            {"Content-type": "application/x-www-form-urlencoded"},
        )
        # check if the request was successful
        conn.getresponse()


@contextmanager
def run_long_task(name: str, notify: bool = True):
    "When notify=False, will only push notifiations when encountering errors."

    try:
        start = time.time()
        yield
    except Exception as e:
        pushover_alert(f"Failed: {name}.", str(e))
        raise e
    time_taken = time.time() - start
    pushover_alert(
        f"Finished: {name}.",
        f"Time taken: {time_taken:.1f}s",
        notify=notify,
    )


class PickleCache(Generic[T1]):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def cached(self, rel_path: Path | str, func: Callable[[], T1]) -> T1:
        path = self.cache_dir / rel_path
        if not path.exists():
            value = func()
            path.parent.mkdir(parents=True, exist_ok=True)
            logging.info("Saving to cache: %s", path)
            with path.open("wb") as f:
                pickle.dump(value, f)
            return value
        else:
            logging.info("Loading from cache: %s", path)
            with path.open("rb") as f:
                return pickle.load(f)

    def set(self, rel_path: Path | str, value: T1):
        path = self.cache_dir / rel_path
        with path.open("wb") as f:
            pickle.dump(value, f)

    def remove(self, rel_path: Path | str):
        path = self.cache_dir / rel_path
        if path.exists():
            path.unlink()

    def clear(self):
        if self.cache_dir.exists():
            logging.info(f"Clearing cache: at: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
        else:
            logging.warning(f"No cache found at: {self.cache_dir}")


def assert_eq(left: T1, right: T1) -> None:
    if left != right:
        raise AssertionError(f"{left} != {right}")
