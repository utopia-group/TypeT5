import ast
import difflib
import io
import logging
import math
import multiprocessing
import os
import pickle
import shutil
import time
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Generator,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    cast,
)
import warnings

import ipywidgets as widgets
import libcst as cst
import numpy as np
import pandas as pd
from IPython.display import display
from libcst.metadata import CodePosition, CodeRange

# from tqdm.auto import tqdm
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

TokenizerType = RobertaTokenizer
ModelType = T5ForConditionalGeneration

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def proj_root() -> Path:
    return Path(__file__).parent.parent.parent


def get_config_dict() -> dict:
    if (path := proj_root() / "config" / "typet5.json").exists():
        return json.loads(read_file(path))
    else:
        fefault = {
            "data_root": str(proj_root()),
            "datasets_root": str(proj_root()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fefault))
        return fefault


def get_config(key: str) -> Optional[str]:
    return get_config_dict().get(key)


def get_gpu_id(default: int) -> int:
    if (s := os.getenv("GPU_ID")) is not None:
        return int(s)
    else:
        print("GPU_ID not set, using:", default)
        return default


def get_dataroot() -> Path:
    if (v := get_config("data_root")) is None:
        return proj_root()
    else:
        return Path(v)


def get_dataset_dir(dataname: str) -> Path:
    if (v := get_config("datasets_root")) is None:
        return get_dataroot() / "datasets" / dataname
    else:
        return Path(v) / dataname


def get_model_dir(trained=True) -> Path:
    post = "trained" if trained else "training"
    return get_dataroot() / "models" / post


def get_eval_dir(dataname: str, experiment_name: str) -> Path:
    return get_dataroot() / "evaluations" / dataname / experiment_name


def mk_testset_from_repos(name="InferTypes4Py", repos: Sequence[Path] | None = None):
    if repos is None:
        repos = [proj_root()]
    dest = get_dataset_dir(name) / "repos" / "test" / "SPOT"
    if dest.exists():
        print("Deleting old dataset at: ", dest)
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for root in repos:
        root = proj_root()
        shutil.copytree(root / "src", dest / "src")
        shutil.copytree(root / "tests", dest / "tests")
        return dest


def raise_error(msg: str) -> T1:  # type: ignore
    raise RuntimeError(msg)


def load_model_spot(path) -> ModelType:
    return cast(ModelType, ModelType.from_pretrained(path))


def load_tokenizer_spot() -> TokenizerType:
    return TokenizerType.from_pretrained("Salesforce/codet5-base")


def _turn_off_tokenizer_warning(tokenizer: TokenizerType):
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True


DefaultTokenizer = load_tokenizer_spot()
_turn_off_tokenizer_warning(DefaultTokenizer)


def decode_tokens(tks, skip_special_tokens=False):
    return DefaultTokenizer.decode(tks, skip_special_tokens=skip_special_tokens)


def rec_iter_files(
    dir: Path,
    dir_filter: Callable[[Path], bool],
) -> Generator[Path, None, None]:
    """Recursively iterate over all files in a directory whose parent dirs satisfies the given
    `dir_filter`. Note that unlike `glob`, if a directory is filtered out, all
    its children won't be traversed, leading to potentially better performance in certain use cases.
    """
    assert dir.is_dir()

    def rec(path: Path) -> Generator[Path, None, None]:
        if path.is_dir():
            if not dir_filter(path):
                return
            for child in path.iterdir():
                yield from rec(child)
        else:
            yield path

    return rec(dir)


DefaultWorkers: int = multiprocessing.cpu_count() // 2


@contextmanager
def with_default_workers(workers: int):
    global DefaultWorkers
    old_workers = DefaultWorkers
    DefaultWorkers = workers
    try:
        yield
    finally:
        DefaultWorkers = old_workers


def pmap(
    f: Callable[..., T1],
    *f_args: Any,
    desc: str,
    max_workers: int | None = None,
    tqdm_args: dict = {},
) -> list[T1]:
    """
    Parallel map with progress displaying.
    """
    n = len(f_args[0])
    assert_eq(n, *(len(xs) for xs in f_args))

    if max_workers is None:
        max_workers = DefaultWorkers
    if max_workers <= 1:
        outs = list[T1]()
        for i in tqdm(range(n), desc=desc, **tqdm_args):
            outs.append(f(*(a[i] for a in f_args)))
        return outs

    chunksize = max(1, n // (50 * max_workers))
    r = process_map(
        f,
        *f_args,
        chunksize=chunksize,
        max_workers=max_workers,
        desc=desc,
        tqdm_class=tqdm,
        **tqdm_args,
    )
    assert isinstance(r, list)
    return r


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


def not_none(x: Optional[T1]) -> T1:
    assert x is not None
    return x


def as_any(x) -> Any:
    return x


def seq_flatten(xs: Iterable[Iterable[T1]]) -> Generator[T1, None, None]:
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


def groupby(iterable: Iterable[T1], keyfunc: Callable[[T1], T2]) -> dict[T2, list[T1]]:
    groups = dict[T2, list[T1]]()
    for item in iterable:
        key = keyfunc(item)
        groups.setdefault(key, []).append(item)
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
    def __init__(self, file, module_map: Callable[[str], str], **kw_args) -> None:
        self.module_map = module_map
        super().__init__(file, **kw_args)

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.module_map(module), name)


@dataclass
class TimeLogger:
    times: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def timed(self, name: str):
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

    def clear(self):
        self.times.clear()


class TaskMonitor(ABC):
    @abstractmethod
    def log_task(self, name: str):
        pass


class EmptyLoggingMonitor(TaskMonitor):
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
            with self.timer.timed(task_name):
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
    if not config_file.exists():
        warnings.warn(
            f"No pushover config file found at {config_file}. Not able to push message."
        )
    if print_to_console or not config_file.exists():
        print(f"Pushover: ({title}) {message}")
    elif notify:
        config = json.loads(config_file.read_text())
        conn.request(
            "POST",
            "/1/messages.json",
            urllib.parse.urlencode(  # type: ignore
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
    print(f"Starting task: {name}")
    try:
        start = time.time()
        yield
    except Exception as e:
        pushover_alert(f"Failed: {name}.", str(e))
        raise e
    time_taken = time.time() - start
    pushover_alert(
        f"Finished: '{name}'.",
        f"Time taken: {time_taken:.1f}s",
        notify=notify,
    )


class PickleCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def cached(self, rel_path: Path | str, func: Callable[[], T1]) -> T1:
        path = self.cache_dir / rel_path
        if not path.exists():
            value = func()
            path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"[PickleCache] Saving to cache: '{path}'")
            with path.open("wb") as f:
                pickle.dump(value, f)
            return value
        else:
            logging.info(f"[PickleCache] Loading from cache: '{path}'")
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
        else:
            logging.warning(f"[PickleCache] File not found: '{path}'")

    def clear(self):
        if self.cache_dir.exists():
            logging.info(f"Clearing cache: at: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
        else:
            logging.warning(f"No cache found at: {self.cache_dir}, skip clearing.")


def assert_eq(x: T1, *xs: T1, extra_message: Callable[[], str] = lambda: "") -> None:
    for i in range(len(xs)):
        x = xs[i - 1] if i > 0 else x
        y = xs[i]
        assert x == y, (
            f"{x} (of type {type(x).__name__}) != {y} (of type {type(y).__name__}) at equality {i}.\n"
            + extra_message()
        )


def scalar_stats(xs) -> dict[str, Any]:
    x = np.array(xs)
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "min": x.min(),
        "max": x.max(),
    }


def cumulative_counts(elems: Sequence[int]) -> tuple[list[int], list[int]]:
    counts = Counter(elems)
    keys = sorted(counts.keys())
    n = 0
    ys = []
    for k in keys:
        n += counts[k]
        ys.append(n)
    return keys, ys


def pickle_dump(file: Path, obj):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("wb") as f:
        pickle.dump(obj, f)


def _module_map(name: str):
    segs = name.split(".")
    if segs and segs[0] == "spot":
        return "typet5." + ".".join(segs[1:])
    return name


def pickle_load(file: Path):
    with file.open("rb") as f:
        return ModuleRemapUnpickler(f, _module_map).load()


def get_subset(data, key: slice | Iterable):
    if isinstance(key, slice):
        return data[key]
    else:
        return [data[i] for i in key]


def dict_subset(d: dict[T1, T2], n: int) -> dict[T1, T2]:
    keys = list(d)[:n]
    return {k: d[k] for k in keys}


def pretty_print_dict(
    d: dict,
    level: int = 0,
    max_show_level: int = 1000,
    float_precision: int = 5,
):
    for k, v in d.items():
        print("   " * level, end="")
        if isinstance(v, float):
            print(f"{k}: %.{float_precision}g" % v)
        elif isinstance(v, dict) or isinstance(v, list):
            if level >= max_show_level:
                print(f"{k}: ...")
            else:
                print(f"{k}:")
                if isinstance(v, list):
                    v = {f"[{i}]": e for i, e in enumerate(v)}
                pretty_print_dict(v, level=level + 1, max_show_level=max_show_level)
        else:
            print(f"{k}: {v}")


def pretty_show_dict(
    d: dict,
    level: int = 0,
    max_show_level: int = 1000,
    float_precision: int = 5,
) -> str:
    with redirect_stdout(io.StringIO()) as s:
        pretty_print_dict(d, level, max_show_level, float_precision)
        return s.getvalue()


def show_string_diff(str1, str2) -> str:
    diffs = difflib.unified_diff(str1.splitlines(), str2.splitlines())
    return "\n".join(diffs)


def add_line_numbers(code: str):
    lines = code.split("\n")
    ln_digits = int(math.log(len(lines), 10)) + 1
    format_s = "{ln:" + str(ln_digits) + "d}|  {line}"
    return "\n".join(format_s.format(ln=i + 1, line=l) for i, l in enumerate(lines))


class CountedAcc(NamedTuple):
    n_correct: int
    n_total: int

    @property
    def acc(self) -> float:
        return safe_div(self.n_correct, self.n_total)

    def __str__(self):
        acc = safe_div(self.n_correct, self.n_total)
        return f"{acc:.2%} (count={show_count(self.n_total)})"

    def __repr__(self):
        acc = safe_div(self.n_correct, self.n_total)
        return f"CountedAcc({acc:.2%}, count={self.n_total})"


def show_count(c: int):
    if c < 1000:
        return str(c)
    return f"{c / 1000:.1f}k"


class GroupedAccCounter(Generic[T1]):
    """
    A counter class that keeps track of the number of correct and total predictions
    for key of type `T1`.
    """

    def __init__(self) -> None:
        self.correct_counter = Counter[T1]()
        self.total_counter = Counter[T1]()

    def count(self, key: T1, n_correct: int | bool, total: int) -> None:
        self.correct_counter[key] += int(n_correct)
        self.total_counter[key] += total

    def grouped_accs(
        self, key: Callable = lambda x: x, sort_by: Callable = lambda x: x
    ) -> dict[Any, CountedAcc]:
        return {
            str(key(k)): CountedAcc(self.correct_counter[k], self.total_counter[k])
            for k in sorted(self.total_counter.keys(), key=sort_by)
        }

    def overall_acc(self) -> CountedAcc:
        return CountedAcc(
            sum(self.correct_counter.values()), sum(self.total_counter.values())
        )


def safe_div(a, b):
    if b == 0:
        return float("nan")
    return a / b


def get_modified_args(instance, flatten: bool = False) -> dict[str, Any] | None:
    """Collect only the arguments that differ from the default value, or return the value
    itself if `instance` does not contain `__annotations__`."""
    if not hasattr(instance, "__annotations__"):
        return None

    cls = type(instance)
    delta = dict[str, Any]()
    # collect all values that are different from the default
    if hasattr(cls, "_field_defaults"):
        default_values = cls._field_defaults
    else:
        default_values = {
            attr: getattr(cls, attr)
            for attr in instance.__annotations__
            if hasattr(cls, attr)
        }
    for attr in instance.__annotations__:
        v = getattr(instance, attr)
        if attr in default_values and default_values[attr] == v:
            continue
        rec_args = get_modified_args(v, flatten=flatten)
        if rec_args is None:
            delta[attr] = v
        else:
            if flatten:
                delta.update(rec_args)
            else:
                delta[attr] = rec_args
    return delta


def show_dict_as_tuple(d: dict) -> str:
    elems = dict[str, str]()
    for k, v in d.items():
        if isinstance(v, dict):
            v = show_dict_as_tuple(v)
        elems[k] = str(v)
    return "(" + ", ".join(f"{k}={v}" for k, v in elems.items()) + ")"


def repr_modified_args(instance, flatten: bool = False) -> str:
    ma = get_modified_args(instance, flatten=flatten)
    type_name = type(instance).__name__
    return type_name + show_dict_as_tuple(ma) if ma else type_name + "()"


def merge_dicts(dicts: Sequence[dict[T1, Any]]) -> dict[T1, list]:
    assert len(dicts) > 0
    keys = dicts[0].keys()
    result = {k: [] for k in keys}
    for d in dicts:
        assert_eq(keys, d.keys())
        for k in keys:
            result[k].append(d[k])
    return result


def get_single(xs: Sequence[T1]) -> T1:
    assert len(xs) == 1
    return xs[0]


def get_unique_ids(xs: Sequence[T1]) -> list[int]:
    """Get the indices of the unique elements in xs while preserving the order."""
    seen = set()
    ids = []
    for i, x in enumerate(xs):
        if x not in seen:
            seen.add(x)
            ids.append(i)
    return ids


def print_limited(s: str, max_lines: int = 50):
    lines = s.split("\n")
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    return print("\n".join(lines))


@dataclass
class MovingAvg:
    """
    When `alpha > 0`, applies exponential moving average, otherwise applies simple moving average.
    """

    alpha: float
    value: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        if self.count == 0:
            self.value = value
        elif self.alpha > 0:
            self.value = (1 - self.alpha) * self.value + self.alpha * value
        else:
            self.value = (self.value * self.count + value) / (self.count + 1)
        self.count += 1

    def __repr__(self) -> str:
        return f"(value={self.value:.4f}, count={self.count})"


import asyncio


async def throttled_async_run(f, xs: Sequence, concurrency: int):
    """Run `f` on `xs` asynchronously, but limit the max number of concurrent tasks to `concurrency`."""
    sem = asyncio.Semaphore(concurrency)

    async def task(x):
        async with sem:
            return await f(x)

    tasks = [task(x) for x in xs]
    return await asyncio.gather(*tasks)


def move_all_files(src_dir: Path, dest_dir: Path, glob_pattern: str = "**/*"):
    for f in src_dir.glob(glob_pattern):
        target = dest_dir / f.relative_to(src_dir)
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(f, target)


# used for showing cst nodes.
_EmptyModule = cst.Module([])


def show_expr(expr: cst.CSTNode, quoted: bool = True) -> str:
    s = _EmptyModule.code_for_node(expr)
    if quoted:
        s = f"cst'{s}'"
    return s
