from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import (
    Callable,
    ItemsView,
    List,
    Sequence,
    Tuple,
    Dict,
    Set,
    Optional,
    TypeVar,
    Union,
    Generator,
)
from typing import cast
import libcst as cst
import os
from pathlib import Path
from tqdm import tqdm


class SpecialNames:
    Return = "<return>"
    Missing = "<missing>"


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


def parallel_map_unordered(
    f: Callable[[T1], T2],
    items: Sequence[T1],
    executor: ThreadPoolExecutor | ProcessPoolExecutor,
    log_progress=True,
) -> List[T2]:
    """Apply f to each item in parallel. Note that the order of the results is not guaranteed."""
    fs = [executor.submit(f, rep) for rep in items]
    if log_progress:
        return [f.result() for f in tqdm(as_completed(fs), total=len(fs))]
    else:
        return [f.result() for f in as_completed(fs)]


def seq_flatten(xs: Sequence[Sequence[T1]]) -> Generator[T1, None, None]:
    return (item for sublist in xs for item in sublist)