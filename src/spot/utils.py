from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Sequence,
    Optional,
    TypeVar,
    Union,
    Generator,
)
from typing import cast
import libcst as cst
import ast
import os
from pathlib import Path
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.metrics import confusion_matrix
import numpy as np


class SpecialNames:
    Return = "<return>"
    Missing = "<missing>"
    Lambda = "<lambda>"
    Empty = "<empty>"


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


def seq_flatten(xs: Sequence[Sequence[T1]]) -> Generator[T1, None, None]:
    return (item for sublist in xs for item in sublist)


def join_str(segs: Sequence[str], seps: Sequence[str]) -> str:
    assert len(seps) == len(segs) - 1, f"{len(seps)} != {len(segs) - 1}"
    all_segs = [segs[0]]
    for s, sep in zip(segs[1:], seps):
        all_segs.append(sep)
        all_segs.append(s)
    return "".join(all_segs)


def accuracy_by_labels(y_preds: Sequence[T1], y_true: Sequence[T1], top_k: Optional[int]=None):
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
