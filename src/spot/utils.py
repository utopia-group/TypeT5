import ast
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import libcst as cst
import numpy as np
from libcst.metadata import CodePosition, CodeRange
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


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


def issorted(xs: Iterable) -> bool:
    current = None
    for x in xs:
        if current is not None and x < current:
            return False
        current = x
    return True


def replace_strs_by_pos(original: str, replaces: Sequence[tuple[CodeRange, str]]):
    def as_tuple(p: CodePosition):
        return (p.line, p.column)

    lines = original.split("\n")
    out_segs = list[str]()
    ptr = CodePosition(1, 1)

    def advance_to(target: CodePosition, output: bool):
        nonlocal ptr
        assert ptr.line <= target.line, f"ptr: {ptr}, target: {target}"
        if output:
            while ptr.line < target.line:
                out_segs.append(lines[ptr.line - 1][ptr.column - 1 :])
                out_segs.append("\n")
                ptr = CodePosition(ptr.line + 1, 1)
            assert ptr.line == target.line, f"ptr: {ptr}, target: {target}"
            out_segs.append(lines[ptr.line - 1][ptr.column - 1 : target.column - 1])
        ptr = target

    replaces_sorted = sorted(
        replaces, key=lambda x: (as_tuple(x[0].start), as_tuple(x[0].end))
    )
    for (r1, t1), (r2, t2) in zip(replaces_sorted, replaces_sorted[1:]):
        assert as_tuple(r1.end) <= as_tuple(
            r2.start
        ), f"overlapping ranges:\n   {r1}: {t1}\n   {r2}: {t2}"

    while bool(replaces_sorted):
        r, rtext = replaces_sorted.pop(0)
        advance_to(r.start, True)
        advance_to(r.end, False)
        out_segs.append(rtext)
    last_pos = CodePosition(len(lines), len(lines[-1]))
    advance_to(last_pos, True)

    return "".join(out_segs)


def patch_code_with_extra(
    code: str, predictions: dict[CodeRange, str], errors: dict[CodePosition, str]
) -> str:
    replaces = []
    extra_id = 0
    for r, t in predictions.items():
        replaces.append((CodeRange(r.start, r.start), f"/* {t} */"))
        replaces.append((r, f"<extra_id_{extra_id}>"))
        extra_id += 1
    for p, e in errors.items():
        replaces.append((CodeRange(p, p), f"/* error: {e} */"))
    return replace_strs_by_pos(code, replaces)
