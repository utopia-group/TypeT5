import html
import re
from typing import Sequence

import colored
import ipywidgets as widgets
import plotly.express as px

from spot.data import (
    ChunkedDataset,
    CountedAcc,
    CtxArgs,
    PythonType,
    SrcDataset,
    code_to_check_from_preds,
)
from spot.model import DatasetPredResult, DecodingArgs
from spot.type_check import normalize_type
from spot.utils import *


def visualize_chunks(chunks: ChunkedDataset, height="500px") -> widgets.VBox:
    def show(i):
        d = chunks.data[i]
        print("Labels:", chunks.tokenizer.decode(d["labels"]))
        print("============== code =================")
        print(chunks.tokenizer.decode(d["input_ids"]))

    return interactive_sized(show, {"i": (0, len(chunks.data) - 1)}, height=height)


def visualize_preds_on_code(
    dataset: ChunkedDataset,
    preds: list[list[PythonType]],
) -> widgets.VBox:
    def show_batch(i: int):
        pred_types = preds[i]
        tokenizer = dataset.tokenizer
        typpes_enc = [
            tokenizer.encode(str(t), add_special_tokens=False) for t in pred_types
        ]

        label_types = dataset.chunks_info[i].types
        code_tks = inline_predictions(
            dataset.data["input_ids"][i], typpes_enc, tokenizer
        )
        code_dec = tokenizer.decode(code_tks, skip_special_tokens=False)
        code_dec = code_inline_extra_ids(code_dec, label_types)
        src_ids = sorted(list(set(dataset.chunks_info[i].src_ids)))
        files = [dataset.files[i] for i in src_ids]
        meta = "".join(
            [
                "labels: ",
                str(label_types),
                "\n",
                "preds: ",
                str(pred_types),
                "\n",
                "files: ",
                str(files),
            ]
        )
        display(in_scroll_pane(meta, height=None))

        code = widgets.HTML(
            "<pre style='line-height: 1.2; padding: 10px; color: rgb(212,212,212); background-color: rgb(30,30,30); }'>"
            + colorize_code_html(html.escape(code_dec))
            + "</pre>"
        )
        display(in_scroll_pane(code))

    slider = widgets.IntSlider(
        0, min=0, max=len(dataset.data) - 1, continuous_update=False
    )
    return widgets.interactive(show_batch, i=slider)


def inline_predictions(
    input_tks: Sequence[int],
    predictions: Sequence[Sequence[int]],
    tokenizer: TokenizerSPOT,
) -> list[int]:
    """Inline the model predictions into the input code and then decode"""
    out_tks = list[int]()
    extra_id = 0
    next_special = tokenizer.additional_special_tokens_ids[99 - extra_id]
    for tk in input_tks:
        out_tks.append(tk)
        if tk == next_special:
            out_tks.extend(predictions[extra_id])
            extra_id += 1
            next_special = tokenizer.additional_special_tokens_ids[99 - extra_id]
    assert extra_id == len(predictions), f"{extra_id} != {len(predictions)}"
    return out_tks


def visualize_sequence(
    contents: Sequence[str | widgets.Widget], height: Optional[str] = "500px"
) -> widgets.VBox:
    assert len(contents) > 0

    slider = widgets.IntSlider(min=0, max=len(contents) - 1, value=0)
    slider_label = widgets.Label(value=f"({len(contents)} total)")

    def select(i: int):
        el = contents[i]
        if isinstance(el, str):
            print(el)
        else:
            display(el)

    out = widgets.interactive_output(select, {"i": slider})
    if height is not None:
        out.layout.height = height  # type: ignore
    box_layout = widgets.Layout(overflow="scroll")
    return widgets.VBox(
        [
            widgets.HBox([slider, slider_label]),
            widgets.Box((out,), layout=box_layout),
        ]
    )


def visualize_sequence_tabs(
    contents: Sequence[str | widgets.Widget],
    height: Optional[str] = None,
    titles: Sequence[str] | None = None,
    selected: int | None = None,
) -> widgets.VBox:
    assert len(contents) > 0

    children = list[widgets.Widget]()
    for el in contents:
        if isinstance(el, str):
            el = string_widget(el)
        children.append(el)

    out = widgets.Tab(children=children)
    for i in range(len(children)):
        title = titles[i] if titles is not None else str(i)
        out.set_title(i, title)
    if height is not None:
        out.layout.height = height  # type: ignore
    box_layout = widgets.Layout(overflow="scroll")

    if selected is None:
        selected = len(contents) - 1
    out.selected_index = selected

    return widgets.VBox((out,), layout=box_layout)


def in_scroll_pane(
    content: widgets.Widget | str, height: Optional[str] = "500px"
) -> widgets.Box:
    if isinstance(content, str):
        with (out := widgets.Output()):
            print(content)
        content = out
    box_layout = widgets.Layout(overflow="scroll", height=height)
    return widgets.Box([content], layout=box_layout)


def interactive_sized(
    f,
    kwargs: dict,
    height: Optional[str] = "500px",
) -> widgets.VBox:
    out = widgets.interactive(f, **kwargs)
    panel = out.children[-1]
    return widgets.VBox(
        [
            *out.children[:-1],
            in_scroll_pane(panel, height=height),
        ]
    )


def visualize_dicts(dicts: Sequence[dict], titles: Sequence[str] | None = None):
    def show_dict_with_change(d: dict, prev: Optional[dict]):
        result = dict()
        for k in d:
            v = d[k]
            v0 = prev.get(k, None) if prev is not None else None
            match v, v0:
                case (CountedAcc() as v, CountedAcc() as v0):
                    result[k] = f"{str(v)} [{v.acc - v0.acc:+.2%}]"
                case (CountedAcc(), _):
                    result[k] = f"{str(v)}"
                case (float(), float()) | (int(), int()):
                    result[k] = f"{v:.4g} [{v - v0:.4g}]"
                case (float(), _) | (int(), _):
                    result[k] = f"{v:.4g}"
                case (dict(), dict() | None):
                    result[k] = show_dict_with_change(v, v0)
                case _:
                    result[k] = str(v)
        return result

    def display_acc(round):
        d = dicts[round]
        prev = None if round == 0 else dicts[round - 1]
        return pretty_display_dict(show_dict_with_change(d, prev))

    tabs = [display_acc(i) for i in range(len(dicts))]
    if titles is None:
        titles = [f"R{i}" for i in range(len(dicts))]
    return visualize_sequence_tabs(tabs, titles=titles)


def visualize_conf_matrix(results: dict[str, DatasetPredResult], top_k: int = 15):
    def show_conf(name, top_k):
        pred_r = results[name]
        labels = [
            normalize_type(t).head_name()
            for info in pred_r.chunks.chunks_info
            for t in info.types
        ]
        all_preds = [
            normalize_type(t).head_name() for t in seq_flatten(pred_r.predictions)
        ]
        unique_types = len(set(labels))
        top_k = min(top_k, unique_types)
        m = confusion_matrix_top_k(all_preds, labels, top_k)
        display_conf_matrix(m)

    tabs = []
    for name in results:
        with (out := widgets.Output()):
            show_conf(name, top_k)
        tabs.append(out)

    return visualize_sequence_tabs(tabs, titles=list(results.keys()))


def colorize_code_html(code: str) -> str:
    "Highligh the special comments in the type checker-augmented python code."
    output = list[str]()
    in_comment = False
    for i in range(len(code)):
        c = code[i]
        prev = code[i - 1] if i > 0 else None
        next = code[i + 1] if i < len(code) - 1 else None
        if not in_comment and c == "/" and next == "*":
            output.append("<span style='color: rgb(106, 153, 85)'>")
            in_comment = True
        output.append(c)
        if in_comment and prev == "*" and c == "/":
            output.append("</span>")
            in_comment = False
    new_code = "".join(output)

    def replace(m: re.Match[str]):
        ml = re.match(r"&lt;label;([^;]+);label&gt;", m[0])
        assert ml is not None
        l = ml.group(1)
        return f"<span style='color: rgb(78, 201, 176)'>({l})</span>"

    return re.sub(r"(&lt;label;[^;]+;label&gt;)", replace, new_code)


def code_inline_extra_ids(code: str, preds: list):
    def replace(m: re.Match[str]):
        mi = re.match(r"<extra_id_(\d+)>", m[0])
        assert mi is not None
        id = int(mi.group(1))
        label = str(preds[id])
        return f"<label;{label};label>"

    return re.sub(r"(<extra_id_\d+>)", replace, code)


def code_inline_type_masks(code: str, preds: list, label_color: Optional[str] = None):
    i = 0
    if label_color is not None:
        color_mark = colored.fg(label_color)
        reset_mark = colored.attr("reset")

    def replace(m: re.Match[str]):
        nonlocal i
        l = str(preds[i])
        i += 1
        if label_color is not None:
            l = color_mark + l + reset_mark
        return l

    return re.sub(SpecialNames.TypeMask, replace, code)


def string_widget(s: str):
    return widgets.HTML(string_to_html(s))


def string_to_html(s: str) -> str:
    return f"<div style='white-space: pre-wrap; line-height: 1.2; font-family: monospace, monospace;'>{s}</div>"


def pretty_display_dict(d: dict, float_precision: int = 5):
    outputs = list[widgets.Widget]()
    for expand in [False, True]:
        max_level = 1000 if expand else 0
        d_s = pretty_show_dict(
            d, float_precision=float_precision, max_show_level=max_level
        )
        o = widgets.HTML(string_to_html(d_s))
        outputs.append(o)

    tab = widgets.Tab()
    tab.children = outputs
    tab.set_title(0, "Compressed")
    tab.set_title(1, "Expanded")
    return tab


def visualize_counts(
    values: Counter[str] | dict[str, Counter[str]],
    x_name: str,
    top_k: int | list[str] = 15,
    title: str | None = None,
):
    if isinstance(values, Counter):
        values = {"Source": values}
    y_names = list(values.keys())
    counters = list(values.values())
    if isinstance(top_k, int):
        top_keys = [k for k, _ in counters[0].most_common(top_k)]
    else:
        top_keys = top_k
    data = list[dict]()
    for s in y_names:
        c = values[s]
        for k in top_keys:
            data.append({x_name: k, "Count": c.get(k, 0), "Source": s})
    df = pd.DataFrame(data)
    if title is None:
        title = f"{x_name} distribution"
    return px.bar(df, x=x_name, y="Count", color="Source", title=title)


import plotly.express as px

from spot.type_env import MypyFeedback
from spot.utils import groupby, pretty_print_dict


def plot_feedback_distribution(
    feedbacks: Iterable[MypyFeedback],
):
    error_code_counter = Counter[str]()
    for fb in feedbacks:
        error_code_counter[fb.error_code] += 1
    top_feedbacks = dict(error_code_counter.most_common(10))
    df = pd.DataFrame(error_code_counter.most_common(), columns=["error_code", "count"])
    display(px.bar(df, x="error_code", y="count", title="Error code frequencies"))
    return top_feedbacks


def show_feedback_stats(dataset: SrcDataset):
    fb_list: list[list[MypyFeedback]] = dataset.extra_stats["mypy_feedbacks"]
    stats = {}
    for k in ["feedbacks_per_file", "type_check_success_ratio"]:
        stats[k] = dataset.extra_stats[k]
    stats["total_feedbacks"] = sum(len(l) for l in fb_list)
    num_labels = sum(len(s.types) for s in dataset.all_srcs)
    stats["feedbacks_per_label"] = stats["total_feedbacks"] / num_labels
    stats["top_feedbacks"] = plot_feedback_distribution(seq_flatten(fb_list))
    pretty_print_dict(stats)
    fdbk_srcs = [(f, src) for src, fs in zip(dataset.all_srcs, fb_list) for f in fs]
    error_groups = groupby(fdbk_srcs, lambda x: x[0].error_code)
    return error_groups


def visualize_feedbacks_in_srcs(
    dataset: SrcDataset,
):
    error_groups = show_feedback_stats(dataset)
    fdbks = list(seq_flatten(list(error_groups.values())))
    n_total = len(fdbks)

    def viz(i):
        fdbk, src = fdbks[i]
        code = code_inline_type_masks(src.origin_code, src.types)
        text = (
            f"feedback: {fdbk}\n" + "=========code=========\n" + add_line_numbers(code)
        )
        display(string_widget(text))

    return interactive_sized(viz, {"i": (0, n_total - 1)})
