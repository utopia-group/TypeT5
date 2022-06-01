import html
import re
from typing import Sequence

import colored
import ipywidgets as widgets

from spot.data import ChunkedDataset, CtxArgs, PythonType
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
        out.layout.height = height
    box_layout = widgets.Layout(overflow="scroll")
    return widgets.VBox(
        [
            widgets.HBox([slider, slider_label]),
            widgets.Box((out,), layout=box_layout),
        ]
    )


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
