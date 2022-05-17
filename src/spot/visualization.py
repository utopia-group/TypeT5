import html
import re
from typing import Sequence

import ipywidgets as widgets

from spot.data import ChunkedDataset, CtxArgs, PythonType
from spot.utils import TokenizerSPOT


def visualize_batch(
    dataset: ChunkedDataset,
    i: int,
    preds: list[list[PythonType]],
    tokenizer: TokenizerSPOT,
    ctx_args: CtxArgs,
) -> str:
    pred_types = preds[i]
    typpes_enc = [
        tokenizer.encode(str(t), add_special_tokens=False) for t in pred_types
    ]

    label_types = dataset.chunks_info[i].types
    code_tks = inline_predictions(dataset.data["input_ids"][i], typpes_enc, tokenizer)
    sep_1 = tokenizer.encode(
        "\n---------⬆context⬆---------\n", add_special_tokens=False
    )
    sep_2 = tokenizer.encode(
        "\n---------⬇context⬇---------\n", add_special_tokens=False
    )
    ctx_margin = ctx_args.ctx_margin
    code_tks = (
        code_tks[:ctx_margin]
        + sep_1
        + code_tks[ctx_margin:-ctx_margin]
        + sep_2
        + code_tks[-ctx_margin:]
    )
    code_dec = tokenizer.decode(code_tks, skip_special_tokens=False)
    code_dec = code_inline_extra_ids(code_dec, label_types)
    src_ids = sorted(list(set(dataset.chunks_info[i].src_ids)))
    files = [dataset.files[i] for i in src_ids]
    return "".join(
        [
            "labels: ",
            str(label_types),
            "\n",
            "preds: ",
            str(pred_types),
            "\n",
            "files: ",
            str(files),
            "\n",
            "========================== Code =======================\n",
            code_dec,
            "\n",
        ]
    )


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


def display_code_sequence(texts: Sequence[str], titles=None):
    if titles is None:
        titles = range(len(texts))

    def code_to_html(code):
        return colorize_code_html(html.escape(code))

    outputs = [
        widgets.HTML(
            value="<pre style='line-height: 1.2; padding: 10px; color: rgb(212,212,212); background-color: rgb(30,30,30); }'>"
            + code_to_html(s)
            + "</pre>"
        )
        for s in texts
    ]

    tab = widgets.Tab(outputs)
    for i, t in enumerate(titles):
        tab.set_title(i, str(t))
    return tab


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
