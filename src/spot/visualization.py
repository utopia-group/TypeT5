import html
import re
from typing import Sequence

import ipywidgets as widgets


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
    for i in range(len(code)):
        c = code[i]
        prev = code[i - 1] if i > 0 else None
        next = code[i + 1] if i < len(code) - 1 else None
        if c == "/" and next == "*":
            output.append("<span style='color: rgb(106, 153, 85)'>")
        output.append(c)
        if prev == "*" and c == "/":
            output.append("</span>")
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
