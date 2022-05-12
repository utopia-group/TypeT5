import html
from typing import Sequence

import ipywidgets as widgets


def display_code_sequence(texts: Sequence[str], titles=None):
    if titles is None:
        titles = range(len(texts))
    outputs = [
        widgets.HTML(value=f"<pre style='line-height:1.2;'>{html.escape(s)}</pre>")
        for s in texts
    ]

    tab = widgets.Tab(outputs)
    for i, t in enumerate(titles):
        tab.set_title(i, str(t))
    return tab
