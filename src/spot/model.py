from collections import Counter

import numpy as np
from transformers import RobertaTokenizer
from transformers.pipelines import Pipeline

from spot.data import mask_type_annots, output_ids_as_types, tokenize_masked
from spot.type_env import AnnotCat, normalize_type
from spot.utils import *


def compute_metrics(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    types_cat: list[list[AnnotCat]],
    tokenizer: RobertaTokenizer,
    conf_top_k=10,
):
    # apply the tokenizer decoder to each rows
    assert len(predictions.shape) == 2
    assert (n_rows := predictions.shape[0]) == label_ids.shape[0]
    n_preds = 0
    n_correct_by_cat = Counter[AnnotCat]()
    n_partial_by_cat = Counter[AnnotCat]()
    n_label_by_cat = Counter[AnnotCat]()
    pred_types, label_types = [], []
    for i in tqdm(range(n_rows), desc="decoding types"):
        pred = output_ids_as_types(predictions[i, :], tokenizer)
        label = output_ids_as_types(label_ids[i, :], tokenizer)
        cats = types_cat[i]
        n_label_by_cat.update(cats)
        n_preds += len(pred)
        for (p, l, cat) in zip(pred, label, cats):
            p = normalize_type(p)
            l = normalize_type(l)
            pred_types.append(p.head_name())
            label_types.append(l.head_name())
            if p == l:
                n_correct_by_cat[cat] += 1
            if p.head_name() == l.head_name():
                n_partial_by_cat[cat] += 1

    accuracy_partial = {"total": n_partial_by_cat.total() / n_label_by_cat.total()}
    for k in n_partial_by_cat.keys():
        accuracy_partial[k.name] = n_partial_by_cat[k] / n_label_by_cat[k]

    accuracy_full = {"total": n_correct_by_cat.total() / n_label_by_cat.total()}
    for k in n_correct_by_cat.keys():
        accuracy_full[k.name] = n_correct_by_cat[k] / n_label_by_cat[k]

    return {
        "accuracy_partial": accuracy_partial,
        "accuracy_full": accuracy_full,
        "n_labels": n_label_by_cat.total(),
        "n_predictions": n_preds,
        "label_types": label_types,
        "pred_types": pred_types,
    }
