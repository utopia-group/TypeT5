from collections import Counter

import numpy as np
from transformers.pipelines import Pipeline

from spot.data import mask_type_annots, output_ids_as_types, tokenize_masked
from spot.type_env import normalize_type
from spot.utils import *


def compute_metrics(
    predictions: np.ndarray, label_ids: np.ndarray, tokenizer, conf_top_k=10
):
    # apply the tokenizer decoder to each rows
    assert len(predictions.shape) == 2
    assert (n_rows := predictions.shape[0]) == label_ids.shape[0]
    n_labels = 0
    n_preds = 0
    n_correct_partial = 0
    n_correct_full = 0
    pred_types, label_types = [], []
    for i in tqdm(range(n_rows), desc="decoding types"):
        pred = output_ids_as_types(predictions[i, :], tokenizer)
        label = output_ids_as_types(label_ids[i, :], tokenizer)
        n_labels += len(label)
        n_preds += len(pred)
        for (p, l) in zip(pred, label):
            p = normalize_type(p)
            l = normalize_type(l)
            pred_types.append(p.head_name())
            label_types.append(l.head_name())
            if p == l:
                n_correct_full += 1
            if p.head_name() == l.head_name():
                n_correct_partial += 1

    return {
        "accuracy_partial": n_correct_partial / n_labels,
        "accuracy_full": n_correct_full / n_labels,
        "n_labels": n_labels,
        "n_predictions": n_preds,
        "label_types": label_types,
        "pred_types": pred_types,
    }
