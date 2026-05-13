"""Metrics: per-class F1, sensitivity, specificity, AUC, confusion matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _specificity(cm: np.ndarray, cls: int) -> float:
    tn = cm.sum() - cm[cls].sum() - cm[:, cls].sum() + cm[cls, cls]
    fp = cm[:, cls].sum() - cm[cls, cls]
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def classification_report(
    logits: np.ndarray,
    y_true: np.ndarray,
    class_names: tuple[str, ...],
) -> dict:
    labels = list(range(len(class_names)))
    probs = softmax(logits)
    y_pred = probs.argmax(axis=-1)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    sens = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    spec = [_specificity(cm, i) for i in labels]

    auc: dict[str, float | None] = {}
    for i, name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() in (0, len(y_bin)):
            auc[name] = None
        else:
            auc[name] = float(roc_auc_score(y_bin, probs[:, i]))

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "per_class": {
            name: {
                "f1": f1[i],
                "sensitivity": sens[i],
                "specificity": spec[i],
                "auc": auc[name],
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
