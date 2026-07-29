"""Leave-one-patient-out evaluation with subject-level bootstrap CIs.

Every patient is predicted once by a model trained on all others (honest,
subject-level, no leakage), predictions are pooled, and scored once with the
same subject bootstrap CI + permutation test used everywhere else in the repo.
"""

from __future__ import annotations

import numpy as np

from tremor.data import CLASS_NAMES
from tremor.evaluate import classification_report
from tremor.stats import bootstrap_subject_ci, permutation_test


def loso_predict(model_factory, X, y):
    """Leave-one-patient-out predictions. model_factory() -> fresh model."""
    y = np.asarray(y)
    preds = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        model = model_factory().fit(X[tr], y[tr])
        preds[i] = int(model.predict(X[i:i + 1])[0])
    return preds


def evaluate(model_factory, X, y, subjects, n_boot: int = 2000,
             n_perm: int = 2000, seed: int = 0) -> dict:
    y = np.asarray(y)
    y_pred = loso_predict(model_factory, X, y)

    onehot = np.eye(len(CLASS_NAMES))[y_pred]
    rep = classification_report(np.log(onehot + 1e-6), y, CLASS_NAMES)
    ci = bootstrap_subject_ci(y, y_pred, np.asarray(subjects), CLASS_NAMES,
                              n_boot=n_boot, seed=seed)
    perm = permutation_test(y, y_pred, np.asarray(subjects), CLASS_NAMES,
                            n_perm=n_perm, seed=seed)

    # Sub-axis accuracies from the pooled predictions.
    tremor_true = y != 0
    n_vs_tremor = float(((y_pred != 0) == tremor_true).mean())
    pdet_mask = (y != 0) & (y_pred != 0)       # true tremor predicted tremor
    pd_vs_et = (float((y_pred[pdet_mask] == y[pdet_mask]).mean())
                if pdet_mask.any() else float("nan"))

    return {
        "macro_f1": rep["macro_f1"],
        "accuracy": rep["accuracy"],
        "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
        "confusion_matrix": rep["confusion_matrix"],
        "ci": {k: {"point": e.point, "lo": e.lo, "hi": e.hi} for k, e in ci.items()},
        "permutation_p": perm["p_value"],
        "n_vs_tremor_acc": n_vs_tremor,
        "pd_vs_et_acc": pd_vs_et,
        "y_true": y.tolist(), "y_pred": y_pred.tolist(),
    }


def evaluate_hybrid(model_factory, X1, X2, y, subjects, n_boot: int = 2000,
                    n_perm: int = 2000, seed: int = 0) -> dict:
    """Leave-one-patient-out for a HybridTwoStage using two feature matrices
    (X1 for stage 1, X2 for stage 2), aligned row-for-row."""
    y = np.asarray(y)
    preds = np.empty(len(y), dtype=int)
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        model = model_factory().fit(X1[tr], X2[tr], y[tr])
        preds[i] = int(model.predict(X1[i:i + 1], X2[i:i + 1])[0])

    onehot = np.eye(len(CLASS_NAMES))[preds]
    rep = classification_report(np.log(onehot + 1e-6), y, CLASS_NAMES)
    ci = bootstrap_subject_ci(y, preds, np.asarray(subjects), CLASS_NAMES,
                              n_boot=n_boot, seed=seed)
    perm = permutation_test(y, preds, np.asarray(subjects), CLASS_NAMES,
                            n_perm=n_perm, seed=seed)
    tremor_true = y != 0
    pdet_mask = (y != 0) & (preds != 0)
    return {
        "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
        "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
        "confusion_matrix": rep["confusion_matrix"],
        "ci": {k: {"point": e.point, "lo": e.lo, "hi": e.hi} for k, e in ci.items()},
        "permutation_p": perm["p_value"],
        "n_vs_tremor_acc": float(((preds != 0) == tremor_true).mean()),
        "pd_vs_et_acc": (float((preds[pdet_mask] == y[pdet_mask]).mean())
                         if pdet_mask.any() else float("nan")),
    }


def print_result(tag: str, res: dict) -> None:
    c = res["ci"]
    print(f"\n=== {tag} ===")
    print(f"  macro-F1 {res['macro_f1']:.3f}  acc {res['accuracy']:.3f}  "
          f"permutation p={res['permutation_p']:.4f}")
    for cls in CLASS_NAMES:
        e = c[cls]
        print(f"  {cls:>2} F1 {res['per_class_f1'][cls]:.3f}  "
              f"[{e['lo']:.3f}, {e['hi']:.3f}]")
    print(f"  N-vs-tremor acc {res['n_vs_tremor_acc']:.3f} | "
          f"PD-vs-ET acc {res['pd_vs_et_acc']:.3f}")
    print(f"  confusion (rows=true N/PD/ET): {res['confusion_matrix']}")
