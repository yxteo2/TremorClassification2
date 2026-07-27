"""Subject-clustered uncertainty for pooled out-of-fold predictions.

The recordings in this dataset are *clustered by subject* (each patient
contributes several trials), and the whole point of the LOSO harness is that a
subject appears in exactly one test fold. Any confidence interval or
significance test must therefore resample / permute at the **subject** level,
not the recording level — treating the ~270 recordings as independent would
report an interval several times too narrow, because trials of one subject are
near-duplicates.

Two tools are provided, both operating on pooled out-of-fold predictions:

* :func:`bootstrap_subject_ci` — cluster (block) bootstrap: resample subjects
  with replacement, gather all of each drawn subject's recordings, recompute
  the metric. Gives a 95% CI on macro-F1 and each per-class F1.
* :func:`permutation_test` — permute the label assigned to each subject
  (holding predictions fixed) to get a null distribution for macro-F1, hence a
  p-value that the predictions carry real subject-label information.

Everything here is torch-free and depends only on numpy / scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    ci: float  # e.g. 0.95

    def __repr__(self) -> str:  # pragma: no cover - display only
        pct = int(round(self.ci * 100))
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] ({pct}% CI)"


def _subject_labels(subjects: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_subjects, one_label_per_subject).

    Each subject is assumed to carry a single class label across all of its
    recordings; this is enforced by the data layout (class is part of the
    filename). If a subject somehow spans labels we take its majority label and
    do not fail, so the tests stay usable on messy inputs.
    """
    uniq = np.unique(subjects)
    labels = np.empty(len(uniq), dtype=y_true.dtype)
    for i, s in enumerate(uniq):
        vals = y_true[subjects == s]
        # majority label (ties -> smallest label); robust to accidental mixing
        vals_u, counts = np.unique(vals, return_counts=True)
        labels[i] = vals_u[np.argmax(counts)]
    return uniq, labels


def _f1_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """[macro_f1, f1_class0, f1_class1, ...] for a single resample."""
    labels = list(range(n_classes))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return np.concatenate([[per_class.mean()], per_class])


def bootstrap_subject_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    class_names: tuple[str, ...],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Cluster-bootstrap CI for macro-F1 and each per-class F1.

    Parameters
    ----------
    y_true, y_pred : (N,) int arrays of pooled out-of-fold labels/predictions.
    subjects       : (N,) subject id per recording (the cluster key).
    class_names    : names in label order, e.g. ("N", "PD", "ET").
    n_boot         : number of subject-resamples.
    ci             : central interval mass (0.95 -> 2.5/97.5 percentiles).

    Returns a dict keyed "macro_f1" and each class name.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subjects = np.asarray(subjects)
    n_classes = len(class_names)

    uniq = np.unique(subjects)
    # Precompute recording indices per subject so each resample is cheap.
    idx_by_subject = {s: np.flatnonzero(subjects == s) for s in uniq}

    point = _f1_metrics(y_true, y_pred, n_classes)

    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n_classes + 1), dtype=float)
    n_subj = len(uniq)
    for b in range(n_boot):
        drawn = uniq[rng.integers(0, n_subj, size=n_subj)]
        sel = np.concatenate([idx_by_subject[s] for s in drawn])
        boot[b] = _f1_metrics(y_true[sel], y_pred[sel], n_classes)

    alpha = (1.0 - ci) / 2.0
    lo = np.percentile(boot, 100 * alpha, axis=0)
    hi = np.percentile(boot, 100 * (1.0 - alpha), axis=0)

    names = ["macro_f1", *class_names]
    return {
        name: Estimate(point=float(point[i]), lo=float(lo[i]), hi=float(hi[i]), ci=ci)
        for i, name in enumerate(names)
    }


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    class_names: tuple[str, ...],
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Subject-level label-permutation p-value for macro-F1.

    The null hypothesis is that predictions are unrelated to the true labels.
    We permute the *per-subject* label (so the class-count structure is
    preserved and no subject is split), broadcast it back to that subject's
    recordings, and recompute macro-F1 with the observed predictions held
    fixed. The p-value is the fraction of permutations whose macro-F1 reaches
    the observed value (with the customary +1 smoothing).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    subjects = np.asarray(subjects)
    n_classes = len(class_names)

    uniq, subj_labels = _subject_labels(subjects, y_true)
    # Map each recording to its subject's row in `uniq` for fast broadcast.
    subj_to_row = {s: i for i, s in enumerate(uniq)}
    row_of = np.array([subj_to_row[s] for s in subjects])

    observed = float(_f1_metrics(y_true, y_pred, n_classes)[0])

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(subj_labels)
        y_perm = perm[row_of]
        stat = float(_f1_metrics(y_perm, y_pred, n_classes)[0])
        if stat >= observed:
            ge += 1

    p_value = (ge + 1) / (n_perm + 1)
    return {"observed_macro_f1": observed, "p_value": p_value, "n_perm": n_perm}
