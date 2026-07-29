"""Rank time-frequency conversions by class separability — BEFORE any model.

Idea: choose the representation that best separates N/PD/ET using model-free
separability metrics on the transformed features, then train the deep model only
on the winner. A separability pass computes each transform ONCE per recording
(not per epoch), so slow transforms like HHT are feasible here.

For each method we turn every recording's time-frequency image into a compact
per-recording vector (mean + std over time of the log-spectrogram) and score:
  * Fisher trace ratio  — between-class / within-class scatter (higher = better)
  * silhouette          — class-labelled cluster separation (higher = better)
  * subject-CV LDA F1   — light supervised proxy, GroupKFold on subject
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tremor.data import CLASS_NAMES
from tremor.datasets import TremorDataset


def _reduce(spec: np.ndarray) -> np.ndarray:
    """(F, T) time-frequency image -> mean & std over time, concatenated."""
    return np.concatenate([spec.mean(axis=1), spec.std(axis=1)])


def method_features(recs, method: str, fs: float = 100.0, f_max: float = 15.0,
                    **tfd_kwargs):
    """Reduced per-recording features for one TFD method (transform once each)."""
    target_length = int(min(r.x.shape[1] for r in recs))
    ds = TremorDataset(
        recs, target_length=target_length, fs=fs, f_max=f_max,
        tfd_method=method, normalize="per_recording", augment=False,
        oversample_to=None, length_mode="truncate", **tfd_kwargs,
    )
    X = np.stack([_reduce(ds[i][0].numpy()) for i in range(len(ds))])
    y = np.array([r.y for r in recs])
    subjects = np.array([r.subject for r in recs])
    return X, y, subjects


def patient_decomp_features(recs, method: str, fs: float = 100.0,
                            f_max: float = 15.0, **tfd_kwargs):
    """Per-PATIENT reduced TF features (mean over the patient's recordings).

    Returns (X, y, patients) aligned by sorted patient id, so two calls with
    different methods can be column-stacked / used per stage of a hybrid model.
    """
    Xr, yr, subj = method_features(recs, method, fs=fs, f_max=f_max, **tfd_kwargs)
    patients = sorted(set(subj.tolist()))
    X = np.stack([Xr[subj == p].mean(axis=0) for p in patients])
    label = {s: int(l) for s, l in zip(subj, yr)}
    y = np.array([label[p] for p in patients])
    return X, y, np.array(patients)


def fisher_trace_ratio(X: np.ndarray, y: np.ndarray) -> float:
    Xs = StandardScaler().fit_transform(X)
    overall = Xs.mean(axis=0)
    sw = sb = 0.0
    for c in np.unique(y):
        Xc = Xs[y == c]
        mu = Xc.mean(axis=0)
        sw += float(((Xc - mu) ** 2).sum())
        sb += float(len(Xc) * ((mu - overall) ** 2).sum())
    return sb / (sw + 1e-12)


def subject_cv_lda_f1(X, y, subjects, n_splits: int = 5) -> float:
    Xs = StandardScaler().fit_transform(X)
    k = int(min(n_splits, len(np.unique(subjects))))
    gkf = GroupKFold(n_splits=k)
    preds = np.empty(len(y), dtype=int)
    for tr, te in gkf.split(Xs, y, groups=subjects):
        lda = LinearDiscriminantAnalysis().fit(Xs[tr], y[tr])
        preds[te] = lda.predict(Xs[te])
    return float(f1_score(y, preds, average="macro"))


def pd_vs_et_lda_f1(X, y, subjects, n_splits: int = 5) -> float:
    """Subject-CV LDA macro-F1 on the hard PD-vs-ET axis only."""
    m = y != 0
    if m.sum() < 4 or len(np.unique(y[m])) < 2:
        return float("nan")
    return subject_cv_lda_f1(X[m], y[m], subjects[m], n_splits=n_splits)


def separability(X, y, subjects) -> dict:
    Xs = StandardScaler().fit_transform(X)
    return {
        "fisher": fisher_trace_ratio(X, y),
        "silhouette": float(silhouette_score(Xs, y)),
        "subjcv_lda_f1": subject_cv_lda_f1(X, y, subjects),
        "pdet_lda_f1": pd_vs_et_lda_f1(X, y, subjects),
        "n_features": int(X.shape[1]),
    }


def rank_methods(recs, methods, fs: float = 100.0, f_max: float = 15.0):
    """Return {method: metrics} plus prints a ranking by subject-CV LDA F1."""
    out = {}
    for m in methods:
        X, y, subj = method_features(recs, m, fs=fs, f_max=f_max)
        out[m] = separability(X, y, subj)
    return out
