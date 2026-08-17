"""Per-patient spectral tables and the bilateral asymmetry features.

These build the (patients, features) matrices every model consumes.
"""

from __future__ import annotations

import numpy as np

from common.training import loso_nn
from models.architectures import SpectrumBiLSTM

#: Names of the six left-right asymmetry descriptors, in column order.
ASYM_NAMES = ("corr", "cos", "peak_df", "log_peak_ratio",
              "log_power_ratio", "l1")


# --------------------------------------------------------------------------- #
# Runner -- so results are reproducible without a scratch script
# --------------------------------------------------------------------------- #
def spectrum_table(recs, ch=slice(3, 6), fs=100.0, f_lo=3.0, f_hi=15.0,
                   nperseg=512):
    """(patients, F) normalised power spectrum, axes averaged.

    Averaging the axes is a rotation-invariant reduction, not a free loss --
    keeping them separate (``AxisFusionNet``) measured worse at this n.
    """
    from collections import defaultdict
    from scipy.signal import welch
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
        P = P.mean(0)
        k = (f >= f_lo) & (f <= f_hi)
        s = P[k]
        rows[r.subject].append(s / (s.sum() + 1e-20))
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[p], 0) for p in pats])),
            np.array([lab[p] for p in pats]), np.array(pats))
def best_model(n_bins, num_classes=2):
    """The configuration that measured best: frequency BiLSTM, hidden 32.

    bal-acc 0.913 / recall 1.000 with class weighting, or bal-acc 0.790 /
    AUC 0.942 / precision 0.667 without. 9,090 parameters -- the capacity
    optimum; h=128 and the 11-86 M pretrained backbones are both far past it.
    """
    return SpectrumBiLSTM(n_bins, num_classes, hidden=32)
def evaluate(recs, axis="PD_vs_ET", class_weight=False, model_fn=None,
             seeds=(0, 1, 2), epochs=150, ch=slice(3, 6)):
    """End-to-end: recordings -> spectra -> LOSO -> metrics dict."""
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)
    X, y3, g = spectrum_table(recs, ch=ch)
    if axis == "PD_vs_ET":
        m = y3 != 0
        X, y, g = X[m], (y3[m] == 2).astype(int), g[m]
    else:
        y = (y3 != 0).astype(int)
    fn = model_fn or (lambda: best_model(X.shape[1]))
    prob = loso_nn(X, y, g, fn, seeds=seeds, epochs=epochs,
                   class_weight=class_weight)
    pred = (prob >= 0.5).astype(int)
    bal = 0.5 * (recall_score(y, pred, pos_label=1, zero_division=0)
                 + recall_score(y, pred, pos_label=0, zero_division=0))
    return {"bal_acc": bal, "auc": roc_auc_score(y, prob),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "n": len(y), "n_pos": int(y.sum()), "prob": prob, "y": y,
            "patients": g}
def bilateral_table(recs, side_of, ch=slice(3, 6), fs=100.0, f_lo=3.0,
                    f_hi=15.0, nperseg=512):
    """(patients, 2*F) left|right spectra, for :class:`BilateralAttention`.

    ``side_of(rec) -> "left" | "right" | None``. A patient missing one limb is
    dropped rather than zero-filled: a zero spectrum is not "no tremor", it is
    an out-of-distribution input, and with 6 ET subjects one such row moves the
    metric.
    """
    from collections import defaultdict
    from scipy.signal import welch
    rows, lab = defaultdict(lambda: {"left": [], "right": []}), {}
    for r in recs:
        s = side_of(r)
        if s is None:
            continue
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
        P = P.mean(0)
        k = (f >= f_lo) & (f <= f_hi)
        v = P[k]
        rows[r.subject][s].append(v / (v.sum() + 1e-20))
        lab[r.subject] = r.y
    pats = [p for p in sorted(rows) if rows[p]["left"] and rows[p]["right"]]
    X = np.array([np.concatenate([np.mean(rows[p]["left"], 0),
                                  np.mean(rows[p]["right"], 0)]) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))
def asym_feats(Xb):
    """Explicit left-right interaction from a ``bilateral_table`` matrix.

    This is the hand-coded counterpart to what :class:`BilateralAttention`
    would have to learn: six numbers describing how the two limbs' spectra
    differ, rather than a 2F-token attention map over them.

    The clinical premise is real -- PD signs begin unilaterally and stay more
    severe on that side, while ET is typically more symmetric -- and it is
    cheap to state directly:

    ``corr``            correlation of the two mean-centred spectral shapes
    ``cos``             cosine similarity of the raw shapes
    ``peak_df``         |peak-bin difference| between limbs
    ``log_peak_ratio``  log ratio of peak heights
    ``log_power_ratio`` log ratio of total in-band power
    ``l1``              L1 distance between the two shapes

    Takes ``(patients, 2F)`` as produced by :func:`bilateral_table`; returns
    ``(patients, 6)``.
    """
    f = Xb.shape[1] // 2
    L, R = Xb[:, :f], Xb[:, f:]
    eps = 1e-12
    Lc, Rc = L - L.mean(1, keepdims=True), R - R.mean(1, keepdims=True)
    corr = ((Lc * Rc).sum(1)
            / (np.linalg.norm(Lc, axis=1) * np.linalg.norm(Rc, axis=1) + eps))
    cos = ((L * R).sum(1)
           / (np.linalg.norm(L, axis=1) * np.linalg.norm(R, axis=1) + eps))
    pk = np.abs(L.argmax(1) - R.argmax(1)).astype(float)
    hi = np.log((L.max(1) + eps) / (R.max(1) + eps))
    pw = np.log((L.sum(1) + eps) / (R.sum(1) + eps))
    l1 = np.abs(L - R).sum(1)
    return np.column_stack([corr, cos, pk, hi, pw, l1])
