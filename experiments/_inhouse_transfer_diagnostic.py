"""Pre-fit diagnostic for in-house transfer learning, and why in-house precN is low.

Two questions, both answerable without deep fits (invariant 10):

1. **Why is in-house 3-class precN only ~0.66 when N-vs-tremor AUC is 0.84?**
   Hypothesis from `readjudication_list.md` (merged cohort): N<->PD confusion is
   phenotype -- PD patients with no postural tremor look normal. Measured here
   model-free on in-house patients: un-normalised 3-15 Hz tremor RMS per class,
   and the fraction of PD patients whose tremor sits inside the N range.

2. **What could PADS transfer to in-house patients at all?** A model fitted on
   PADS, applied unchanged to in-house patients (logistic regression on the
   16-bin multitaper + descriptors, z-scored within cohort), per axis, against
   the in-house model's own CV AUC. If N-vs-tremor transfers but PD-vs-ET does
   not (as `pd_vs_et_transfer.md` found for PD-vs-ET), pretraining can only
   help the N/tremor part of the network, and ET gains should not be expected.

PREDICTION: (1) a large share (> 25 %) of in-house PD sit below the N 75th
percentile of tremor RMS. (2) PADS->in-house N-vs-tremor AUC > 0.75;
PD-vs-ET < 0.65 (inside the ~0.66 null top).

Run: ``python -m experiments._inhouse_transfer_diagnostic``
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from experiments.own_data_10et import build

CH = slice(3, 6)


def tremor_rms(recs):
    sos = butter(4, [3, 15], btype="band", fs=100.0, output="sos")
    per = {}
    for r in recs:
        x = sosfiltfilt(sos, r.x[CH], axis=-1)
        per.setdefault(r.subject, []).append(np.log10(np.sqrt((x ** 2).mean())))
    return {p: np.mean(v) for p, v in per.items()}


def z(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def cv_auc(X, y, reps=10):
    a = []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
            s[te] = m.fit(X[tr], y[tr]).decision_function(X[te])
        a.append(roc_auc_score(y, s))
    return float(np.mean(a))


def main():
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))

    print("1. In-house tremor RMS (log10, 3-15 Hz, lower arm, OUT)")
    for name, recs in (("2015", rA), ("NewData", rB)):
        amp = tremor_rms(recs)
        lab = {r.subject: r.y for r in recs}
        v = {c: np.array([amp[p] for p in amp if lab[p] == c]) for c in (0, 1, 2)}
        q75 = np.percentile(v[0], 75)
        print(f"  {name:>8}  median N {np.median(v[0]):+.2f}  PD {np.median(v[1]):+.2f}"
              f"  ET {np.median(v[2]):+.2f}   PD below N-p75: "
              f"{(v[1] < q75).mean():.0%}   ET below N-p75: {(v[2] < q75).mean():.0%}")

    A, B, C = build()
    Xin = np.vstack([z(np.hstack([A[0], A[1]])), z(np.hstack([B[0], B[1]]))])
    yin = np.concatenate([A[3], B[3]])
    Xp, yp = z(np.hstack([C[0], C[1]])), C[3]

    print("\n2. PADS -> in-house, fitted once on PADS, applied unchanged")
    for axis in ("N vs tremor", "PD vs ET"):
        if axis == "N vs tremor":
            f = lambda X, y: (X, (y != 0).astype(int))
        else:
            f = lambda X, y: (X[y != 0], (y[y != 0] == 2).astype(int))
        Xt, yt = f(Xp, yp)
        Xi, yi = f(Xin, yin)
        m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Xt, yt)
        tr = roc_auc_score(yi, m.decision_function(Xi))
        rng = np.random.default_rng(0)
        null = [roc_auc_score(rng.permutation(yi), m.decision_function(Xi))
                for _ in range(500)]
        lo, hi = np.percentile(null, [2.5, 97.5])
        print(f"  {axis:>12}  transfer AUC {tr:.3f} (null [{lo:.3f}, {hi:.3f}])"
              f"   in-house own CV {cv_auc(Xi, yi):.3f}   PADS own CV {cv_auc(Xt, yt):.3f}")
    print("MARKER_DONE")


if __name__ == "__main__":
    main()
