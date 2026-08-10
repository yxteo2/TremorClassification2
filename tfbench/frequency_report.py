"""Mean / max frequency across every cohort and condition.

Generates the three-part frequency report:
  A. median max_freq and mean_freq per class
  B. PD-vs-ET direction, effect size and significance
  C. classification from those TWO features alone -- with PRECISION, since
     balanced accuracy alone hides what class_weight="balanced" costs

All cohorts go through the identical path: Welch PSD (the only Parseval-exact
transform in ``tfbench.transforms``), 3-15 Hz, wrist-equivalent sensor
(2015/NewData ``lower_arm``, PADS wrist), aggregated per patient, patient-level
LOSO with class-balanced logistic regression.

Run:  python -m tfbench.frequency_report
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfbench.descriptors import describe
from tfbench.transforms import m_welch
from tremor.stats import bootstrap_subject_ci

WRIST_2015 = slice(3, 6)
PADS_WRIST = slice(0, 3)


def load_cohorts(data_root="Data"):
    """Every (cohort, condition, recordings, channel-slice) used in the report."""
    from pdetn.crossdataset import load_pads_extracted
    from pdetn.load_2025 import load_2025
    from tremor.quaternion_data import load_quaternion_recordings

    out = []
    for cond in ("REST", "OUT", "WING"):
        out.append(("2015", cond,
                    load_quaternion_recordings(data_root, action=cond,
                                               mode="angular_velocity"), WRIST_2015))
    for cond in ("REST", "OUT"):
        out.append(("NewData", f"{cond} (seg)",
                    load_2025(mode="angular_velocity", conditions=(cond,)), WRIST_2015))
    for folder, task, label in (("pads_relaxed", "Relaxed", "Relaxed"),
                                ("pads_stretchhold", "StretchHold", "StretchHold")):
        try:
            out.append(("PADS", label,
                        load_pads_extracted(folder, task=task), PADS_WRIST))
        except FileNotFoundError:
            print(f"  (skipping PADS {label}: {folder}/ not present)")
    return out


def patient_freqs(recs, ch):
    """(V, y, patients) where V columns are [max_freq, mean_freq]."""
    d, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        de = describe(*m_welch(x))
        d[r.subject].append((de["max_freq"], de["mean_freq"]))
        lab[r.subject] = r.y
    p = sorted(d)
    return (np.array([np.mean(d[k], 0) for k in p]),
            np.array([lab[k] for k in p]), np.array(p))


def _evaluate(X, y, g):
    prob = cross_val_predict(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, class_weight="balanced")),
        X, y, groups=g, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    bal = 0.5 * (recall_score(y, pred, pos_label=1, zero_division=0)
                 + recall_score(y, pred, pos_label=0, zero_division=0))
    e = bootstrap_subject_ci(y, pred, g, ["neg", "pos"], n_boot=2000, seed=0)["pos"]
    return dict(acc=accuracy_score(y, pred),
                maj=max(np.mean(y == 1), np.mean(y == 0)), bal=bal,
                auc=roc_auc_score(y, prob), f1=f1_score(y, pred),
                prec=precision_score(y, pred, zero_division=0),
                rec=recall_score(y, pred, pos_label=1, zero_division=0),
                lo=e.lo, hi=e.hi)


def report(data_root="Data"):
    cohorts = load_cohorts(data_root)
    S = {}

    print("### A. Median MAX and MEAN frequency (Hz) per class")
    print(f"{'cohort':>9}{'condition':>13}" + "".join(f"{c:>18}" for c in ("N", "PD", "ET")))
    print(f"{'':>22}" + "".join(f"{'max / mean':>18}" for _ in range(3)))
    for coh, cond, recs, ch in cohorts:
        V, y, g = patient_freqs(recs, ch)
        S[(coh, cond)] = (V, y, g)
        cells = [f"{np.median(V[y == k, 0]):.2f} / {np.median(V[y == k, 1]):.2f}"
                 if (y == k).any() else "—" for k in (0, 1, 2)]
        print(f"{coh:>9}{cond:>13}" + "".join(f"{c:>18}" for c in cells), flush=True)

    print("\n### B. PD vs ET — direction, effect, significance")
    print(f"{'cohort':>9}{'condition':>13}{'measure':>11}{'PD':>7}{'ET':>7}"
          f"{'direction':>12}{'effect':>9}{'p':>9}")
    for (coh, cond), (V, y, g) in S.items():
        if (y == 1).sum() < 3 or (y == 2).sum() < 3:
            continue
        for j, nm in ((0, "max_freq"), (1, "mean_freq")):
            a, b = V[y == 1, j], V[y == 2, j]
            u, p = mannwhitneyu(a, b)
            eff = 2 * u / (len(a) * len(b)) - 1
            d = "PD FASTER" if np.median(a) > np.median(b) else "PD slower"
            print(f"{coh:>9}{cond:>13}{nm:>11}{np.median(a):>7.2f}{np.median(b):>7.2f}"
                  f"{d:>12}{eff:>+9.3f}{p:>9.4f}{' *' if p < 0.05 else ''}", flush=True)

    print("\n### C. Classification from max+mean frequency ALONE")
    print(f"{'cohort':>9}{'condition':>13}{'axis':>13}{'n':>6}{'maj':>7}"
          f"{'bal-acc':>9}{'AUC':>7}{'prec':>7}{'rec':>7}{'F1 [95% CI]':>20}")
    for (coh, cond), (V, y, g) in S.items():
        for axis, (Xa, ya, ga) in (
                ("N-vs-Tremor", (V, (y != 0).astype(int), g)),
                ("PD-vs-ET", (V[y != 0], (y[y != 0] == 2).astype(int), g[y != 0]))):
            if len(np.unique(ya)) < 2:
                continue
            m = _evaluate(Xa, ya, ga)
            f1_ci = "{:.3f} [{:.2f},{:.2f}]".format(m["f1"], m["lo"], m["hi"])
            print(f"{coh:>9}{cond:>13}{axis:>13}{len(ya):>6}"
                  f"{m['maj']:>7.3f}{m['bal']:>9.3f}{m['auc']:>7.3f}"
                  f"{m['prec']:>7.3f}{m['rec']:>7.3f}{f1_ci:>20}", flush=True)
    return S


if __name__ == "__main__":
    report()
