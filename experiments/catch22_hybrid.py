"""Do the temporal state features add to the spectral descriptors on PADS?

`catch22_family.py` found, on PADS PD-vs-ET (28 ET, permutation-tested):

    descriptors                 AUC 0.794   (the standing best)
    catch22 state subset (6)    AUC 0.805   (the highest measured)
    catch22 full (22)           AUC 0.761
    descriptors + catch22 (32)  AUC 0.749   (dilution, as the rule predicts)

The state subset is the six catch22 features that encode Häring et al.'s
mechanism -- several discrete stable oscillator states in PD against a single
pacemaker in ET. It was fixed **a priori** from that paper, not selected on this
data, so its 0.805 is not a selection artifact. But 0.805 against 0.794 is well
inside the noise of a single unpaired comparison and means nothing yet.

This pairs them, and tests whether they combine.

The repo's combination rule (`score_vs_feature_fusion.md`) says combination pays
only when the members are **comparable in strength** and **differ in kind**, and
that it should happen at the **score** level. This is the cleanest instance of
that situation the project has produced:

    descriptors            spectral  -- shape of the power spectrum
    catch22 state subset   temporal  -- how the waveform moves between states

Neither reads the other's information, and on PADS they are within 0.011 AUC of
each other. If the rule is right, the rank-averaged hybrid should beat both, and
the feature-level concatenation should not.

Arms: each family alone, their rank-averaged combination, and their
concatenation as the control. Paired bootstrap CIs against `descriptors` over 20
repeats of stratified CV.

Run: ``python -m experiments.catch22_hybrid``
"""

from __future__ import annotations

import os

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signal_processing.catch22_features import (FEATURE_NAMES, STATE_FEATURES,
                                                patient_table as c22_table)

REPEATS = 20
COLS = ("AUC", "precPD", "precET", "macroP", "ETsens")
STATE_IDX = [FEATURE_NAMES.index(n) for n in STATE_FEATURES]


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def oof(X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    se = recall_score(y, pr, pos_label=1, zero_division=0)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET), se]


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(COLS))]


def rank01(v):
    return rankdata(v) / len(v)


def main():
    from common.cohorts import desc_table
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table

    cohorts = [("2015", load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                slice(3, 6)),
               ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        cohorts.append(("PADS", load_pads_extracted("pads_stretchhold"),
                        slice(0, 3)))

    print("building tables ...", flush=True)
    blocks = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        c22, _, p22 = c22_table(recs, ch=ch)
        idx = {p: i for i, p in enumerate(p22)}
        C = np.array([c22[idx[p]] if p in idx else np.zeros(len(FEATURE_NAMES))
                      for p in sp[2]])
        blocks[tag] = {"descriptors": desc_table(recs, ch),
                       "state": C[:, STATE_IDX], "y": sp[1]}
        print(f"  {tag:>8}: {len(sp[1])} patients")

    for gname, tags, k in (("PADS", ["PADS"], 5),
                           ("MERGED", ["2015", "NewData", "PADS"], 5)):
        tags = [t for t in tags if t in blocks]
        y3 = np.concatenate([blocks[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        D = np.nan_to_num(np.vstack([blocks[t]["descriptors"]
                                     for t in tags]))[m]
        S = np.nan_to_num(np.vstack([blocks[t]["state"] for t in tags]))[m]
        U = np.hstack([D, S])

        print(f"\n{'='*80}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {REPEATS} repeats")
        print(f"{'='*80}")

        ARMS = ("descriptors (spectral)", "catch22 state (temporal)",
                "rank-avg hybrid", "concat (control)")
        res = {a: [] for a in ARMS}
        for rep in range(REPEATS):
            folds = list(StratifiedKFold(k, shuffle=True,
                                         random_state=rep).split(D, y))
            pD, pS, pU = oof(D, y, folds), oof(S, y, folds), oof(U, y, folds)
            res["descriptors (spectral)"].append(scores(y, pD))
            res["catch22 state (temporal)"].append(scores(y, pS))
            res["rank-avg hybrid"].append(
                scores(y, 0.5 * (rank01(pD) + rank01(pS))))
            res["concat (control)"].append(scores(y, pU))
        for a in ARMS:
            res[a] = np.array(res[a])

        print(f"{'arm':>28}{'dim':>5}" + "".join(f"{c:>9}" for c in COLS)
              + "   sd(AUC)")
        dims = {"descriptors (spectral)": D.shape[1],
                "catch22 state (temporal)": S.shape[1],
                "rank-avg hybrid": 0, "concat (control)": U.shape[1]}
        for a in ARMS:
            mu = res[a].mean(0)
            dd = dims[a] if dims[a] else "-"
            print(f"{a:>28}{dd:>5}" + "".join(f"{v:>9.3f}" for v in mu)
                  + f"{res[a][:, 0].std():>11.3f}")

        base = res["descriptors (spectral)"]
        print("\npaired vs descriptors, same folds:")
        for a in ARMS[1:]:
            print(f"  {a}:")
            for (dd, lo, hi), c in zip(paired(res[a], base), COLS):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

        print("\npaired hybrid vs catch22 state alone:")
        for (dd, lo, hi), c in zip(paired(res["rank-avg hybrid"],
                                          res["catch22 state (temporal)"]),
                                   COLS):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
