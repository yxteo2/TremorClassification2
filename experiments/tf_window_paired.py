"""Paired test of the short-window spectrum against the current representation.

`tf_window_control.py` separated three explanations for the short-window gain and
settled two of them, on single unpaired CV estimates:

    arm                              PADS AUC   MERGED AUC
    A multitaper, 16 bins (current)     0.798        0.675
    B multitaper, 8 bins                0.818        0.671
    C short-window MEDIAN, 16           0.825        0.694
    D short-window MEAN, 16             0.830        0.701

* **D >= C** — the across-frame estimator is irrelevant, so "robust estimation
  over many frames" is refuted. A mean does as well or better.
* **B ~ C on PADS** — going from 16 to 8 multitaper bins recovers most of the PADS
  gain on its own, so there most of the effect is *coarseness*, which is already
  this project's top-ranked lever rather than a new one.
* **B ~ A on MERGED** — coarseness explains nothing there, yet D still gains
  +0.026. On the merged cohort a genuine short-window effect remains.

Every one of those is a single number per arm. Differences of 0.02-0.03 at these
sample sizes are exactly the size that has repeatedly evaporated here: a paired
macroP +0.021 [-0.006, +0.048] became +0.005 [-0.020, +0.028] on doubling the
splits (`early_fusion_confirm.md`). So none of it is believable yet.

This pairs them properly: per repeat, every arm is scored on the **same folds**,
giving one row per repeat and a paired bootstrap over 30 repeats. The question is
narrow — does the short-window spectrum beat the current multitaper
representation at equal dimensionality, and is any of it explained by bin count?

Run: ``python -m experiments.tf_window_paired``
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.tf_variability_screen import clf, scores
from experiments.tf_window_control import WIN, logbin_n
from signal_processing.tf_variability import blocks, patient_table

REPEATS = 30
COLS = ("AUC", "precPD", "precET", "macroP")


def oof_folds(X, y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(COLS))]


def main():
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table

    cohorts = [("2015", load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                slice(3, 6)),
               ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        cohorts.append(("PADS", load_pads_extracted("pads_stretchhold"),
                        slice(0, 3)))

    print("building tables ...", flush=True)
    B = blocks()
    store = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        raw = method_table(recs, "multitaper", ch)[0]
        X, _, p = patient_table(recs, ch=ch, nperseg=WIN, stat="mean")
        idx = {q: i for i, q in enumerate(p)}
        dim = X.shape[1] if len(X) else 34
        T = np.array([X[idx[q]] if q in idx else np.zeros(dim) for q in sp[2]])
        store[tag] = {"y": sp[1], "A": logbin_n(raw, 16),
                      "B": logbin_n(raw, 8), "D": T[:, B["median"]]}
        print(f"  {tag:>8}: {len(sp[1])} patients")

    ARMS = (("A multitaper 16 (current)", "A"),
            ("B multitaper 8", "B"),
            ("D short-window mean 16", "D"))

    for gname, tags, k in (("PADS", ["PADS"], 5),
                           ("MERGED", ["2015", "NewData", "PADS"], 5)):
        tags = [t for t in tags if t in store]
        y3 = np.concatenate([store[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = {key: np.nan_to_num(np.vstack([store[t][key] for t in tags]))[m]
             for _, key in ARMS}

        print(f"\n{'='*84}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {REPEATS} paired repeats")
        print(f"{'='*84}")

        res = {key: [] for _, key in ARMS}
        for rep in range(REPEATS):
            folds = list(StratifiedKFold(k, shuffle=True,
                                         random_state=rep).split(X["A"], y))
            for _, key in ARMS:
                res[key].append(scores(y, oof_folds(X[key], y, folds)))
        for key in res:
            res[key] = np.array(res[key])

        print(f"{'arm':>28}{'dim':>5}" + "".join(f"{c:>9}" for c in COLS)
              + "   sd(AUC)")
        for lab, key in ARMS:
            print(f"{lab:>28}{X[key].shape[1]:>5}"
                  + "".join(f"{v:>9.3f}" for v in res[key].mean(0))
                  + f"{res[key][:, 0].std():>11.3f}")

        print("\npaired vs A, same folds:")
        for lab, key in ARMS[1:]:
            print(f"  {lab}:")
            for (dd, lo, hi), c in zip(paired(res[key], res["A"]), COLS):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {c:>7} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

        print("\n  paired D vs B (short window beyond coarseness):")
        for (dd, lo, hi), c in zip(paired(res["D"], res["B"]), COLS):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>7} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
