"""Does spectral VARIABILITY carry PD-vs-ET signal the frequency marginal loses?

Every transform in this repo collapses its time-frequency surface to the
frequency marginal (`P.mean(0)`) before anything downstream sees it. This screens
what the discarded time axis was worth, as fixed summary statistics rather than
as a surface to be learned — learned time-axis models already lost
(`time_domain_deep.md`).

Feature blocks per recording, each frame normalised to unit total power so
everything describes spectral *shape* and its movement, never amplitude:

    median   (16)  per-bin median across frames — close to the current pipeline
    iqr      (16)  per-bin inter-quartile range across frames
    flux      (1)  mean L1 change between consecutive frames
    wander    (1)  sd of the per-frame peak frequency, in Hz

**Window length is swept, which this project has never done.** Every transform is
fixed at nperseg 256 or 512 — 2.6 s or 5.1 s, i.e. 20-60 cycles of a 4-12 Hz
tremor. Resolving state *switching* wants short frames; resolving *frequency*
wants long ones. That trade-off is the defining choice in time-frequency analysis
and it has never been tested here. 64 / 128 / 256 / 512 samples = 0.64 / 1.28 /
2.56 / 5.12 s.

Validated on synthetics before use: a single stable 6 Hz pacemaker gives IQR 0.614
/ flux 0.070 / wander 0.000 Hz, while a signal switching between 5 and 7 Hz states
gives 1.959 / 0.943 / 1.001 — 3x, 13x and unbounded. Rotation and scale
invariance hold to 1e-11.

Every cell is checked against a **permutation null** with the pipeline refitted
per replicate, per invariant 7: at these ET counts a bootstrap will call chance
results significant.

This is a cheap logistic-regression screen. Only a block that clears its null and
beats the standing `descriptors` baseline is worth taking to the deep model.

Run: ``python -m experiments.tf_variability_screen``
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signal_processing.tf_variability import blocks, patient_table

REPEATS, NPERM = 10, 200
WINDOWS = (64, 128, 256, 512)
COLS = ("AUC", "precPD", "precET", "macroP")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def oof(X, y, k, repeats=REPEATS, seed0=0):
    acc = np.zeros(len(y))
    for rep in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=seed0 + rep).split(X, y):
            p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc += p
    return acc / repeats


def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET)]


def perm_p(X, y, k, obs, n=NPERM):
    rng = np.random.default_rng(0)
    null = []
    for i in range(n):
        yp = rng.permutation(y)
        try:
            null.append(roc_auc_score(yp, oof(X, yp, k, repeats=1,
                                              seed0=1000 + i)))
        except ValueError:
            pass
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) \
        / (1 + len(null))
    return lo, hi, p


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

    print("building tables (4 window lengths) ...", flush=True)
    store = {}
    for tag, recs, ch in cohorts:
        sp = spectrum_table(recs, ch=ch)
        d = {"y": sp[1], "descriptors": desc_table(recs, ch)}
        for w in WINDOWS:
            X, _, p = patient_table(recs, ch=ch, nperseg=w)
            idx = {q: i for i, q in enumerate(p)}
            dim = X.shape[1] if len(X) else 34
            d[w] = np.array([X[idx[q]] if q in idx else np.zeros(dim)
                             for q in sp[2]])
        store[tag] = d
        miss = sum(1 for q in sp[2] if q not in idx)
        print(f"  {tag:>8}: {len(sp[1])} patients"
              f"{f', {miss} without TF features' if miss else ''}")

    B = blocks()
    ARMS = (("median (frequency marginal)", "median"),
            ("iqr only", "iqr"),
            ("variability (iqr+flux+wander)", "variability"),
            ("median + variability", "all"))

    for gname, tags, k in (("PADS", ["PADS"], 5),
                           ("MERGED", ["2015", "NewData", "PADS"], 5)):
        tags = [t for t in tags if t in store]
        y3 = np.concatenate([store[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)

        print(f"\n{'='*92}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {NPERM} permutations")
        print(f"{'='*92}")

        D = np.nan_to_num(np.vstack([store[t]["descriptors"]
                                     for t in tags]))[m]
        s = scores(y, oof(D, y, k))
        lo, hi, pv = perm_p(D, y, k, s[0])
        print(f"{'descriptors (baseline)':>32}{'--':>6}{D.shape[1]:>5}"
              + "".join(f"{v:>9.3f}" for v in s)
              + f"{f'[{lo:.3f},{hi:.3f}]':>18}{pv:>7.3f}"
              + ("*" if pv < 0.05 else " "), flush=True)
        print(f"{'block':>32}{'win':>6}{'dim':>5}"
              + "".join(f"{c:>9}" for c in COLS) + f"{'null 95%':>18}{'p':>8}")

        for lab, key in ARMS:
            for w in WINDOWS:
                T = np.nan_to_num(np.vstack([store[t][w] for t in tags]))[m]
                X = T if key == "all" else T[:, B[key]]
                s = scores(y, oof(X, y, k))
                lo, hi, pv = perm_p(X, y, k, s[0])
                flag = "*" if pv < 0.05 else " "
                print(f"{lab:>32}{w:>6}{X.shape[1]:>5}"
                      + "".join(f"{v:>9.3f}" for v in s)
                      + f"{f'[{lo:.3f},{hi:.3f}]':>18}{pv:>7.3f}{flag}",
                      flush=True)

        # class means of the two scalar variability features, best window
        for w in WINDOWS:
            T = np.vstack([store[t][w] for t in tags])[m]
            fl, wa = T[:, B["flux"]][:, 0], T[:, B["wander"]][:, 0]
            print(f"    win {w:>4}  flux  PD {fl[y==0].mean():.4f} "
                  f"ET {fl[y==1].mean():.4f}   |  wander  "
                  f"PD {wa[y==0].mean():.3f} ET {wa[y==1].mean():.3f} Hz")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
