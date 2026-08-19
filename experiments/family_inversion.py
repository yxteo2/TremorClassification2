"""Are the feature families really anti-predictive in-house, or just silent?

`experiments/score_ensemble.py` measured PD-vs-ET AUC per family and found four
of seven **below chance** on in-house patients:

    family        PADS AUC   in-house AUC
    descriptors     0.794        0.399
    harmonics       0.725        0.404
    asymmetry       0.735        0.430
    ampmod          0.700        0.440
    axes            0.539        0.613

Taken at face value that is a strong claim -- not "these features carry no
in-house signal" but "they carry it backwards", which would mean a PADS-fitted
model is worse than useless on in-house patients rather than merely useless.

The claim is not yet supported. Those numbers are means over 20 repeats of
cross-validation on **the same 119 patients**, so the repeat-to-repeat spread
measures CV noise, not patient sampling. Nothing in it says what would happen on
a different 119 patients, and 21 ET is where every interval in this project gets
wide.

This bootstraps over **patients**, which is the sampling unit the claim is about:
resample patients with replacement, recompute AUC on the held-out-fold
probabilities, and read whether the interval excludes 0.5. Resampling is
stratified within class so that every replicate keeps both classes present.

Run: ``python -m experiments.family_inversion``
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.pd_vs_et import build

REPEATS, NBOOT = 20, 4000
FAMILIES = ("descriptors", "spectrum", "stability", "axes", "harmonics",
            "ampmod", "asymmetry")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def oof_mean(X, y, k, repeats=REPEATS):
    """Held-out probability per patient, averaged over repeats to damp CV noise."""
    acc = np.zeros(len(y))
    for rep in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=rep).split(X, y):
            p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc += p
    return acc / repeats


def patient_bootstrap_auc(y, p, n=NBOOT, seed=0):
    """Class-stratified bootstrap over patients. Returns (mean, lo, hi)."""
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    out = []
    for _ in range(n):
        b = np.concatenate([rng.choice(i0, len(i0), replace=True),
                            rng.choice(i1, len(i1), replace=True)])
        out.append(roc_auc_score(y[b], p[b]))
    return float(np.mean(out)), *np.percentile(out, [2.5, 97.5])


def main():
    data = build()
    # 2015-only and NewData-only sit alongside the pooled in-house arm on
    # purpose. If a family is near chance in each cohort separately but
    # anti-predictive on the pool, the inversion is an artifact of pooling --
    # one model fitted across two cohorts with different feature offsets can
    # learn the cohort difference and apply it as if it were the class
    # difference. That is a different finding from "the feature runs backwards".
    groups = {"PADS": (["PADS"], 5),
              "in-house pooled": (["2015", "NewData"], 3),
              "2015 only": (["2015"], 3),
              "NewData only": (["NewData"], 3)}

    store = {}
    for gname, (tags, k) in groups.items():
        y3 = np.concatenate([data[t][1] for t in tags])
        keep = y3 != 0
        y = (y3[keep] == 2).astype(int)
        if y.sum() < 5:
            print(f"skipping {gname}: only {int(y.sum())} ET")
            continue
        print(f"\n{'='*78}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}   "
              f"AUC with a class-stratified bootstrap over PATIENTS")
        print(f"{'='*78}")
        print(f"{'family':>14}{'dim':>5}{'AUC':>8}{'95 % CI':>20}"
              f"{'  verdict'}")
        store[gname] = {}
        for f in FAMILIES:
            X = np.nan_to_num(np.vstack([data[t][0][f] for t in tags]))[keep]
            p = oof_mean(X, y, k)
            store[gname][f] = (y, p)
            mu, lo, hi = patient_bootstrap_auc(y, p)
            if lo > 0.5:
                v = "predictive"
            elif hi < 0.5:
                v = "ANTI-predictive"
            else:
                v = "indistinguishable from chance"
            print(f"{f:>14}{X.shape[1]:>5}{mu:>8.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20}  {v}", flush=True)

    # Direct test of the inversion: same family, does PADS - in-house differ?
    if "PADS" in store and "in-house pooled" in store:
        print(f"\n{'='*78}")
        print("PADS AUC minus in-house AUC, per family "
              "(independent bootstraps, 95 % CI on the difference)")
        print(f"{'='*78}")
        rng = np.random.default_rng(7)
        for f in FAMILIES:
            ya, pa = store["PADS"][f]
            yb, pb = store["in-house pooled"][f]
            d = []
            for _ in range(NBOOT):
                def bs(y_, p_):
                    i0 = np.flatnonzero(y_ == 0)
                    i1 = np.flatnonzero(y_ == 1)
                    b = np.concatenate([rng.choice(i0, len(i0), replace=True),
                                        rng.choice(i1, len(i1), replace=True)])
                    return roc_auc_score(y_[b], p_[b])
                d.append(bs(ya, pa) - bs(yb, pb))
            lo, hi = np.percentile(d, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"{f:>14}  {np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
