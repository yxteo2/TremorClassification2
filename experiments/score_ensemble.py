"""Combine the feature families at the SCORE level instead of the feature level.

Eight feature unions in this project have underperformed their best member
(`concat+asym` 0.554 vs 0.709; `descriptors+stability` 0.754 vs 0.807;
`multitaper+traj+stability` 0.639 vs 0.660). Every one of them concatenated
feature blocks and fitted a single model. At 404 patients with 49 ET,
dimensionality binds harder than information, and appending a block spends
degrees of freedom faster than it adds signal.

Two unions did work -- `axes + stability` and `logreg + one-class Mahalanobis`
(`oneclass_hybrid.md`) -- and the second worked at the **score** level: fit each
model separately, convert to within-fold ranks, average. That costs no extra
degrees of freedom in the classifier at all, because each member still sees only
its own block.

This tests whether the rule generalises. The families are known to be strongly
**cohort-inverted**, which is exactly the situation an ensemble should help:

    family        PADS AUC   in-house AUC    source
    spectrum        0.790        0.552        four_families.md
    descriptors     0.807        0.482        temporal_stability.md
    stability       0.742        0.652        temporal_stability.md
    axes            0.558        0.641        four_families.md
    harmonics       0.736        0.402        four_families.md

No single family is good on both. If score-averaging recovers most of the better
family on each cohort without being told which cohort it is looking at, that is a
more useful model than any of them -- and it is the kind of claim a paper can
make, because it is about robustness rather than a headline number.

Arms:

  each family alone        the seven blocks, logistic regression
  rank-avg ALL             uniform average of within-fold ranks, no selection,
                           so nothing is tuned on test
  rank-avg, AUC-weighted   weights from an inner CV on the training fold only
  concat ALL               the known-bad feature-level control
  + one-class              the rank-avg ensemble with the PD-density score added
                           as one more member

The headline comparison is **rank-avg ALL vs the best single family**, and that
comparison is deliberately unfair to the ensemble: the best family is picked
post-hoc per cohort, with hindsight the ensemble does not get.

Run: ``python -m experiments.score_ensemble``
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.pd_vs_et import build

REPEATS = 20
FAMILIES = ("descriptors", "spectrum", "stability", "axes", "harmonics",
            "ampmod", "asymmetry")
COLS = ("AUC", "precPD", "precET", "macroP", "ETsens")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def oof_family(X, y, folds):
    """Out-of-fold probability for one feature block."""
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def oof_oneclass(X, y, folds):
    """Mahalanobis distance from the PD density, fitted on training PD only.

    Higher = further from PD = more ET-like. Never sees an ET patient in
    training, which is the point: it does not have to learn the rare class.
    """
    s = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        A, B = sc.transform(X[tr]), sc.transform(X[te])
        pd_only = A[y[tr] == 0]
        try:
            m = MinCovDet(support_fraction=0.9, random_state=0).fit(pd_only)
            s[te] = m.mahalanobis(B)
        except Exception:            # singular covariance on a thin fold
            mu = pd_only.mean(0)
            s[te] = ((B - mu) ** 2).sum(1)
    return s


def rank01(v):
    return rankdata(v) / len(v)


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


def inner_weights(blocks, y, tr, seed):
    """AUC of each family from a CV *inside* the training fold only."""
    w = []
    inner = list(StratifiedKFold(3, shuffle=True,
                                 random_state=seed).split(tr, y[tr]))
    for f in FAMILIES:
        X = blocks[f][tr]
        p = np.zeros(len(tr))
        for a, b in inner:
            p[b] = clf().fit(X[a], y[tr][a]).predict_proba(X[b])[:, 1]
        try:
            w.append(max(roc_auc_score(y[tr], p) - 0.5, 0.0))
        except ValueError:
            w.append(0.0)
    w = np.array(w)
    return w / w.sum() if w.sum() > 0 else np.ones(len(FAMILIES)) / len(FAMILIES)


def main():
    data = build()
    groups = {"PADS": ["PADS"], "in-house": ["2015", "NewData"],
              "MERGED": ["2015", "NewData", "PADS"]}

    for gname, tags in groups.items():
        y3 = np.concatenate([data[t][1] for t in tags])
        keep = y3 != 0
        y = (y3[keep] == 2).astype(int)
        blocks = {f: np.nan_to_num(np.vstack([data[t][0][f] for t in tags]))[keep]
                  for f in FAMILIES}
        allX = np.hstack([blocks[f] for f in FAMILIES])
        k = 3 if gname == "in-house" else 5

        print(f"\n{'='*80}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"prevalence {y.mean():.3f}   {REPEATS} repeats")
        print(f"{'='*80}")
        print(f"{'model':>26}{'dim':>6}" + "".join(f"{c:>9}" for c in COLS))

        res = {f: [] for f in FAMILIES}
        for extra in ("rank-avg ALL", "rank-avg AUC-weighted", "concat ALL",
                      "rank-avg ALL + one-class"):
            res[extra] = []

        for rep in range(REPEATS):
            folds = list(StratifiedKFold(k, shuffle=True,
                                         random_state=rep).split(allX, y))
            oof = {f: oof_family(blocks[f], y, folds) for f in FAMILIES}
            for f in FAMILIES:
                res[f].append(scores(y, oof[f]))

            R = np.stack([rank01(oof[f]) for f in FAMILIES], 1)
            res["rank-avg ALL"].append(scores(y, R.mean(1)))

            # weights fitted inside each training fold, applied to its test fold
            wp = np.zeros(len(y))
            for tr, te in folds:
                w = inner_weights(blocks, y, tr, rep)
                wp[te] = (R[te] * w).sum(1)
            res["rank-avg AUC-weighted"].append(scores(y, wp))

            res["concat ALL"].append(scores(y, oof_family(allX, y, folds)))

            oc = rank01(oof_oneclass(blocks["descriptors"], y, folds))
            res["rank-avg ALL + one-class"].append(
                scores(y, np.column_stack([R, oc]).mean(1)))

        for kk in res:
            res[kk] = np.array(res[kk])
        for kk in list(FAMILIES) + ["concat ALL", "rank-avg ALL",
                                    "rank-avg AUC-weighted",
                                    "rank-avg ALL + one-class"]:
            dim = (blocks[kk].shape[1] if kk in FAMILIES
                   else (allX.shape[1] if kk == "concat ALL" else 0))
            print(f"{kk:>26}{dim if dim else '-':>6}" +
                  "".join(f"{v:>9.3f}" for v in res[kk].mean(0)), flush=True)

        best = max(FAMILIES, key=lambda f: res[f].mean(0)[3])
        print(f"\n  best single family by macroP (chosen with hindsight): {best}")
        for lab in ("rank-avg ALL", "rank-avg AUC-weighted", "concat ALL",
                    "rank-avg ALL + one-class"):
            print(f"    {lab} vs {best}:")
            for (d, lo, hi), c in zip(paired(res[lab], res[best]), COLS):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"      {c:>7} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
