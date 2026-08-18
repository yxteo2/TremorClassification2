"""Tree ensembles and kernel methods on the hand features -- a real gap.

The hand-feature problem here is textbook **tabular**: 10-15 informative
features, 119-423 patients. This project has only ever used logistic regression
and small MLPs on it, and gradient-boosted trees are the standard best-in-class
method for that regime. They have never been tried.

(A 1-D CNN over the feature *vector* would not be appropriate -- convolution
assumes neighbouring positions are related, and feature order is arbitrary.
`max_freq` sitting next to `bandwidth` in the array carries no meaning. Over the
spectrum the assumption holds, which is why `Spectrum1DCNN` works there.)

Why trees might beat a linear model on this specific problem:

* the class boundary may be non-monotone -- e.g. ET sits in a *middle* band of
  linearity while both healthy and severe PD sit outside it, which a linear
  model cannot express and a tree splits trivially;
* features interact -- `four_families.md` found spatial shape and temporal
  steadiness are independent, and `axes + stability` was the first union to beat
  both members, which is exactly where interactions live;
* trees need no scaling and tolerate the very different units here (Hz,
  dimensionless ratios, entropies).

Why they might not: with 21 ET patients in-house, any model with real capacity
can memorise, and every high-capacity model in this project has lost.

Run: ``python -m experiments.tabular_models``
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.pd_vs_et import build

REPEATS = 10


def models(n_pos):
    """Sized for the sample: shallow trees, strong regularisation."""
    return {
        "logreg (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000,
                                                 class_weight="balanced")),
        "logreg L1": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, penalty="l1",
                                                 solver="liblinear", C=0.5,
                                                 class_weight="balanced")),
        "SVM rbf": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, C=1.0,
                                  gamma="scale", class_weight="balanced")),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=max(3, n_pos // 6),
            class_weight="balanced_subsample", random_state=0, n_jobs=1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, max_depth=4, min_samples_leaf=max(3, n_pos // 6),
            class_weight="balanced_subsample", random_state=0, n_jobs=1),
        "HistGradBoost": HistGradientBoostingClassifier(
            max_depth=3, max_iter=200, learning_rate=0.05,
            min_samples_leaf=max(5, n_pos // 4), l2_regularization=1.0,
            class_weight="balanced", random_state=0),
        "HistGradBoost shallow": HistGradientBoostingClassifier(
            max_depth=2, max_iter=120, learning_rate=0.05,
            min_samples_leaf=max(8, n_pos // 3), l2_regularization=5.0,
            class_weight="balanced", random_state=0),
    }


def evaluate(mk, X, y, k, repeats=REPEATS):
    rows = []
    for rep in range(repeats):
        prob = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=rep).split(X, y):
            m = mk()
            m.fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        P, R, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1],
                                                     zero_division=0)
        rows.append([roc_auc_score(y, prob), P[0], P[1], 0.5 * (R[0] + R[1])])
    return np.array(rows)


def paired(a, b, name):
    d = a - b
    print(f"  {name}:")
    for i, nm in enumerate(("AUC", "precPD", "precET", "bal-acc")):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def main():
    import copy
    data = build()
    settings = [
        ("in-house (2015+NewData)", ["2015", "NewData"], ["axes"], 3),
        ("in-house, all hand features", ["2015", "NewData"],
         ["axes", "stability", "descriptors", "harmonics", "ampmod"], 3),
        ("MERGED (all three)", ["2015", "NewData", "PADS"],
         ["axes", "stability"], 5),
        ("PADS", ["PADS"], ["descriptors", "stability"], 5),
    ]
    for name, tags, keys, k in settings:
        blocks = {q: np.vstack([data[t][0][q] for t in tags])
                  for q in data[tags[0]][0]}
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(np.hstack([blocks[q] for q in keys])[m])
        print(f"\n{'='*84}")
        print(f"{name}   PD vs ET   n={len(y)} PD={int((y==0).sum())} "
              f"ET={int(y.sum())}   features: {'+'.join(keys)} (dim {X.shape[1]})")
        print(f"{'='*84}")
        print(f"{'model':>24}{'AUC':>16}{'precPD':>16}{'precET':>16}"
              f"{'bal-acc':>16}")
        res = {}
        for label, proto in models(int(y.sum())).items():
            res[label] = evaluate(lambda p=proto: copy.deepcopy(p), X, y, k)
            mu, sd = res[label].mean(0), res[label].std(0)
            print(f"{label:>24}"
                  + "".join(f"{mu[i]:>10.3f} +/-{sd[i]:<4.3f}" for i in range(4)))
        base = res["logreg (baseline)"]
        best = max((q for q in res if q != "logreg (baseline)"),
                   key=lambda q: res[q][:, 0].mean())
        print(f"\npaired vs logreg, {REPEATS} repeats:")
        for q in ("HistGradBoost", "RandomForest", "SVM rbf", best):
            if q in res:
                paired(res[q], base, q)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
