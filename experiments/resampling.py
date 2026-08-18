"""SMOTE and other resampling on binary PD-vs-ET -- an untried gap.

Every model in this project has handled imbalance through **loss reweighting**
(`class_weight="balanced"`). Data-level resampling of the minority class has
never been tested here, and SMOTE is the standard method for it.

**The correctness point that decides whether these numbers mean anything:**
resampling must be fitted **inside each training fold**, never on the full
dataset before splitting. SMOTE interpolates between minority neighbours, so
fitting it before the split creates synthetic training points derived from test
patients -- the most common way SMOTE results are overstated. Here every sampler
sits inside an `imblearn.Pipeline`, which applies it during `fit` only and never
to the data being predicted.

Samplers tested:

``none``            the current approach: class_weight="balanced", no resampling
``RandomOver``      duplicate minority patients
``SMOTE``           interpolate between minority neighbours
``BorderlineSMOTE`` interpolate only near the decision boundary
``ADASYN``          more synthesis where the minority is locally outnumbered
``SVMSMOTE``        synthesise along an SVM boundary
``SMOTETomek``      SMOTE then remove Tomek links
``SMOTEENN``        SMOTE then edited nearest neighbours

Resamplers run **without** class weights, since combining both double-corrects.
A class-weighted SMOTE arm is included to show that.

A caution specific to this data: with 21 in-house ET patients, roughly 14 reach
each training fold, so every synthetic point is a blend of a handful of real
patients. `k_neighbors` is clamped accordingly.

Run: ``python -m experiments.resampling``
"""

from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (ADASYN, SMOTE, BorderlineSMOTE,
                                    RandomOverSampler, SVMSMOTE)
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.pd_vs_et import build

REPEATS = 10


def samplers(k):
    """k_neighbors clamped to what the minority count can support."""
    kw = dict(random_state=0)
    return {
        "none (class_weight)": None,
        "RandomOver": RandomOverSampler(**kw),
        "SMOTE": SMOTE(k_neighbors=k, **kw),
        "BorderlineSMOTE": BorderlineSMOTE(k_neighbors=k, **kw),
        "ADASYN": ADASYN(n_neighbors=k, **kw),
        "SVMSMOTE": SVMSMOTE(k_neighbors=k, **kw),
        "SMOTETomek": SMOTETomek(smote=SMOTE(k_neighbors=k, **kw), **kw),
        "SMOTEENN": SMOTEENN(smote=SMOTE(k_neighbors=k, **kw), **kw),
        "SMOTE + class_weight": SMOTE(k_neighbors=k, **kw),
    }


def make_pipe(name, samp):
    """Resampling sits INSIDE the pipeline: fit-time only, never at predict."""
    cw = "balanced" if ("class_weight" in name or samp is None) else None
    lr = LogisticRegression(max_iter=5000, class_weight=cw)
    steps = [("scale", StandardScaler())]
    if samp is not None:
        steps.append(("resample", samp))
    steps.append(("clf", lr))
    return ImbPipeline(steps)


def evaluate(name, samp, X, y, k_folds, repeats=REPEATS):
    rows, failed = [], 0
    for rep in range(repeats):
        prob = np.zeros(len(y))
        for tr, te in StratifiedKFold(k_folds, shuffle=True,
                                      random_state=rep).split(X, y):
            try:
                pipe = make_pipe(name, samp)
                pipe.fit(X[tr], y[tr])
                prob[te] = pipe.predict_proba(X[te])[:, 1]
            except Exception:
                failed += 1
                prob[te] = 0.5
        pred = (prob >= 0.5).astype(int)
        P, R, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1],
                                                     zero_division=0)
        rows.append([roc_auc_score(y, prob), P[0], P[1], 0.5 * (R[0] + R[1])])
    a = np.array(rows)
    mu, sd = a.mean(0), a.std(0)
    note = f"  ({failed} folds failed)" if failed else ""
    print(f"{name:>22}" + "".join(f"{mu[i]:>10.3f} +/-{sd[i]:<4.3f}"
                                  for i in range(4)) + note, flush=True)
    return a


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
    data = build()
    settings = [
        ("in-house (2015+NewData)", ["2015", "NewData"], ["axes"], 3),
        ("MERGED (all three)", ["2015", "NewData", "PADS"],
         ["axes", "stability"], 5),
        ("PADS", ["PADS"], ["spectrum"], 5),
    ]
    for name, tags, keys, k_folds in settings:
        blocks = {q: np.vstack([data[t][0][q] for t in tags])
                  for q in data[tags[0]][0]}
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(np.hstack([blocks[q] for q in keys])[m])
        n_min_train = int(y.sum() * (k_folds - 1) / k_folds)
        k = max(1, min(5, n_min_train - 1))
        print(f"\n{'='*80}")
        print(f"{name}   PD vs ET   n={len(y)} PD={int((y==0).sum())} "
              f"ET={int(y.sum())}   features {'+'.join(keys)} (dim {X.shape[1]})")
        print(f"~{n_min_train} ET per training fold, k_neighbors={k}")
        print(f"{'='*80}")
        print(f"{'sampler':>22}{'AUC':>16}{'precPD':>16}{'precET':>16}"
              f"{'bal-acc':>16}")
        res = {}
        for label, samp in samplers(k).items():
            res[label] = evaluate(label, samp, X, y, k_folds)
        base = res["none (class_weight)"]
        print(f"\npaired vs class_weight baseline, {REPEATS} repeats:")
        # SVMSMOTE was omitted from this list in the first run -- it is the
        # best performer on the two larger cohorts, so its paired interval is
        # exactly the one that matters.
        for q in ("SMOTE", "BorderlineSMOTE", "SVMSMOTE", "ADASYN",
                  "RandomOver", "SMOTETomek", "SMOTE + class_weight"):
            paired(res[q], base, q)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
