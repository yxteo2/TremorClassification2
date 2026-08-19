"""Paired test: one-class Mahalanobis vs logistic regression, and a hybrid.

One-class is the only method in `fewshot_oneclass.py` to beat logistic
regression in-house on all three precision metrics (precPD +0.004, precET
+0.019, macroP +0.011), and it does so with half the variance. The margins are
well inside what noise could produce, so they need a paired test.

The hybrid is worth testing for a specific reason: the two models use different
information. Logistic regression fits the PD/ET **boundary**; one-class models
only the PD **distribution** and never looks at ET. That independence is what
made `axes + stability` the one successful union in this project, whereas
methods doing the same job (RandomForest + SVMSMOTE) failed to combine.

Scores are rank-averaged, not probability-averaged: the one-class score is a
Mahalanobis distance, not a probability, so averaging raw values would let the
distance dominate arbitrarily.

Run: ``python -m experiments.oneclass_paired``
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.fewshot_oneclass import OneClassScorer
from experiments.pd_vs_et import build

REPEATS = 20


def run(kind, X, y, k):
    rows = []
    for rep in range(REPEATS):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=rep).split(X, y):
            lp = Pipeline([("s", StandardScaler()),
                           ("c", LogisticRegression(max_iter=5000,
                                                    class_weight="balanced"))]
                          ).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            oc = OneClassScorer("mahalanobis").fit(X[tr], y[tr]).score(X[te])
            if kind == "logreg":
                s[te] = lp
            elif kind == "oneclass":
                s[te] = oc
            else:
                s[te] = 0.5 * (rankdata(lp) / len(lp) + rankdata(oc) / len(oc))
        pr = (s >= np.quantile(s, 1 - y.mean())).astype(int)
        rows.append([precision_score(y, pr, pos_label=0, zero_division=0),
                     precision_score(y, pr, pos_label=1, zero_division=0),
                     roc_auc_score(y, s),
                     recall_score(y, pr, pos_label=1, zero_division=0),
                     recall_score(y, pr, pos_label=0, zero_division=0)])
    a = np.array(rows)
    a = np.column_stack([a[:, 0], a[:, 1], 0.5 * (a[:, 0] + a[:, 1]), a[:, 2],
                         a[:, 3], a[:, 4]])
    return a


def paired(a, b, name):
    d = a - b
    print(f"  {name}:")
    for i, nm in enumerate(("precPD", "precET", "macroP", "AUC", "ETsens",
                            "PDspec")):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def ppv(se, sp, p):
    return se * p / (se * p + (1 - sp) * (1 - p) + 1e-12)


def main():
    data = build()
    for name, tags, keys, k in (("in-house", ["2015", "NewData"], ["axes"], 3),
                                ("PADS", ["PADS"], ["descriptors", "stability"], 5)):
        blocks = {q: np.vstack([data[t][0][q] for t in tags])
                  for q in data[tags[0]][0]}
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(np.hstack([blocks[q] for q in keys])[m])
        print(f"\n{'='*74}")
        print(f"{name}  PD vs ET  n={len(y)} PD={int((y==0).sum())} "
              f"ET={int(y.sum())}   {REPEATS} repeats")
        print(f"{'='*74}")
        print(f"{'method':>24}{'precPD':>9}{'precET':>9}{'macroP':>9}{'AUC':>9}"
              f"{'PPV@0.5':>10}")
        res = {}
        for kind, lab in (("logreg", "logreg"),
                          ("oneclass", "one-class mahalanobis"),
                          ("hybrid", "rank-avg hybrid")):
            a = run(kind, X, y, k)
            res[lab] = a
            mu, sd = a.mean(0), a.std(0)
            print(f"{lab:>24}{mu[0]:>9.3f}{mu[1]:>9.3f}{mu[2]:>9.3f}"
                  f"{mu[3]:>9.3f}{ppv(mu[4], mu[5], 0.5):>10.3f}"
                  f"   (macroP sd {sd[2]:.3f})")
        print("\npaired vs logreg:")
        for q in ("one-class mahalanobis", "rank-avg hybrid"):
            paired(res[q], res["logreg"], q)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
