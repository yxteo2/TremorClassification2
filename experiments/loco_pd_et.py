"""Does a PD-vs-ET model trained on one cohort work on the other?

Every PD-vs-ET number in this project is **within-cohort** cross-validation. That
answers "can this cohort be separated", and on PADS the answer is yes (five of six
families beat a permutation null, descriptors AUC 0.794, p = 0.005 --
`permutation_null.md`). It does not answer the question a clinic asks, which is
whether a model fitted at one site works at another.

The two directions are not equivalent and both are worth reporting:

  PADS -> in-house   the useful direction. PADS has 28 ET, the largest ET group
                     available, and is where the signal demonstrably lives. If a
                     PADS-fitted model works in-house, the in-house ET shortage
                     stops being binding.
  in-house -> PADS   the control. In-house has no family that clears its own
                     permutation null, so a model fitted there should transfer
                     poorly; if it does not, something is wrong with the
                     comparison rather than with the data.

**The bootstrap here is legitimate in a way it was not for cross-validation.**
`permutation_null.md` showed that resampling patients while holding out-of-fold
predictions fixed is anti-conservative, because it ignores the variance of
refitting. Under leave-one-cohort-out the model is fitted once on a *different*
cohort and never refitted, so conditional on that model the only thing varying is
which test patients were drawn -- which is exactly what a patient bootstrap
estimates. The interval below is therefore conditional on the training cohort,
and is honest about test-set sampling.

Reported as AUC (threshold-free, so it does not depend on an operating point that
would have to be transferred too) plus precision at the test cohort's own
prevalence quantile.

Run: ``python -m experiments.loco_pd_et``
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.pd_vs_et import build

NBOOT = 4000
FAMILIES = ("descriptors", "spectrum", "stability", "axes", "harmonics",
            "ampmod")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def boot_auc(y, p, n=NBOOT, seed=0):
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

    def block(tags, f):
        y3 = np.concatenate([data[t][1] for t in tags])
        keep = y3 != 0
        X = np.nan_to_num(np.vstack([data[t][0][f] for t in tags]))[keep]
        return X, (y3[keep] == 2).astype(int)

    directions = (("PADS", ["PADS"], "in-house", ["2015", "NewData"]),
                  ("in-house", ["2015", "NewData"], "PADS", ["PADS"]))

    for src, stags, dst, dtags in directions:
        Xs0, ys = block(stags, "descriptors")
        Xd0, yd = block(dtags, "descriptors")
        print(f"\n{'='*84}")
        print(f"train on {src} (n={len(ys)}, ET={int(ys.sum())})  ->  "
              f"test on {dst} (n={len(yd)}, ET={int(yd.sum())}, "
              f"prevalence {yd.mean():.3f})")
        print(f"{'='*84}")
        print(f"{'family':>14}{'AUC':>8}{'95 % CI':>20}{'precPD':>9}"
              f"{'precET':>9}   verdict")

        for f in FAMILIES:
            Xs, _ = block(stags, f)
            Xd, _ = block(dtags, f)
            if Xs.shape[1] != Xd.shape[1]:
                print(f"{f:>14}   shape mismatch, skipped")
                continue
            m = clf().fit(Xs, ys)
            p = m.predict_proba(Xd)[:, 1]
            mu, lo, hi = boot_auc(yd, p)
            pr = (p >= np.quantile(p, 1 - yd.mean())).astype(int)
            pPD = precision_score(yd, pr, pos_label=0, zero_division=0)
            pET = precision_score(yd, pr, pos_label=1, zero_division=0)
            if lo > 0.5:
                v = "TRANSFERS"
            elif hi < 0.5:
                v = "transfers INVERTED"
            else:
                v = "no transfer (CI spans 0.5)"
            print(f"{f:>14}{mu:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
                  f"{pPD:>9.3f}{pET:>9.3f}   {v}", flush=True)

        # all families concatenated, for completeness
        Xs = np.hstack([block(stags, f)[0] for f in FAMILIES])
        Xd = np.hstack([block(dtags, f)[0] for f in FAMILIES])
        p = clf().fit(Xs, ys).predict_proba(Xd)[:, 1]
        mu, lo, hi = boot_auc(yd, p)
        pr = (p >= np.quantile(p, 1 - yd.mean())).astype(int)
        print(f"{'ALL concat':>14}{mu:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
              f"{precision_score(yd, pr, pos_label=0, zero_division=0):>9.3f}"
              f"{precision_score(yd, pr, pos_label=1, zero_division=0):>9.3f}")

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
