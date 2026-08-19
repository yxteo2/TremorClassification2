"""Is below-chance AUC evidence of anti-prediction, or an artifact of tiny folds?

`experiments/family_inversion.py` reports PD-vs-ET AUCs well below 0.5 with
patient-bootstrap intervals that exclude 0.5 -- descriptors 0.339, harmonics
0.323, asymmetry 0.307 on pooled in-house patients -- and on NewData two families
come out at **exactly 0.000**: all 6 ET patients score below all 23 PD patients,
every repeat. The repo's standing notes make the same kind of claim ("on 2015
every frequency feature is BELOW chance, 0.29-0.32").

An AUC of exactly 0.000 is not a finding. It is perfect inverted separation, and
there is a mechanism that produces it without any anti-predictive signal
existing:

  With 6 ET and 3-fold CV, each training fold defines the ET class from **4**
  patients while the 2 held-out ET are excluded. Removing a patient from a
  4-patient centroid moves that centroid a long way -- and away from the very
  patient being scored. The 23 PD patients lose one of ~15 per fold, so their
  centroid barely moves. `class_weight="balanced"` amplifies this by upweighting
  the four remaining ET. Held-out ET therefore land systematically on the PD
  side, and because ET is the more dispersed class they land *past* the PD
  patients, which is what drives AUC below 0.5 rather than merely to 0.5.

If that is the mechanism, then **permuting the labels should reproduce it**. Under
a permutation the classes carry no information, so an unbiased procedure must
give a null centred at 0.5. If instead the null sits below 0.5, the procedure has
a negative bias at this sample size and every below-chance reading in this
project is measuring that bias rather than the data.

This runs the identical pipeline on permuted labels and reports where the null
actually sits, per cohort and family, together with the permutation p-value of
the observed AUC against its own null (which is valid whatever the null's
centre).

Run: ``python -m experiments.cv_null``
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.pd_vs_et import build

NPERM, REPEATS = 200, 5
FAMILIES = ("descriptors", "spectrum", "stability", "axes", "harmonics",
            "ampmod")


def clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000,
                                            class_weight="balanced"))


def oof_auc(X, y, k, repeats=REPEATS, seed0=0):
    acc = np.zeros(len(y))
    for rep in range(repeats):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=seed0 + rep).split(X, y):
            p[te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        acc += p
    return roc_auc_score(y, acc / repeats)


def main():
    data = build()
    groups = {"PADS": (["PADS"], 5),
              "in-house pooled": (["2015", "NewData"], 3),
              "2015 only": (["2015"], 3),
              "NewData only": (["NewData"], 3)}

    for gname, (tags, k) in groups.items():
        y3 = np.concatenate([data[t][1] for t in tags])
        keep = y3 != 0
        y = (y3[keep] == 2).astype(int)
        if y.sum() < 5:
            continue
        print(f"\n{'='*88}")
        print(f"{gname}  PD vs ET  n={len(y)}  ET={int(y.sum())}  "
              f"{k}-fold, {NPERM} label permutations")
        print(f"{'='*88}")
        print(f"{'family':>14}{'observed':>10}{'null mean':>11}"
              f"{'null 95 %':>20}{'p(perm)':>10}   note")

        for f in FAMILIES:
            X = np.nan_to_num(np.vstack([data[t][0][f] for t in tags]))[keep]
            obs = oof_auc(X, y, k)
            rng = np.random.default_rng(0)
            null = []
            for i in range(NPERM):
                yp = rng.permutation(y)
                try:
                    null.append(oof_auc(X, yp, k, repeats=1, seed0=1000 + i))
                except ValueError:
                    pass
            null = np.array(null)
            lo, hi = np.percentile(null, [2.5, 97.5])
            # two-sided permutation p-value against the procedure's own null
            p = (1 + np.sum(np.abs(null - null.mean())
                            >= abs(obs - null.mean()))) / (1 + len(null))
            note = ""
            if hi < 0.5:
                note = "NULL IS BELOW 0.5 -- procedure is biased here"
            elif lo > 0.5:
                note = "null above 0.5"
            print(f"{f:>14}{obs:>10.3f}{null.mean():>11.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>20}{p:>10.3f}   {note}",
                  flush=True)

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
