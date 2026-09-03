"""Is MiniRocket's failure its dimensionality, or its features? A cheap decider.

`rocket_waveform.md` refuted the prediction that an *unlearned* time-domain
estimator would beat the learned TCN on the same waveform. MiniRocket landed at
macroP 0.555–0.558 against the TCN's 0.626 and the reported model's 0.643, both
arms significantly worse (macroP −0.088 and −0.085 *).

That leaves the standing conclusion it was built on in need of repair. The repo
said:

> Time-domain information is only reachable here through estimators that do not
> have to be learned from this cohort.

MiniRocket satisfies that and still lost, so "unlearned" is the wrong
abstraction. Two candidate replacements, and they make opposite predictions:

**(a) Dimensionality.** catch22 is **22** features; MiniRocket is **9 996** on
404 patients — 25× more features than patients, and ~38× more than the ~260 in a
training fold. This project's single most-repeated finding is that "at 404
patients with 49 ET, dimensionality binds harder than information", recorded
across sixteen feature unions. If that is the whole story, **reducing MiniRocket
to ~22 dimensions should recover catch22-level performance.**

**(b) The features themselves.** catch22's statistics were *selected* on 93
datasets for classification performance and low redundancy; MiniRocket's kernels
are generic. If the features simply do not encode tremor structure, **reduction
will not help** and may hurt.

## Why this is cheap

Neither hypothesis needs the deep model. Both arms are MiniRocket features into
the same logistic head, differing only in dimensionality, paired on the same
splits — so this runs in minutes rather than hours and answers the question
directly.

Reduction is by **PCA fitted on the training fold only** (unsupervised, so no
label leak) to 22, 64 and 256 components, against the full 9 996. Supervised
selection would confound the two hypotheses, since choosing features *by label*
imports exactly the "selected for classification" property that (b) is about.

## The prediction, recorded before the run

**(a).** The dimensionality rule is this project's most reproduced result, and
9 996 features on ~260 training patients is its most extreme instance by a wide
margin. So reduction should improve MiniRocket substantially, with the optimum
nearer 22–64 than 9 996.

If reduction does *nothing*, (b) is the answer and the corrected rule becomes
"catch22 works because its features were selected for this kind of problem", not
"because it is low-dimensional".

Run: ``python -m experiments.rocket_dimensionality``
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.rocket_waveform import build_waveform, score

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
DIMS = (22, 64, 256, None)          # None = the full 9 996 features


def main():
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    d = FM.build()
    y, key = d["y"], d["key"]
    recs, keep = load_cohorts()
    W = build_waveform(recs, keep, y)
    print(f"waveform {W.shape}; MiniRocket and PCA both fitted on the training "
          f"fold only\n", flush=True)

    ARMS = [f"PCA {k}" if k else "full 9996" for k in DIMS]
    res, done = resume_load("rocket_dimensionality", ARMS)

    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]

        mr = MiniRocketMultivariate(random_state=sp)
        F = {k: np.nan_to_num(np.asarray(v, dtype=np.float64)) for k, v in
             (("tr", mr.fit_transform(W[tr])), ("va", mr.transform(W[va])),
              ("te", mr.transform(W[te])))}

        for k, a in zip(DIMS, ARMS):
            if k is None:
                Xtr, Xva, Xte = F["tr"], F["va"], F["te"]
            else:
                pca = PCA(n_components=min(k, len(tr) - 1), random_state=0)
                Xtr = pca.fit_transform(F["tr"])
                Xva, Xte = pca.transform(F["va"]), pca.transform(F["te"])
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000,
                                                 class_weight="balanced"))
            m.fit(Xtr, y[tr])
            pv, pt = m.predict_proba(Xva), m.predict_proba(Xte)
            res[a].append(score(pt, tune_offsets(pv, y[va]), y[te]))
        resume_save("rocket_dimensionality", res, sp)
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>14}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>14}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")
    print(f"{'catch22 ref':>14}   ties the spectral descriptors on PADS "
          f"PD-vs-ET (AUC 0.798 vs 0.794)")

    base = res["full 9996"]
    print("\npaired vs the full 9996 features:")
    for a in ARMS[:-1]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nreading it: a large gain from reduction supports (a) dimensionality;")
    print("a null supports (b) the features themselves, and the corrected rule")
    print("becomes 'selected for classification', not 'unlearned'.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
