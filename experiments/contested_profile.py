"""What is a contested patient, physically? The question the ceiling turns on.

`ensemble_diversity.md` measured the ceiling's shape — 59.5 % of patients
unanimous and 68.8 % correct, 40.5 % contested and 48.5 % correct against a
46.5 % constant baseline — and found the contested set concentrated by cohort
(NewData 0.573 against 2015 0.307, class composition controlled). It listed one
untested follow-up as the next thing worth knowing:

> whether contested patients cluster by tremor severity or amplitude, which would
> be a **mechanism** rather than a cohort label.

That distinction is the whole point. "NewData patients are contested" is a fact
about provenance and suggests nothing to build. "Weak-tremor patients are
contested" is a fact about physics and says exactly what to change: the
representation is failing where the oscillation is small relative to voluntary
motion, and the fix is amplitude-aware processing, not a better classifier.

`contested_gating.md` closed the alternative route — the contested set retains
structure, but nothing tried sees structure the deep model does not already see,
so the remaining path is a representation that separates those patients. This is
the measurement that would aim it.

## Design

Contested status is defined per test fold, so it is measured per patient across
the folds that patient lands in. Over 20 splits at a 20 % test fraction each
patient appears in about four test folds, giving a per-patient **contested rate**
rather than a single noisy flag.

That rate is then related to the ten interpretable spectral descriptors
(`frequency/descriptors.py`): max/mean/median frequency, spectral spread,
entropy, Q factor, peak share, frequency IQR, low/high ratio, total power.

Two things are reported, and they answer different questions:

**Correlation per descriptor** — Spearman rho of contested rate against each
descriptor. Reported both overall and **within class**, because contested rate
already differs by class (N 0.294, PD 0.497, ET 0.420), so any descriptor that
merely tracks class will correlate spuriously. The within-class figure is the one
to read.

**Predictability** — can contested rate be predicted from the descriptors at all?
Scored by patient-level 5-fold cross-validation, so no patient is in both train
and test. An AUC near 0.5 means contestedness is not a property of these signal
descriptors, and the search for a mechanism has to look elsewhere — at the raw
waveform, the recording length, or the number of recordings. An AUC well above
0.5 names the handle.

Nothing here changes the reported model. It is a diagnostic.

Run: ``python -m experiments.contested_profile``
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC
from experiments.final_model import build
from experiments.pooling_rules import fit_members
from frequency.descriptors import DESCRIPTOR_NAMES

# 30 rather than the usual 20: the per-patient contested rate is k/appearances,
# so at 20 splits it takes only ~5 distinct values and the median split used for
# the AUC is badly tied. 30 splits gives ~6 appearances and a finer rate.
SPLITS = 30
COHORTS = ("2015", "NewData", "PADS")


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    desc, traj, spec = d["DESC"], d["TRAJ"], d["SPEC"]["multitaper"]
    coh = np.array([k.rsplit("_", 1)[0] for k in key])
    n = len(y)

    seen = np.zeros(n)
    cont = np.zeros(n)

    print(f"n={n}  {SPLITS} splits, ~{SPLITS*TEST_FRAC:.0f} test appearances "
          f"per patient\n", flush=True)

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        _, T = fit_members(spec, np.hstack([desc, A]), traj, y, tr, va, te)
        arg = np.stack([T[i].argmax(1) for i in range(len(T))])
        unan = (arg == arg[0]).all(0)
        seen[te] += 1
        cont[te] += (~unan).astype(float)
        print(f"  split {sp+1}/{SPLITS}  contested {float((~unan).mean()):.3f}",
              flush=True)

    ok = seen >= 2
    rate = np.full(n, np.nan)
    rate[ok] = cont[ok] / seen[ok]
    print(f"\n{int(ok.sum())} patients seen in >=2 test folds; "
          f"mean appearances {seen[ok].mean():.1f}")
    print(f"contested rate: mean {np.nanmean(rate):.3f}, "
          f"sd {np.nanstd(rate):.3f}")
    for c in (0, 1, 2):
        m = ok & (y == c)
        print(f"    class {c}: {np.nanmean(rate[m]):.3f}")
    for c in COHORTS:
        m = ok & (coh == c)
        print(f"    {c:>8}: {np.nanmean(rate[m]):.3f}")

    print(f"\n{'descriptor':>18}{'rho overall':>13}{'rho | N':>10}"
          f"{'rho | PD':>10}{'rho | ET':>10}{'mean|within':>13}")
    rows = []
    for j, nm in enumerate(DESCRIPTOR_NAMES[:desc.shape[1]]):
        x = desc[:, j]
        if np.allclose(x, x[0]):
            print(f"{nm:>18}   constant column, skipped")
            continue
        ro = spearmanr(x[ok], rate[ok]).correlation
        per = []
        for c in (0, 1, 2):
            m = ok & (y == c)
            per.append(spearmanr(x[m], rate[m]).correlation if m.sum() > 8
                       else np.nan)
        mw = float(np.nanmean(per))
        rows.append((nm, ro, per, mw))
        print(f"{nm:>18}{ro:>13.3f}" + "".join(f"{v:>10.3f}" for v in per)
              + f"{mw:>13.3f}")

    if rows:
        best = max(rows, key=lambda r: abs(r[3]))
        print(f"\nstrongest within-class association: {best[0]} "
              f"(mean rho {best[3]:+.3f})")

    # Is contestedness predictable from the descriptors at all?
    print("\nPREDICTABILITY -- patient-level 5-fold CV, no patient in both "
          "train and test")
    hi = (rate[ok] > np.nanmedian(rate[ok])).astype(int)
    X = desc[ok]
    aucs = []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for a, b in cv.split(X, hi):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced"))
        m.fit(X[a], hi[a])
        aucs.append(roc_auc_score(hi[b], m.predict_proba(X[b])[:, 1]))
    print(f"  descriptors -> often-contested   AUC {np.mean(aucs):.3f} "
          f"(sd {np.std(aucs):.3f})")

    # and with the class label added, to see how much is just class
    Xc = np.hstack([X, np.eye(3)[y[ok]]])
    aucs2 = []
    for a, b in cv.split(Xc, hi):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced"))
        m.fit(Xc[a], hi[a])
        aucs2.append(roc_auc_score(hi[b], m.predict_proba(Xc[b])[:, 1]))
    print(f"  + true class label               AUC {np.mean(aucs2):.3f} "
          f"(sd {np.std(aucs2):.3f})")

    # The median split is tied when the rate takes few values, so also predict
    # the rate itself and score by rank correlation, which uses all of it.
    from sklearn.linear_model import RidgeCV
    r = rate[ok]
    pred = np.zeros(len(r))
    from sklearn.model_selection import KFold
    for a, b in KFold(5, shuffle=True, random_state=0).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        m.fit(X[a], r[a])
        pred[b] = m.predict(X[b])
    rho = spearmanr(pred, r).correlation
    print(f"  descriptors -> contested RATE    Spearman rho {rho:+.3f} "
          f"(out-of-fold)")
    print("\n  AUC near 0.5 means contestedness is not a property of these")
    print("  descriptors, and the mechanism has to be sought in the raw")
    print("  waveform, recording length or recording count instead.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
