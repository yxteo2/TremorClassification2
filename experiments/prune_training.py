"""Drop the hardest N and PD patients from training, and see if ET precision rises.

The idea: some majority-class patients are mislabelled, atypical, or simply
uninformative, and they drag the decision boundary across the minority class.
Removing them should sharpen PD-vs-ET without costing anything, because N and PD
are abundant (167 and 188) while ET is not (49) and is never touched.

**CORRECTION — the motivating premise was wrong.** This was written expecting
label noise in the majority classes, citing "20 PADS records labelled parkinsonian
are Atypical Parkinsonism". The extracted manifest's `raw_label` takes exactly
three values — "Parkinson's", "Healthy", "Essential Tremor" — so
`extract_pads.py`'s strict exact-match has *already* excluded the atypical
records; they sit in PADS's differential-diagnoses group, dropped except for ET.
There is **no known majority-class label contamination left to remove.** The
measured results below are unaffected; only the motivation was.

## The control that makes it interpretable

Dropping the *hardest* k patients confounds two things: **which** patients leave,
and the fact that **k majority patients left at all**. Fewer majority examples is
itself a class-balance change, and this project has already shown class balance
moves ET precision hard (uncapped PADS drives precET from 0.612 to 0.221).

So every hard-drop arm is matched by a **random-drop** arm removing the same
number from the same classes. If random does as well, the effect is undersampling
and has nothing to do with which patients were chosen.

## Avoiding the obvious leak

Difficulty is scored **inside the training fold only**, by 5-fold inner CV on the
training patients. Validation is left intact — it tunes the class priors, and
pruning it would bias them. Test is never touched, seen, or scored against. The
scorer is a logistic regression on the same packed features rather than the deep
model itself: much cheaper, and selection does not need to use the final model.

Arms, merged 3-class protocol, 20 splits, paired:

  k=0                baseline, the reported model
  hard-drop 5        the 5 worst-predicted N and 5 worst PD leave training
  hard-drop 15       dose-response
  random-drop 5      matched control
  random-drop 15     matched control

Run: ``python -m experiments.prune_training``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL, build
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS, SEEDS = 20, (0, 1, 2)
DROP_CLASSES = (0, 1)          # N and PD only; ET is never dropped


def difficulty(X, y, tr, seed=0):
    """1 - p(true class) for each TRAINING patient, from inner CV on train only.

    Nothing outside ``tr`` is read. Higher means the model finds that patient
    harder to place, which is what "would decrease training accuracy" means
    operationally.
    """
    Xt, yt = X[tr], y[tr]
    p = np.zeros((len(tr), 3))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    for a, b in cv.split(Xt, yt):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced"))
        m.fit(Xt[a], yt[a])
        pr = m.predict_proba(Xt[b])
        for j, cl in enumerate(m.classes_):
            p[b, cl] = pr[:, j]
    return 1.0 - p[np.arange(len(tr)), yt]


def prune(X, y, tr, k, mode, seed=0):
    """Return training indices with k patients of each DROP_CLASS removed."""
    if k <= 0:
        return tr
    if mode == "hard":
        d = difficulty(X, y, tr, seed=seed)
    elif mode == "easy":
        # the mirror image. If the hardest majority patients are boundary-
        # defining rather than noisy -- which is what the hard/random contrast
        # says -- then dropping the EASIEST should be harmless, since those sit
        # far from the boundary and constrain it least.
        d = -difficulty(X, y, tr, seed=seed)
    else:
        d = np.random.default_rng(1000 + seed).random(len(tr))
    keep = np.ones(len(tr), bool)
    for cl in DROP_CLASSES:
        pos = np.flatnonzero(y[tr] == cl)
        if len(pos) <= k:
            continue
        worst = pos[np.argsort(-d[pos])[:k]]
        keep[worst] = False
    return tr[keep]


def fit_eval(spec, desc, traj, y, tr, va, te):
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
             for s in SEEDS]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
    pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]
    packed = np.hstack([spec, D, traj])

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print("dropping from N and PD only; ET is never touched")
    print("difficulty scored by inner CV on the TRAINING fold alone\n", flush=True)

    ARMS = (("k=0 (baseline)", 0, "hard"),
            ("easy-drop 5", 5, "easy"),
            ("easy-drop 15", 15, "easy"),
            ("random-drop 5", 5, "random"),
            ("random-drop 15", 15, "random"))
    res = {a: [] for a, _, _ in ARMS}
    dropped_cohort = []

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        for lab, k, mode in ARMS:
            tr2 = prune(packed, y, tr, k, mode, seed=sp)
            res[lab].append(fit_eval(spec, D, traj, y, tr2, va, te))
            if lab == "easy-drop 15":
                gone = np.setdiff1d(tr, tr2)
                dropped_cohort.append(gone)
        print(f"  split {sp+1}/{SPLITS}  train {len(tr)} -> "
              f"{len(prune(packed, y, tr, 15, 'easy', seed=sp))} at k=15",
              flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, _, _ in ARMS:
        print(f"{lab:>20}" + "".join(f"{v:>9.3f}" for v in res[lab].mean(0))
              + f"{res[lab][:, 3].std():>12.3f}")

    base = res["k=0 (baseline)"]
    print("\npaired vs k=0, same splits:")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\neasy vs random at the same k — is it WHICH patients, or just fewer?")
    for k in (5, 15):
        print(f"  k={k}:")
        for (dd, lo, hi), c in zip(paired(res[f"easy-drop {k}"],
                                          res[f"random-drop {k}"]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    # which patients does the hard rule keep choosing?
    from collections import Counter
    cnt = Counter(int(i) for g in dropped_cohort for i in g)
    coh = np.array(["2015"] * 151 + ["NewData"] * 56 +
                   ["PADS"] * (len(y) - 207)) if len(y) > 207 else None
    print(f"\nmost frequently dropped patients at k=15 "
          f"({SPLITS} splits, 30 slots per split):")
    for i, c in cnt.most_common(10):
        tag = f"  cohort {coh[i]}" if coh is not None and i < len(coh) else ""
        print(f"    idx {i:>4}  class {int(y[i])}  dropped in {c}/{SPLITS} "
              f"splits{tag}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
