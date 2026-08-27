"""Balanced bagging: many models on different majority subsamples, all ET kept.

The reported model averages **3 seeds trained on identical data**. The seeds differ
only in weight initialisation, so the three members are highly correlated and the
average removes little variance.

Two findings from this session make a stronger ensemble worth trying, and both are
measurements rather than hopes:

1. **Randomly removing majority patients is free.** `prune_training.md` and
   `influence_prune.md` both measured it: dropping 10 random N/PD costs macroP
   −0.002 and +0.004 respectively, neither significant, with precET moving +0.022
   in one of them. So majority subsampling is a diversity knob available at no
   cost.
2. **No harmful subset exists to find.** Difficulty-based and influence-based
   selection both failed. If no *particular* majority patients are the problem,
   the way to use that headroom is not to choose which to drop but to **drop
   different ones in every ensemble member** and average over the choice.

That is balanced bagging, and it is the standard ensemble for imbalanced data:
each member sees all of the scarce class and a different sample of the abundant
ones. It has never been tried here — the project tested SMOTE, class weights and
capping, all of which change the *single* training set, never the ensemble.

**Why it should help precision specifically.** Precision at 12 % prevalence is
read from the very top of the ranking. Averaging decorrelated models sharpens
exactly that region, because idiosyncratic high-confidence errors of one member
are voted down by members that never saw the patients which caused them.

## The control that decides it

More members is itself a change. `n_seed = B` trains the same number of models on
the **full** training set, differing only by seed. If bagging and seeding tie,
the gain is ensemble size and has nothing to do with subsampling.

  baseline (3 seeds)   the reported model
  6 seeds, full data   more members, no diversity in the data
  6 bags, 70 % majority  more members, each on a different majority subsample

Every bag keeps **all ET** and resamples only N and PD, so the scarce class is
never diluted. Priors are tuned on the untouched validation split exactly as in
the reported model, and test is never touched.

Run: ``python -m experiments.balanced_bagging``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL, build
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
MAJORITY = (0, 1)          # N and PD are subsampled; ET never is
KEEP_FRAC = 0.70


def members(y, tr, n, mode, seed0=0):
    """n training index sets. 'seed' repeats tr; 'bag' subsamples the majority."""
    if mode == "seed":
        return [tr] * n
    out = []
    for b in range(n):
        rng = np.random.default_rng(9000 + seed0 * 100 + b)
        keep = [i for i in tr if y[i] not in MAJORITY]        # every ET stays
        for cl in MAJORITY:
            pos = np.array([i for i in tr if y[i] == cl])
            m = max(int(round(KEEP_FRAC * len(pos))), 10)
            keep.extend(rng.choice(pos, min(m, len(pos)), replace=False))
        out.append(np.array(sorted(keep), int))
    return out


def fit_ensemble(spec, desc, traj, y, trs, va, te):
    """Average probabilities over ensemble members, each with its own seed."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_all, pt_all = [], []
    for b, tr in enumerate(trs):
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            pv, pt = train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd,
                           y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd],
                           seed=b)
            pv_all.append(pv)
            pt_all.append(pt)
    pv, pt = np.mean(pv_all, 0), np.mean(pt_all, 0)
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
    print(f"bags keep {KEEP_FRAC:.0%} of N and PD, and 100 % of ET\n", flush=True)

    ARMS = (("baseline (3 seeds)", 3, "seed"),
            ("6 seeds, full data", 6, "seed"),
            ("6 bags, 70 % majority", 6, "bag"))
    res = {a: [] for a, _, _ in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        for lab, n, mode in ARMS:
            trs = members(y, tr, n, mode, seed0=sp)
            res[lab].append(fit_ensemble(spec, D, traj, y, trs, va, te))
        sizes = [len(t) for t in members(y, tr, 6, "bag", seed0=sp)]
        print(f"  split {sp+1}/{SPLITS}  train {len(tr)}, bag sizes "
              f"{min(sizes)}-{max(sizes)}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>24}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, _, _ in ARMS:
        print(f"{lab:>24}" + "".join(f"{v:>9.3f}" for v in res[lab].mean(0))
              + f"{res[lab][:, 3].std():>12.3f}")

    base = res["baseline (3 seeds)"]
    print("\npaired vs the reported model (3 seeds):")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nbags vs seeds at the same ensemble size — is it the SUBSAMPLING?")
    for (dd, lo, hi), c in zip(paired(res["6 bags, 70 % majority"],
                                      res["6 seeds, full data"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nsplit-level win rate, bags vs baseline:")
    b = res["6 bags, 70 % majority"]
    for i, c in enumerate(NM):
        print(f"    {c:>8} {float((b[:, i] > base[:, i]).mean()):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
