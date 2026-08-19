"""The class priors are tuned for macro F1, but the target metric is precision.

Validation-tuned class priors are the second-largest measured contributor in this
project -- "ET precision 0.475 -> 0.612, the single largest gain". They are fitted
by `common.protocol.tune_offsets`, whose docstring reads:

    \"\"\"Per-class logit offsets maximising VALIDATION macro F1.\"\"\"

Macro F1 is not what this project optimises. The standing instruction is
per-class **precision**, especially ET precision, and F1 spends half its weight on
recall. On a class with 49 patients out of 404, the offset that maximises F1 is
systematically more permissive than the one that maximises precision, because
recall is cheap to buy at the low-precision end.

Nothing else about the model changes: the offsets are applied to the network's
output logits *after* training, so one training run per split serves every
objective. That makes this exactly paired -- the arms differ only in which
offset the same probabilities receive -- and nearly free.

Objectives compared:

  macro F1                the current default
  macro precision         the target metric, directly
  macro P, guarded        macro precision, but an offset is rejected unless each
                          class is predicted at least half as often as its
                          validation prevalence. Precision alone can be gamed by
                          predicting one confident ET and nothing else; the guard
                          removes that corner without changing the objective.
  0.5*(macroP + macroF1)  a hedge, in case pure precision overfits ~11 validation
                          ET patients
  balanced accuracy       the recall-only counterpart, as a control on direction

Also swept: the grid. The current one is 9x9 over [-1, 1] for two free offsets,
so the spacing is 0.25 in logit units -- coarse enough that the optimum may sit
between grid points. A 21x21 grid over the same range costs nothing here because
no retraining is involved.

Run: ``python -m experiments.prior_objective``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train
from experiments.final_model import NBIN, SPLITS, TL, build
from models.architectures import (DescriptorFusion, ResidualTCN, Spectrum1DCNN,
                                  TRUNKS, TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")


# --------------------------------------------------------------------------- #
# Offset objectives. Each takes validation probabilities and labels, returns the
# scalar to maximise for a candidate offset.
# --------------------------------------------------------------------------- #
def _pr(yv, pred):
    P, R, F, _ = precision_recall_fscore_support(yv, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return P, R, F


def obj_f1(yv, pred, prev):
    return _pr(yv, pred)[2].mean()


def obj_prec(yv, pred, prev):
    return _pr(yv, pred)[0].mean()


def obj_prec_guarded(yv, pred, prev):
    """Macro precision, rejecting offsets that predict a class too rarely.

    Without the guard, an offset that predicts a single ET patient and gets it
    right scores precision 1.0 on that class. Requiring each class to be
    predicted at least half as often as its validation prevalence removes that
    corner. The objective itself is unchanged inside the feasible region.
    """
    frac = np.array([(pred == c).mean() for c in (0, 1, 2)])
    if np.any(frac < 0.5 * prev):
        return -1.0
    return _pr(yv, pred)[0].mean()


def obj_mix(yv, pred, prev):
    P, _, F, _ = _pr(yv, pred)
    return 0.5 * (P.mean() + F.mean())


def obj_bal(yv, pred, prev):
    return _pr(yv, pred)[1].mean()


OBJECTIVES = (("macro F1 (current)", obj_f1, 9),
              ("macro precision", obj_prec, 9),
              ("macro P, guarded", obj_prec_guarded, 9),
              ("0.5*(macroP+macroF1)", obj_mix, 9),
              ("balanced accuracy", obj_bal, 9),
              ("macro P, guarded, 21x21", obj_prec_guarded, 21),
              ("macro F1, 21x21", obj_f1, 21))


def tune(pv, yv, fn, grid):
    prev = np.array([(yv == c).mean() for c in (0, 1, 2)])
    lp = np.log(pv + 1e-12)
    best, bo = -np.inf, np.zeros(3)
    for b1 in np.linspace(-1, 1, grid):
        for b2 in np.linspace(-1, 1, grid):
            o = np.array([0.0, b1, b2])
            v = fn(yv, (lp + o).argmax(1), prev)
            if v > best:
                best, bo = v, o
    return bo


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(NM))]


def main():
    torch.set_num_threads(1)
    d = build()
    y, key, SPEC = d["y"], d["key"], d["SPEC"]
    D_desc = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    spec = SPEC["multitaper"]
    nd = D_desc.shape[1]
    packed = np.hstack([spec, D_desc, traj])

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print("model: multitaper + trajectory (the reported best), trained ONCE per")
    print("split; every objective re-uses the same probabilities.\n")

    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)

    res = {lab: [] for lab, _, _ in OBJECTIVES}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        lpt = np.log(pt + 1e-12)

        for lab, fn, grid in OBJECTIVES:
            off = tune(pv, y[va], fn, grid)
            pred = (lpt + off).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            res[lab].append([P[0], P[1], P[2], P.mean(), F.mean()])
        print(f"  split {sp+1}/{SPLITS} done", flush=True)

    for k in res:
        res[k] = np.array(res[k])

    print(f"\n{'objective':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, _, _ in OBJECTIVES:
        m = res[lab].mean(0)
        print(f"{lab:>26}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}")

    base = res["macro F1 (current)"]
    print("\npaired vs macro F1 (current), same splits and same trained models:")
    for lab, _, _ in OBJECTIVES[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
