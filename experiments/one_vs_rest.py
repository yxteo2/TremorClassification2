"""One-vs-rest decomposition: does a dedicated ET detector beat the softmax head?

Every model in this project is a **3-class softmax**. Its ET column is fitted
against N and PD simultaneously, sharing a single set of trunk features and a
single normalisation across the three logits. That is one way to build a
3-class decision rule, not the only one, and it has a specific cost for the
scarce class: the gradient that shapes the ET logit is diluted by the two
majority columns, and the softmax's shared normalisation ties ET's confidence
to whatever PD's logit is doing on the same patient.

One-vs-rest trains **three independent binary models** and combines them after
the fact:

    N-vs-rest    167 positives, 237 negatives
    PD-vs-rest   188 positives, 216 negatives
    ET-vs-rest    49 positives, 355 negatives

The ET detector is the point. In the softmax it competes for capacity; here it
gets its own trunk, its own early stopping, and its own class weighting, and it
is trained on **all 404 patients** rather than the 237 tremor patients a
PD-vs-ET model would see. `pd_vs_et_transfer.md` established that PD-vs-ET is
the hard sub-problem; ET-vs-rest is a different framing of the same boundary
with 70 % more negatives to shape it.

**This has never been tried here.** The project's binary work
(`shortwindow_twostage.py`, `axis_specific_inputs.py`) always split N-vs-tremor
first and then PD-vs-ET. Nothing has trained ET against the full remainder.

## Why it could just as easily fail

The three binary scores are **not calibrated against each other** — each is a
sigmoid-ish softmax over its own two classes, fitted with its own balanced class
weights, so their scales differ. Combining them by argmax is only meaningful
after calibration. Two things absorb that here: row-normalisation, and the
existing `tune_offsets` prior search, which is fitted on the untouched
validation split exactly as in the reported model.

Also, OvR triples the number of fitted models, and this project has repeatedly
found that **more models is itself a change** (`balanced_bagging.py`). The
baseline arm therefore uses the identical 3-seed × 2-family recipe, and the
comparison to watch is not "OvR beats baseline" in isolation but whether the
gain survives the same ensemble-size confound that bagging is testing.

## Arms, merged 3-class protocol, 20 splits, paired

  3-class softmax   the reported model: 2 families x 3 seeds, priors on val
  one-vs-rest       3 binaries x 2 families x 3 seeds, normalised, priors on val
  blend             geometric mix of the two, weight chosen on VALIDATION only

The blend weight is model selection on validation, the same status as the class
priors. Test is never touched by any of it.

Run: ``python -m experiments.one_vs_rest``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL, build
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS, SEEDS = 20, (0, 1, 2)
WGRID = np.linspace(0.0, 1.0, 11)      # 0 = pure softmax, 1 = pure one-vs-rest


def _families(spec, desc, traj, nc):
    """The two model families, identical to the reported model but nc-class."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL, num_classes=nc)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=nc, ch=16)
    return ((packed, mk1), (spec, mk2))


def fit_head(spec, desc, traj, yy, tr, va, te, nc):
    """Average of 2 families x 3 seeds. Returns (p_val, p_test), each (n, nc)."""
    pv_l, pt_l = [], []
    for X, mk in _families(spec, desc, traj, nc):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, yy[tr], (X[va] - mu) / sd, yy[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s, nc=nc)
             for s in SEEDS]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def _norm(P):
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def _val_macro_f1(pv, off, yva):
    _, _, F, _ = precision_recall_fscore_support(
        yva, (np.log(pv + 1e-12) + off).argmax(1), labels=[0, 1, 2],
        zero_division=0)
    return F.mean()


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print("one-vs-rest positives per detector: "
          f"N {int((y==0).sum())}, PD {int((y==1).sum())}, "
          f"ET {int((y==2).sum())} against {int((y!=2).sum())} negatives\n",
          flush=True)

    ARMS = ("3-class softmax", "one-vs-rest", "blend (w on val)")
    res = {a: [] for a in ARMS}
    aucs, ws = [], []

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        pv3, pt3 = fit_head(spec, D, traj, y, tr, va, te, 3)
        pv3, pt3 = _norm(pv3), _norm(pt3)

        cv, ct = [], []
        for c in (0, 1, 2):
            yb = (y == c).astype(int)
            a, b = fit_head(spec, D, traj, yb, tr, va, te, 2)
            cv.append(a[:, 1])
            ct.append(b[:, 1])
        pvO, ptO = _norm(np.stack(cv, 1)), _norm(np.stack(ct, 1))

        # diagnostic: how good is the dedicated ET detector, on its own terms?
        yet = (y[te] == 2).astype(int)
        aucs.append([roc_auc_score(yet, ptO[:, 2]),
                     roc_auc_score(yet, pt3[:, 2])])

        off3 = tune_offsets(pv3, y[va])
        offO = tune_offsets(pvO, y[va])
        res["3-class softmax"].append(score(pt3, off3, y[te]))
        res["one-vs-rest"].append(score(ptO, offO, y[te]))

        # blend: weight AND offsets chosen on validation, test untouched
        best = (-1.0, 0.0, off3)
        for w in WGRID:
            bv = _norm(np.exp((1 - w) * np.log(pv3 + 1e-12)
                              + w * np.log(pvO + 1e-12)))
            o = tune_offsets(bv, y[va])
            f = _val_macro_f1(bv, o, y[va])
            if f > best[0]:
                best = (f, w, o)
        _, w, o = best
        bt = _norm(np.exp((1 - w) * np.log(pt3 + 1e-12)
                          + w * np.log(ptO + 1e-12)))
        res["blend (w on val)"].append(score(bt, o, y[te]))
        ws.append(w)

        print(f"  split {sp+1}/{SPLITS}  ET-detector AUC ovr {aucs[-1][0]:.3f} "
              f"vs softmax {aucs[-1][1]:.3f}   blend w={w:.1f}", flush=True)

    for a in res:
        res[a] = np.array(res[a])
    aucs = np.array(aucs)

    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>20}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["3-class softmax"]
    print("\npaired vs the 3-class softmax, same splits:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print(f"\nET-vs-rest detector AUC {aucs[:,0].mean():.3f} "
          f"vs the softmax ET column {aucs[:,1].mean():.3f}  "
          f"(paired diff {np.mean(aucs[:,0]-aucs[:,1]):+.3f})")
    print(f"blend weight chosen on validation: mean {np.mean(ws):.2f}, "
          f"w=0 in {int(np.sum(np.array(ws)==0))}/{SPLITS} splits, "
          f"w=1 in {int(np.sum(np.array(ws)==1))}/{SPLITS}")

    print("\nsplit-level win rate vs the softmax:")
    for a in ARMS[1:]:
        print(f"  {a}: " + "  ".join(
            f"{c} {float((res[a][:, i] > base[:, i]).mean()):.2f}"
            for i, c in enumerate(NM)))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
