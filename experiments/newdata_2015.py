"""Integrate NewData into the 2015 OUT transfer model -- three untested ways.

PADS is already in the model: `ft` pretrains on PADS StretchHold, then fine-tunes
on 2015 (`transfer_2015.md`). NewData (56 patients: 27 HC / 23 PD / 6 ET, the
same OUT task, Moveo sensors at the same three arm positions) has been tried
once: added to PADS in pretraining (`ft_pn`, precET −0.014, n.s.). Untested:

    ft            PADS pretrain -> 2015 fine-tune     (asserted bit-exact)
    nd_pool       PADS pretrain -> fine-tune on 2015 training patients + ALL
                  NewData patients (2015 is still the only test set)
    nd_pool_shuf  the same with NewData's labels permuted, re-drawn per repeat
                  -- attribution: NewData's labels vs extra rows / regularisation
    nd_seq        PADS pretrain -> NewData stage (80 epochs, lr 1e-3, last
                  epoch) -> the usual 2015 fine-tune: far domain to near

NewData is z-scored on its own statistics (as `transfer_2015` does for PADS)
and its 4 asymmetry columns and have-flag are zeroed to match 2015, which has
one arm, so "has two arms" cannot identify the cohort. All arms branch from the
same PADS-pretrained weights per seed (the `training_methods_2015` loop, whose
`ft` arm reproduces `transfer_2015` bit for bit).

## Prediction, recorded before the run

**NewData adds nothing significant to 2015 ET precision** in any arm: its 6 ET
look like its PD on every tremor feature (`inhouse_pd_vs_et.md`), and NewData
in pretraining was null. **nd_pool vs nd_pool_shuf: |precET| < 0.04, n.s.**; a
small N/PD effect either way (NewData's HC / PD are the in-house phenotype).
**nd_seq null.**

Run: ``REPS=0-14 python -m experiments.newdata_2015`` ... then
``python -m experiments.newdata_2015 report``.
"""

from __future__ import annotations

import copy
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.training_methods_2015 import (EPOCHS, FT_EPOCHS, FT_LR, LR, WD,
                                               _T, _finetune, _wt)
from experiments.transfer_2015 import CAP, members, zfit

ARMS = tuple(os.environ.get("ND_ARMS", "ft,nd_pool,nd_pool_shuf,nd_seq").split(","))
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "newdata_2015_runs")


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def branches(model_fn, Xtr, ytr, Xva, yva, Xout, pre, nd, nd_shuf, seed):
    """PADS-pretrain once (as common.protocol.train), then one branch per arm."""
    torch.manual_seed(seed)
    xt, yt = _T(Xtr), _L(ytr)
    xv, yv = _T(Xva), _L(yva)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    xp, ypt = _T(Xp), _L(yp)
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
    pre_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    rng = torch.get_rng_state()
    Xn, yn = nd
    out = {}
    for arm in ARMS:                              # ft first: same RNG as train()
        torch.set_rng_state(rng)
        mv = copy.deepcopy(m)
        if arm == "ft":
            mv = _finetune(mv, xt, yt, xv, yv, _wt(ytr), "ft", pre_state)
        elif arm in ("nd_pool", "nd_pool_shuf"):
            yy = np.concatenate([ytr, yn if arm == "nd_pool" else nd_shuf])
            mv = _finetune(mv, torch.cat([xt, _T(Xn)]), _L(yy), xv, yv, _wt(yy),
                           "ft", pre_state)
        else:                                     # nd_seq
            o2 = torch.optim.AdamW(mv.parameters(), lr=FT_LR, weight_decay=WD)
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, FT_EPOCHS)
            l2 = nn.CrossEntropyLoss(weight=_wt(yn))
            xn, ynt = _T(Xn), _L(yn)
            mv.train()
            for _ in range(FT_EPOCHS):
                o2.zero_grad(); l2(mv(xn), ynt).backward(); o2.step(); s2.step()
            mv = _finetune(mv, xt, yt, xv, yv, _wt(ytr), "ft", pre_state)
        with torch.no_grad():
            out[arm] = [torch.softmax(mv(_T(z)), 1).cpu().numpy() for z in Xout]
    return out


def run(reps):
    torch.set_num_threads(1)
    A, B, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    bd = B[1].copy()
    bd[:, 10:] = 0.0                              # asymmetry + have-flag, as 2015
    ndp = {"packed": np.hstack([B[0], bd, B[2]]), "spec": B[0]}
    yn = B[3]
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    print(f"device={DEVICE}  2015 n={len(y)}  NewData n={len(yn)} "
          f"{np.bincount(yn).tolist()}  arms={ARMS}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        yn_shuf = np.random.default_rng(20000 + rep).permutation(yn)
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest_i, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest_i, test_size=0.25, stratify=y[rest_i],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            pv = {k: [] for k in ARMS}
            pt = {k: [] for k in ARMS}
            for key, mk in (("packed", mk1), ("spec", mk2)):
                X, P, N = own[key], pads[key], ndp[key]
                f = zfit(X[tr])
                per = [branches(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]),
                                (zfit(N)(N), yn), yn_shuf, s)
                       for s in (0, 1, 2)]
                for k in ARMS:
                    pv[k].append(np.mean([r[k][0] for r in per], 0))
                    pt[k].append(np.mean([r[k][1] for r in per], 0))
            for k in ARMS:
                v, t = np.mean(pv[k], 0), np.mean(pt[k], 0)
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        ref = (f"segments_2015_runs/rep{rep:02d}.npz" if rep < 100
               else f"fusion_2015_confirm/rep{rep:02d}.npz")
        if os.path.exists(ref) and "ft" in ARMS:
            r = np.load(ref)
            assert np.array_equal(r["ft"], pred["ft"]), \
                f"rep {rep}: ft arm does not reproduce transfer_2015's ft"
            print(f"rep {rep}: ft arm reproduces saved ft exactly", flush=True)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()})
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {Pr[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    for f in files:
        d = np.load(f)
        y = d["y"]
        m = y != 0
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[k], labels=[0, 1, 2],
                                                         zero_division=0)
            pr = d[f"p_{k}"]
            s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
            o = np.argsort(-pr[:, 2])
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean(),
                           roc_auc_score((y[m] == 2).astype(int), s),
                           (y[o[:5]] == 2).mean()])
    n = len(files)
    R = {k: np.array(v) for k, v in res.items()}
    print(f"2015 OUT test patients; NewData in training, {n} repeats\n")
    print(f"{'arm':>13}" + "".join(f"{c:>9}" for c in NAMES))
    for k in ARMS:
        print(f"{k:>13}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)))
    for a, b in (("nd_pool", "ft"), ("nd_pool", "nd_pool_shuf"),
                 ("nd_pool_shuf", "ft"), ("nd_seq", "ft")):
        if a not in R or b not in R:
            continue
        dd = R[a] - R[b]
        print(f"\n{a} - {b}  (paired, {n} repeats)")
        for i, c in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
