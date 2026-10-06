"""Which fine-tuning epoch should the 2015 transfer model keep?

`loss_curves_2015` (repeats 0-4, 150 models): fine-tuning validation loss is
flat after ~40 epochs, so "best validation loss" on ~30 patients picks a
near-random epoch (IQR 12-62 two-stream, 15-76 TCN, bimodal). The decision
metric disagrees with the loss: TCN test macro-F1 peaks at epochs 10-20 (0.564)
and falls to 0.537 while its loss still improves; two-stream F1 is highest at the
end (0.513). The chosen checkpoints score 0.542 / 0.503.

Every arm comes from the SAME training trajectory -- weights are snapshotted at
the epochs below -- so the only difference is which weights are kept:

    ckpt         best-validation-loss epoch for both members   (= `ft`)
    last         epoch 80 for both
    tuned        two-stream epoch 80, TCN epoch 20
    tuned_ckpt   two-stream best-val epoch, TCN epoch 20

**Selection caveat, handled by design:** epochs 20 and 80 were read off curves
that include test data of repeats 0-4. Those repeats are excluded from the
evaluation; the report uses repeats 5-39 and the fresh partitions 100-139.

## Prediction, recorded before the run

**tuned beats ckpt on macroF1 by +0.005 to +0.015** on both partition sets,
precET not significantly different (the TCN's F1 peak is a 3-class effect, and
it is one of two members); **last ~= ckpt** (both pick late-ish epochs on a flat
curve).

Run: ``REPS=5-17 python -m experiments.checkpoint_rule_2015`` ... and
``REPS=100-140`` ...; then ``python -m experiments.checkpoint_rule_2015 report``.
"""

from __future__ import annotations

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
                                               _T, _wt)
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ckpt", "last", "tuned", "tuned_ckpt")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "checkpoint_rule_2015_runs")
SNAP = 20                    # TCN snapshot epoch (1-based count of updates)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def snapshots(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed):
    """common.protocol.train with pretraining; returns probs for 3 weight choices."""
    torch.manual_seed(seed)
    xt, yt, xv, yv = _T(Xtr), _L(ytr), _T(Xva), _L(yva)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    xp, ypt = _T(Xp), _L(yp)
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
    lw = nn.CrossEntropyLoss(weight=_wt(ytr))
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    best, state, snap = np.inf, None, None
    for e in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        lw(m(xt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lw(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        if e + 1 == SNAP:
            snap = {k: t.detach().clone() for k, t in m.state_dict().items()}

    def probs(sd=None):
        if sd is not None:
            m.load_state_dict(sd)
        m.eval()
        with torch.no_grad():
            return [torch.softmax(m(_T(z)), 1).cpu().numpy() for z in Xout]

    last = probs()
    return {"last": last, "snap": probs(snap), "ckpt": probs(state)}


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEVICE}  2015 n={len(y)}", flush=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:03d}.npz"):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest_i, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest_i, test_size=0.25, stratify=y[rest_i],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            mem = {}
            for name, key, mk in (("two", "packed", mk1), ("tcn", "spec", mk2)):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                runs = [snapshots(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                  [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]), s)
                        for s in (0, 1, 2)]
                mem[name] = {c: [np.mean([r[c][i] for r in runs], 0) for i in (0, 1)]
                             for c in ("ckpt", "last", "snap")}
            choice = {"ckpt": ("ckpt", "ckpt"), "last": ("last", "last"),
                      "tuned": ("last", "snap"), "tuned_ckpt": ("ckpt", "snap")}
            for k, (c2, ct) in choice.items():
                pv = (mem["two"][c2][0] + mem["tcn"][ct][0]) / 2
                pt = (mem["two"][c2][1] + mem["tcn"][ct][1]) / 2
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        ref = (f"segments_2015_runs/rep{rep:02d}.npz" if rep < 100
               else f"fusion_2015_confirm/rep{rep:02d}.npz")
        if os.path.exists(ref) and len(np.load(ref)["y"]) != len(y):
            print(f"rep {rep}: saved reference predates the PD 12 id fix "
                  f"(151 rows vs {len(y)}); bit-exact check skipped", flush=True)
        elif os.path.exists(ref):
            assert np.array_equal(np.load(ref)["ft"], pred["ckpt"]), \
                f"rep {rep}: ckpt arm does not reproduce ft"
            print(f"rep {rep}: ckpt arm reproduces ft exactly", flush=True)
        np.savez(f"{OUT_DIR}/rep{rep:03d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()})


def report():
    for tag, lo_, hi_ in (("selection partitions, repeats 5-39 (0-4 excluded: "
                           "used to read the curves)", 5, 40),
                          ("fresh partitions 100-139", 100, 140)):
        files = [f for f in sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
                 if lo_ <= int(f.split("rep")[-1][:-4]) < hi_]
        if not files:
            continue
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
        print(f"\n=== {tag}: {n} repeats ===")
        print(f"{'arm':>11}" + "".join(f"{c:>9}" for c in NAMES))
        for k in ARMS:
            print(f"{k:>11}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)))
        for a in ARMS[1:]:
            dd = R[a] - R["ckpt"]
            cells = []
            for i in range(len(NAMES)):
                bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                      for s in range(4000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                cells.append(f"{NAMES[i]} {dd[:, i].mean():+.3f}"
                             f"{'*' if lo > 0 or hi < 0 else ' '}")
            print(f"  {a + ' - ckpt':<18} " + "  ".join(cells))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "5-40").split("-"))
        run(range(a, b))
