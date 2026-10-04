"""Newer fine-tuning methods on the 2015 OUT transfer model (`ft`).

`transfer_2015.md`: fine-tuning longer **loses** ET (−0.021 \\*) -- drifting from
the PADS-pretrained weights costs what PADS taught; L2-SP (penalising drift)
raised ranking, not precision; head-only fine-tuning was too weak. Each arm below
changes ONE thing about how `ft` is fine-tuned, aimed at that drift or at the
noisy best-epoch choice on ~30 validation patients:

    ft      the reported recipe (must reproduce saved `ft` bit for bit)
    lpft    LP-FT (Kumar et al., ICLR 2022): 40 epochs head-only (BN frozen),
            then the usual full fine-tune
    wise    WiSE-FT (Wortsman et al., CVPR 2022): 0.5 x pretrained + 0.5 x
            fine-tuned weights (the PADS head has the same 3 classes)
    swa     SWA: uniform average of the weights over fine-tune epochs 41-80,
            BN statistics recomputed on the training fold -- replaces the
            best-validation-epoch checkpoint
    sam     SAM (Foret et al., ICLR 2021), rho 0.05, during fine-tuning
    ls      label smoothing 0.1 on the fine-tuning loss

All arms share each seed's PADS pretraining (identical to `common.protocol.train`)
and branch at fine-tuning; partitions, validation splits, members, seeds,
standardisation and offsets are `transfer_2015`'s. The pretrain/fine-tune loop
is a copy of `train()` with hooks, so the `ft` arm is asserted bit-exact against
`segments_2015_runs` before anything is reported.

## Prediction, recorded before the run

**No method raises ET precision significantly** (this project's training-side
record: logit adjustment, L2-SP, longer/shorter training all null on precision).
**WiSE-FT and LP-FT raise PD-vs-ET AUC by +0.01 to +0.02** (they limit drift,
like L2-SP's +0.012-0.018); SWA reduces split-to-split spread but not the mean;
SAM null; label smoothing null or slightly negative on ET.

Run: ``REPS=0-14 python -m experiments.training_methods_2015`` ... then
``python -m experiments.training_methods_2015 report``.
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
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ft", "lpft", "wise", "swa", "sam", "ls")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "training_methods_2015_runs")
EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


def _head(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def _finetune(m, xt, yt, xv, yv, w, variant, pre_state):
    """common.protocol.train's fine-tuning loop, with one variant hook."""
    params = list(m.parameters())
    if variant == "lpft":                         # stage 1: linear probe
        head = _head(m)
        for p in m.parameters():
            p.requires_grad_(False)
        for p in head.parameters():
            p.requires_grad_(True)
        bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        op = torch.optim.AdamW(list(head.parameters()), lr=FT_LR, weight_decay=WD)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, 40)
        for _ in range(40):
            m.train(); op.zero_grad()
            for b in bns:
                b.eval()
            nn.CrossEntropyLoss(weight=w)(m(xt), yt).backward(); op.step(); sc.step()
        for p in m.parameters():
            p.requires_grad_(True)
    lf_train = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1 if variant == "ls" else 0.0)
    lf_val = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(params, lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    best, state = np.inf, None
    avg, n_avg = None, 0
    for ep in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        loss = lf_train(m(xt), yt)
        loss.backward()
        if variant == "sam":                      # ascend to the worst neighbour
            with torch.no_grad():
                g = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
                eps = [(p, 0.05 * p.grad / (g + 1e-12)) for p in params if p.grad is not None]
                for p, e in eps:
                    p.add_(e)
            opt.zero_grad()
            lf_train(m(xt), yt).backward()
            with torch.no_grad():
                for p, e in eps:
                    p.sub_(e)
        opt.step(); sch.step()
        if variant == "swa" and ep >= FT_EPOCHS // 2:
            sd = {k: t.detach().clone() for k, t in m.state_dict().items()}
            if avg is None:
                avg = {k: v.float().clone() for k, v in sd.items()}
            else:
                for k in avg:
                    avg[k] += sd[k].float()
            n_avg += 1
        m.eval()
        with torch.no_grad():
            v = float(lf_val(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if variant == "swa":
        ref = m.state_dict()
        m.load_state_dict({k: (avg[k] / n_avg).to(ref[k].dtype) for k in avg})
        bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
        for b in bns:                             # recompute BN stats, full batch
            b.reset_running_stats(); b.momentum = None
        m.train()
        with torch.no_grad():
            m(xt)
        for b in bns:
            b.momentum = 0.1
    elif state:
        m.load_state_dict(state)
    if variant == "wise":
        ft = m.state_dict()
        m.load_state_dict({k: ((0.5 * pre_state[k].float() + 0.5 * ft[k].float())
                               .to(ft[k].dtype) if ft[k].is_floating_point() else ft[k])
                           for k in ft})
    m.eval()
    return m


def train_variants(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed):
    """Pretrain once exactly as common.protocol.train, then fine-tune each arm."""
    torch.manual_seed(seed)
    xt, yt = _T(Xtr), torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    xv, yv = _T(Xva), torch.tensor(yva, dtype=torch.long, device=DEVICE)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    xp, ypt = _T(Xp), torch.tensor(yp, dtype=torch.long, device=DEVICE)
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
    pre_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    rng = torch.get_rng_state()
    w = _wt(ytr)
    out = {}
    for v in ARMS:                                # ft first: same RNG as train()
        torch.set_rng_state(rng)
        mv = copy.deepcopy(m)
        mv = _finetune(mv, xt, yt, xv, yv, w, v, pre_state)
        with torch.no_grad():
            out[v] = [torch.softmax(mv(_T(z)), 1).cpu().numpy() for z in Xout]
    return out


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    print(f"device={DEVICE}  2015 OUT n={len(y)}  arms={ARMS}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
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
            pv = {k: [] for k in ARMS}
            pt = {k: [] for k in ARMS}
            for key, mk in (("packed", mk1), ("spec", mk2)):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                per = [train_variants(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                      [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]), s)
                       for s in (0, 1, 2)]
                for k in ARMS:
                    pv[k].append(np.mean([r[k][0] for r in per], 0))
                    pt[k].append(np.mean([r[k][1] for r in per], 0))
            for k in ARMS:
                v, t = np.mean(pv[k], 0), np.mean(pt[k], 0)
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        ref = f"segments_2015_runs/rep{rep:02d}.npz"
        if os.path.exists(ref):
            r = np.load(ref)
            assert np.array_equal(r["ft"], pred["ft"]) and np.allclose(r["p_ft"], prob["ft"]), \
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
    print(f"2015 OUT, fine-tuning methods on ft, {n} repeats\n")
    print(f"{'arm':>6}" + "".join(f"{c:>9}" for c in NAMES) + "  sd(precET)")
    for k in ARMS:
        print(f"{k:>6}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0))
              + f"  {R[k][:, 2].std():.3f}")
    for a in ARMS[1:]:
        dd = R[a] - R["ft"]
        print(f"\n{a} - ft  (paired, {n} repeats)")
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
