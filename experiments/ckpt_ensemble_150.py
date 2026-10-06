"""Checkpoint ensembling and validation-free early stopping, one trajectory.

`loss_curves_2015.md`: fine-tuning validation loss (~30 patients) is flat after
~40 epochs, so the best-validation-loss checkpoint that `ft` keeps is a
near-random draw from that plateau (chosen-epoch IQR 12-62 / 15-76, bimodal);
TCN test macro-F1 peaks at epochs 10-20 while loss keeps improving.
`training_methods_2015.md`: SWA (weight average of epochs 41-80) HURT (precET
-0.049 \\*); `checkpoint_rule_2015`: the two-stream member's last epoch hurt.

Every arm below comes from the SAME fine-tuning trajectory per member and seed
(the `common.protocol.train` recipe: PADS pretrain 200 epochs, fine-tune 80);
softmax outputs on validation and test are recorded at every fine-tuning epoch,
and the arms differ only in which epochs' outputs are kept. All rules are fixed
here, before the run; none looks at test data.

    ft     best-validation-loss epoch (the current recipe; first minimum)
    top5   mean softmax (validation AND test) over the 5 epochs with the lowest
           validation loss -- prediction ensembling (Chen et al. 2017,
           Anderson-Conway et al. 2022), NOT weight averaging
    win    mean softmax over every epoch whose validation loss is within 2 % of
           the minimum (v <= 1.02 * min)
    gd     gradient-disparity early stopping (Forouzesh & Thiran, ECML-PKDD
           2021), no validation set involved: after each epoch's update, the L2
           distance between the gradients of the (class-weighted, as trained)
           loss on two random stratified halves of the TRAINING fold, averaged
           over 3 random half-splits drawn once per run with a fixed seed;
           computed in eval mode (no dropout, BN running statistics), so it has
           no side effect on the trajectory or the RNG; keep the epoch at the
           centre of the 5-epoch window (centred moving average) whose mean
           disparity is minimal

As in the recipe, per-seed outputs are averaged over 3 seeds, then over the two
members (two-stream on spec+desc+traj, ResidualTCN on spec); per-class logit
offsets (`tune_offsets`) are tuned on EACH arm's own validation probabilities.
Protocol = `transfer_2015`: StratifiedKFold(5, shuffle, random_state=rep), 25 %
stratified inner validation (random_state rep*10+fold), PADS capped 90/class
redrawn per repeat, `zfit` per domain. Corrected loader: 150 patients
(61 N / 74 PD / 15 ET). Saved runs used 151 rows, so the `ft` arm here is this
run's own baseline; `python -m experiments.ckpt_ensemble_150 check` asserts the
`ft` arm is bit-identical to `common.protocol.train(pre=...)` on one fold.

## Prediction, recorded before the run

* top5 and win: a variance reduction over a near-random checkpoint, not a new
  decision rule. PD-vs-ET AUC +0.005 to +0.015 vs ft (may reach repeat-level
  significance), precET |delta| < 0.03 and not significant, macroF1 within
  +-0.01. Nothing significant at the PATIENT level (resolution ~+-0.03).
  `win` averages a median >= 10 epochs.
* gd: in full-batch training both halves are trained on, so their gradients
  shrink together and the disparity falls ~monotonically; gd picks late epochs
  (median >= 60) and behaves like `last` in `checkpoint_rule_2015`: AUC -0.01 to
  -0.02 vs ft, precET null-to-negative. Not adopted.

Run: ``REPS=0-20 python -m experiments.ckpt_ensemble_150`` (resumable), then
``python -m experiments.ckpt_ensemble_150 report``.
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
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ft", "top5", "win", "gd")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "ckpt_ensemble_150_runs")
EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80
N_GD_SPLITS, GD_SEED, GD_WIN, TOPK, WIN_TOL = 3, 0, 5, 5, 0.02


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


def _halves(ytr):
    """N_GD_SPLITS stratified random half-splits of the training fold (fixed seed)."""
    rng = np.random.default_rng(GD_SEED)
    out = []
    for _ in range(N_GD_SPLITS):
        a, b = [], []
        for c in np.unique(ytr):
            idx = rng.permutation(np.flatnonzero(ytr == c))
            a.append(idx[:len(idx) // 2]); b.append(idx[len(idx) // 2:])
        out.append((torch.as_tensor(np.sort(np.concatenate(a)), device=DEVICE),
                    torch.as_tensor(np.sort(np.concatenate(b)), device=DEVICE)))
    return out


def _disparity(m, xt, yt, lw, halves):
    """Mean over half-splits of ||grad L(half A) - grad L(half B)||_2, eval mode."""
    m.eval()
    params = [p for p in m.parameters() if p.requires_grad]
    ds = []
    for a, b in halves:
        ga = torch.autograd.grad(lw(m(xt[a]), yt[a]), params, allow_unused=True)
        gb = torch.autograd.grad(lw(m(xt[b]), yt[b]), params, allow_unused=True)
        sq = sum(((x - y) ** 2).sum() for x, y in zip(ga, gb)
                 if x is not None and y is not None)
        ds.append(float(torch.sqrt(sq)))
    return float(np.mean(ds))


def trajectory(model_fn, Xtr, ytr, Xva, yva, Xte, pre, seed):
    """common.protocol.train(pre=...) with every fine-tuning epoch's outputs logged.

    Returns val loss (E,), disparity (E,), val probs (E, nva, 3), test probs (E, nte, 3).
    """
    torch.manual_seed(seed)
    xt, yt, xv, yv, xte = _T(Xtr), _L(ytr), _T(Xva), _L(yva), _T(Xte)
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
    opt = torch.optim.AdamW(list(m.parameters()), lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    halves = _halves(ytr)
    vl, gd, pv, pt = [], [], [], []
    for _ in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        lw(m(xt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            zv = m(xv)
            vl.append(float(lw(zv, yv)))
            pv.append(torch.softmax(zv, 1).cpu().numpy())
            pt.append(torch.softmax(m(xte), 1).cpu().numpy())
        gd.append(_disparity(m, xt, yt, lw, halves))
    return np.array(vl), np.array(gd), np.array(pv), np.array(pt)


def select(vl, gd):
    """Epoch index sets per arm (fixed rules; no test data)."""
    order = np.argsort(vl, kind="stable")
    ma = np.convolve(gd, np.ones(GD_WIN) / GD_WIN, mode="valid")
    return {"ft": np.array([int(np.argmin(vl))]),
            "top5": order[:TOPK],
            "win": np.flatnonzero(vl <= (1 + WIN_TOL) * vl.min()),
            "gd": np.array([int(np.argmin(ma)) + GD_WIN // 2])}


def _pads(C, rep):
    rng = np.random.default_rng(rep)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                              min(CAP, int((C[3] == c).sum())),
                                              replace=False) for c in (0, 1, 2)]))


def _folds(spec, y, rep):
    for fold, (rest_i, te) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
        tr, va = train_test_split(rest_i, test_size=0.25, stratify=y[rest_i],
                                  random_state=rep * 10 + fold)
        yield fold, np.sort(tr), np.sort(va), te


def check():
    """Assert the `ft` arm equals common.protocol.train(pre=...) bit for bit."""
    from common.protocol import train
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    kp = _pads(C, 0)
    pads = {"packed": Cp[kp], "spec": C[0][kp]}
    _, tr, va, te = next(_folds(spec, y, 0))
    for key, mk in (("packed", mk1), ("spec", mk2)):
        X, P = own[key], pads[key]
        f = zfit(X[tr])
        pre = (zfit(P)(P), C[3][kp])
        ref = train(mk, f(X[tr]), y[tr], f(X[va]), y[va], [f(X[va]), f(X[te])],
                    seed=0, pre=pre)
        vl, gd, pv, pt = trajectory(mk, f(X[tr]), y[tr], f(X[va]), y[va], f(X[te]), pre, 0)
        e = select(vl, gd)["ft"][0]
        assert np.array_equal(ref[0], pv[e]) and np.array_equal(ref[1], pt[e]), key
        print(f"{key}: ft arm bit-identical to common.protocol.train (epoch {e}); "
              f"gd epoch {select(vl, gd)['gd'][0]}, win size {len(select(vl, gd)['win'])}",
              flush=True)
    print("CHECK_OK", flush=True)


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    print(f"device={DEVICE}  2015 OUT n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  arms={ARMS}", flush=True)
    assert len(y) == 150, len(y)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        kp = _pads(C, rep)
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        ep_ft = np.zeros((5, 2, 3), int); ep_gd = np.zeros((5, 2, 3), int)
        n_win = np.zeros((5, 2, 3), int)
        curves_vl = np.zeros((5, 2, 3, FT_EPOCHS)); curves_gd = np.zeros((5, 2, 3, FT_EPOCHS))
        for fold, tr, va, te in _folds(spec, y, rep):
            pv = {k: [] for k in ARMS}
            pt = {k: [] for k in ARMS}
            for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                sv = {k: [] for k in ARMS}
                st = {k: [] for k in ARMS}
                for s in (0, 1, 2):
                    vl, gd, PV, PT = trajectory(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                                f(X[te]), (zfit(P)(P), C[3][kp]), s)
                    sel = select(vl, gd)
                    ep_ft[fold, j, s], ep_gd[fold, j, s] = sel["ft"][0], sel["gd"][0]
                    n_win[fold, j, s] = len(sel["win"])
                    curves_vl[fold, j, s], curves_gd[fold, j, s] = vl, gd
                    for k in ARMS:
                        sv[k].append(PV[sel[k]].mean(0)); st[k].append(PT[sel[k]].mean(0))
                for k in ARMS:
                    pv[k].append(np.mean(sv[k], 0)); pt[k].append(np.mean(st[k], 0))
            for k in ARMS:
                v, t = np.mean(pv[k], 0), np.mean(pt[k], 0)
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()},
                 ep_ft=ep_ft, ep_gd=ep_gd, n_win=n_win, vl=curves_vl, gdc=curves_gd)
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {Pr[k][2]:.2f}" for k in ARMS)
              + f"  | ep ft med {np.median(ep_ft):.0f} gd med {np.median(ep_gd):.0f}"
              f" win med {np.median(n_win):.0f}", flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    E = {"ep_ft": [], "ep_gd": [], "n_win": []}
    for f in files:
        d = np.load(f)
        y = d["y"]
        m = y != 0
        for k in E:
            E[k].append(d[k])
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
    print(f"2015 OUT (n={len(y)}), checkpoint ensembling / GD stopping, {n} repeats\n")
    print(f"{'arm':>6}" + "".join(f"{c:>9}" for c in NAMES) + "  sd(precET)  ET preds/rep")
    for k in ARMS:
        npred = np.mean([(np.load(f)[k] == 2).sum() for f in files])
        print(f"{k:>6}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0))
              + f"  {R[k][:, 2].std():.3f}       {npred:.1f}")
    print("\nchosen fine-tuning epochs (0-based, of 80), per member [two-stream, TCN]:")
    for k in E:
        a = np.concatenate([x[None] for x in E[k]])          # (n, 5, 2, 3)
        for j, nm in enumerate(("two-stream", "TCN")):
            v = a[:, :, j].ravel()
            print(f"  {k:>6} {nm:>10}: median {np.median(v):.0f}  IQR "
                  f"{np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}")
    for a in ARMS[1:]:
        dd = R[a] - R["ft"]
        print(f"\n{a} - ft  (paired over repeats, {n})")
        for i, c in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}")
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    for a in ARMS[1:]:
        bootstrap_dir(OUT_DIR, a, "ft")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
