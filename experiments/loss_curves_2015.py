"""Loss curves of the 2015 transfer model (`ft`): pretraining and fine-tuning.

Logs, per epoch, exactly what `common.protocol.train` computes, plus
diagnosis-only quantities it never uses for a decision:

pretraining on PADS (200 epochs)
    pads_train     weighted CE on the PADS training rows (the optimised loss)
    tr15 / va15    weighted CE on the 2015 training / validation folds -- the
                   model has not seen 2015; this is how well PADS alone transfers
fine-tuning on 2015 (80 epochs)
    ft_train       training loss (optimised)
    ft_val         validation loss (the checkpoint criterion)
    ft_test        held-out test loss          -- diagnosis only
    ft_test_f1     held-out macro-F1 at argmax  -- diagnosis only
    best_epoch     the epoch the checkpoint keeps

The loop is the `training_methods_2015` copy of `train()` with logging added;
its final probabilities are asserted equal to `train_variants(...)["ft"]` on
the first model, so the curves describe the model that is reported.

Run: ``REPS=0-5 python -m experiments.loss_curves_2015`` then
``python -m experiments.loss_curves_2015 plot``.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE
from experiments.own_data_10et import build
from experiments.training_methods_2015 import (EPOCHS, FT_EPOCHS, FT_LR, LR, WD,
                                               _T, _wt, train_variants)
from experiments.transfer_2015 import CAP, members, zfit

OUT_DIR = os.environ.get("OUT_DIR", "loss_curves_2015_runs")


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def logged(model_fn, Xtr, ytr, Xva, yva, Xte, yte, pre, seed):
    torch.manual_seed(seed)
    xt, yt, xv, yv = _T(Xtr), _L(ytr), _T(Xva), _L(yva)
    xe, ye = _T(Xte), _L(yte)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    xp, ypt = _T(Xp), _L(yp)
    w = _wt(ytr)
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    lw = nn.CrossEntropyLoss(weight=w)
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    pre_log = np.zeros((EPOCHS, 3))
    m.train()
    for e in range(EPOCHS):
        op.zero_grad(); loss = lf(m(xp), ypt); loss.backward(); op.step(); sp.step()
        # diagnostics in eval mode, then back to train mode -- BN running stats are
        # only updated by train-mode forwards, so this does not change training
        m.eval()
        with torch.no_grad():
            pre_log[e] = [float(loss), float(lw(m(xt), yt)), float(lw(m(xv), yv))]
        m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    ft_log = np.zeros((FT_EPOCHS, 4))
    best, state, best_ep = np.inf, None, -1
    for e in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        loss = lw(m(xt), yt); loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lw(m(xv), yv))
            zt = m(xe)
            ft_log[e] = [float(loss), v, float(lw(zt, ye)),
                         f1_score(yte, zt.argmax(1).cpu().numpy(), labels=[0, 1, 2],
                                  average="macro", zero_division=0)]
        if v < best:
            best, best_ep = v, e
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        p = [torch.softmax(m(_T(z)), 1).cpu().numpy() for z in (Xva, Xte)]
    return pre_log, ft_log, best_ep, p


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk = dict(zip(("twostream", "tcn"), members(desc.shape[1])))
    xkey = {"twostream": "packed", "tcn": "spec"}
    os.makedirs(OUT_DIR, exist_ok=True)
    checked = False
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        logs = {f"{n}_{k}": [] for n in mk for k in ("pre", "ft", "best")}
        for fold, (rest_i, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest_i, test_size=0.25, stratify=y[rest_i],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            for name, fn in mk.items():
                X, P = own[xkey[name]], pads[xkey[name]]
                f = zfit(X[tr])
                pre = (zfit(P)(P), C[3][kp])
                for s in (0, 1, 2):
                    pl, fl, be, p = logged(fn, f(X[tr]), y[tr], f(X[va]), y[va],
                                           f(X[te]), y[te], pre, s)
                    if not checked:   # assert first: logging changed nothing
                        ref = train_variants(fn, f(X[tr]), y[tr], f(X[va]), y[va],
                                             [f(X[va]), f(X[te])], pre, s)["ft"]
                        assert np.allclose(ref[0], p[0]) and np.allclose(ref[1], p[1]), \
                            "logged loop does not reproduce the ft model"
                        print("logged loop reproduces the ft model exactly", flush=True)
                        checked = True
                    logs[f"{name}_pre"].append(pl)
                    logs[f"{name}_ft"].append(fl)
                    logs[f"{name}_best"].append(be)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", **{k: np.array(v) for k, v in logs.items()})
        print(f"rep {rep} done", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    D = {}
    for f in files:
        for k, v in np.load(f).items():
            D.setdefault(k, []).append(v)
    D = {k: np.concatenate(v) for k, v in D.items()}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for r, name in enumerate(("twostream", "tcn")):
        pre, ft, be = D[f"{name}_pre"], D[f"{name}_ft"], D[f"{name}_best"]
        n = len(be)
        a = ax[r, 0]
        for j, lab in enumerate(("PADS train (optimised)", "2015 train fold (unseen)",
                                 "2015 val fold (unseen)")):
            mu, lo, hi = pre[:, :, j].mean(0), *np.percentile(pre[:, :, j], [25, 75], 0)
            a.plot(mu, label=lab); a.fill_between(range(len(mu)), lo, hi, alpha=0.2)
        a.set_title(f"{name}: PADS pretraining (n={n})"); a.set_xlabel("epoch")
        a.set_ylabel("weighted CE"); a.legend(fontsize=8)
        a = ax[r, 1]
        for j, lab in enumerate(("2015 train (optimised)", "2015 val (checkpoint)",
                                 "2015 test (diagnosis)")):
            mu, lo, hi = ft[:, :, j].mean(0), *np.percentile(ft[:, :, j], [25, 75], 0)
            a.plot(mu, label=lab); a.fill_between(range(len(mu)), lo, hi, alpha=0.2)
        a.set_title(f"{name}: fine-tuning on 2015"); a.set_xlabel("epoch"); a.legend(fontsize=8)
        a = ax[r, 2]
        a.hist(be, bins=np.arange(0, 81, 4), alpha=0.6, label="checkpoint epoch (best val)")
        a2 = a.twinx()
        a2.plot(ft[:, :, 3].mean(0), color="k", label="test macro-F1 (diagnosis)")
        a.set_title(f"{name}: chosen epoch vs test macro-F1"); a.set_xlabel("epoch")
        a.legend(loc="upper left", fontsize=8); a2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = "reports/figures/loss_curves_2015.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    for name in ("twostream", "tcn"):
        pre, ft, be = D[f"{name}_pre"], D[f"{name}_ft"], D[f"{name}_best"]
        va = pre[:, :, 2].mean(0)
        te = ft[:, :, 2].mean(0)
        f1 = ft[:, :, 3].mean(0)
        sel_te = np.array([c[b, 2] for c, b in zip(ft, be)])
        sel_f1 = np.array([c[b, 3] for c, b in zip(ft, be)])
        print(f"\n{name} ({len(be)} models)")
        print(f"  pretraining: 2015-val loss min at epoch {va.argmin()} "
              f"({va.min():.3f}); at epoch 199: {va[-1]:.3f}; PADS train "
              f"{pre[:, -1, 0].mean():.3f}")
        print(f"  fine-tune: train {ft[:, 0, 0].mean():.3f} -> {ft[:, -1, 0].mean():.3f};"
              f" val {ft[:, 0, 1].mean():.3f} -> min {ft[:, :, 1].mean(0).min():.3f}"
              f" @ {ft[:, :, 1].mean(0).argmin()} -> {ft[:, -1, 1].mean():.3f}")
        print(f"  test loss: mean-curve min at epoch {te.argmin()} ({te.min():.3f}); "
              f"final {te[-1]:.3f};  test macro-F1 mean-curve max at epoch "
              f"{f1.argmax()} ({f1.max():.3f}), final {f1[-1]:.3f}")
        print(f"  checkpoint epoch: median {np.median(be):.0f}, IQR "
              f"[{np.percentile(be, 25):.0f}, {np.percentile(be, 75):.0f}], "
              f"epoch 0 in {np.mean(be == 0):.0%}, last 10 in {np.mean(be >= 70):.0%}")
        print(f"  at the checkpoint: test loss {sel_te.mean():.3f}, test macro-F1 "
              f"{sel_f1.mean():.3f}  (an oracle per-model best epoch: macro-F1 "
              f"{ft[:, :, 3].max(1).mean():.3f})")


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        plot()
    else:
        a, b = map(int, os.environ.get("REPS", "0-5").split("-"))
        run(range(a, b))
