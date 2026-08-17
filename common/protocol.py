"""Cohort training loop, prior tuning and the evaluation protocol.

``tune_offsets`` fits per-class logit offsets on the VALIDATION split and
applies them unchanged to test -- the single largest precision gain measured
in this project (ET precision 0.475 -> 0.612).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

NAMES = ("2015", "NewData", "PADS")
NBIN, N_ASYM = 16, 4
SPLITS, TEST_FRAC, VAL_FRAC = 10, 0.20, 0.20

from models.architectures import DescriptorFusion, ResidualTCN, Spectrum1DCNN, TRUNKS



def train(model_fn, Xtr, ytr, Xva, yva, Xout, seed=0, epochs=200, lr=3e-3,
          wd=1e-3, nc=3, sw=None, pre=None, ft_lr=1e-3, ft_epochs=80):
    """Full-batch trainer. ``sw`` weights samples; ``pre`` pretrains first."""
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.long)
    xv, yv = T(Xva), torch.tensor(yva, dtype=torch.long)
    m = model_fn()

    def wt(yy):
        c = np.bincount(yy, minlength=nc).astype(float)
        return torch.tensor(c.sum() / (nc * np.maximum(c, 1)), dtype=torch.float32)

    if pre is not None:
        Xp, yp = pre
        xp, ypt = T(Xp), torch.tensor(yp, dtype=torch.long)
        lf = nn.CrossEntropyLoss(weight=wt(yp))
        op = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, epochs)
        m.train()
        for _ in range(epochs):
            op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
        lr, epochs = ft_lr, ft_epochs

    w = wt(ytr)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    swt = T(sw) if sw is not None else None
    best, state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        if swt is None:
            loss = nn.CrossEntropyLoss(weight=w)(m(xt), yt)
        else:
            per = nn.CrossEntropyLoss(weight=w, reduction="none")(m(xt), yt)
            loss = (per * swt).sum() / swt.sum()
        loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(nn.CrossEntropyLoss(weight=w)(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return [torch.softmax(m(T(z)), 1).numpy() for z in Xout]
def tune_offsets(pv, yv):
    """Per-class logit offsets maximising VALIDATION macro F1."""
    best, bo = -1.0, np.zeros(3)
    lp = np.log(pv + 1e-12)
    for b1 in np.linspace(-1, 1, 9):
        for b2 in np.linspace(-1, 1, 9):
            o = np.array([0.0, b1, b2])
            _, _, F, _ = precision_recall_fscore_support(
                yv, (lp + o).argmax(1), labels=[0, 1, 2], zero_division=0)
            if F.mean() > best:
                best, bo = F.mean(), o
    return bo
def run(name, sb, dc, y, key, coh, mode="pool", per_cohort_priors=False,
        splits=SPLITS, verbose=True):
    out = []
    Xf, Xs = np.hstack([sb, dc]), sb
    n_desc = Xf.shape[1] - NBIN
    builders = [
        (Xf, lambda: DescriptorFusion(Spectrum1DCNN(NBIN, 3, ch=8),
                                      TRUNKS["cnn"], NBIN, n_desc, 8 * 2 * 4)),
        (Xs, lambda: ResidualTCN(NBIN, num_classes=3, ch=16)),
    ]
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(sb, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(sb[tv], key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in builders:
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            kw, tr_use = {}, tr
            if mode == "weight":
                cnt = {c: max((coh[tr] == c).sum(), 1) for c in NAMES}
                kw["sw"] = np.array([1.0 / cnt[c] for c in coh[tr]])
            elif mode == "finetune":
                p, q = tr[coh[tr] == "PADS"], tr[coh[tr] != "PADS"]
                if len(p) > 20 and len(q) > 20:
                    kw["pre"] = ((X[p] - mu) / sd, y[p])
                    tr_use = q
            r = [train(mk, (X[tr_use] - mu) / sd, y[tr_use], (X[va] - mu) / sd,
                       y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s, **kw)
                 for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        if per_cohort_priors:
            pred = np.zeros(len(te), int)
            for c in NAMES:
                mv, mt = coh[va] == c, coh[te] == c
                if not mt.any():
                    continue
                off = tune_offsets(pv[mv], y[va][mv]) if mv.sum() >= 8 \
                    else tune_offsets(pv, y[va])
                pred[mt] = (np.log(pt[mt] + 1e-12) + off).argmax(1)
        else:
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    a = np.array(out)
    if verbose:
        m, s = a.mean(0), a.std(0)
        print(f"{name:>36}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a
