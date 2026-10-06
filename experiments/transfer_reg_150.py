"""Transfer regularisers that keep PADS features but not PADS's boundary.

`training_methods_2015.md`: WiSE-FT (0.5 x pretrained + 0.5 x fine-tuned
weights) collapsed the top of the ET ranking (top-5 ET 0.50 -> 0.23 \\*) -- the
PADS-trained decision rule is wrong for 2015, so fine-tuning has to keep the
PADS *features* while moving away from the PADS *boundary*. L2-SP (anchoring all
non-head weights) raised ranking only. The two regularisers below act on the
features and the source head instead of on the weights:

    ft          the current recipe (`common.protocol.train(pre=...)`, asserted
                bit-identical on one fold by ``check``)
    cotune      Co-Tuning (You et al., NeurIPS 2020). A copy of the pretrained
                PADS classifier (the final nn.Linear) is kept as a trainable
                "source head" on the shared features (the input to the final
                nn.Linear, captured by a forward hook; post-dropout in train
                mode). The relationship R[c] = P(y_PADS | y_2015 = c) is the
                row-normalised mean PRETRAINED softmax (eval mode) over the
                fold's 2015 TRAINING patients of 2015 class c (validation and
                test never enter). Fine-tuning loss =
                CE_w(target head) + 1.0 x mean_i soft-CE(source head, R[y_i]).
                The soft term is an unweighted mean over patients (fixed here).
    bss         Batch Spectral Shrinkage (Chen et al., NeurIPS 2019):
                CE_w + 1e-3 x sum of the k=1 smallest squared singular values of
                the full-batch penultimate feature matrix (same hook), during
                fine-tuning only.
    cotune_bss  both terms.

Everything else is `transfer_2015`'s protocol, and all arms branch from the SAME
PADS-pretrained weights per member and seed (pretraining as in
`common.protocol.train`; the RNG state after pretraining is restored before each
arm's fine-tune, `ft` first). Checkpoint rule identical across arms: the epoch
with the lowest class-weighted validation CE of the TARGET head. Fine-tune 80
epochs, AdamW lr 1e-3, wd 1e-3, cosine; the source head is in the same
optimiser. StratifiedKFold(5, shuffle, random_state=rep), 25 % stratified inner
validation (random_state rep*10+fold), PADS StretchHold capped 90/class redrawn
per repeat, `transfer_2015.members` x seeds 0-2, `zfit` per domain,
`tune_offsets` on each arm's own validation probabilities. Corrected loader:
150 patients (61 N / 74 PD / 15 ET). Repeats 0-19. Hyper-parameters (1.0,
1e-3, k=1) are the papers' / the task's, fixed before the run.

## Prediction, recorded before the run

* The relationship matrix is near-diagonal for N but blurred between PD and ET
  (PADS PD-vs-ET transfers weakly: linear AUC 0.589). Co-Tuning's soft targets
  therefore ask the shared features to keep PD and ET *partly confusable* the
  way PADS sees them -- a feature-side cousin of L2-SP / WiSE-FT, but the target
  head is free. Expected: `cotune` PD-vs-ET AUC within +-0.015 of `ft`, precET
  |delta| < 0.03 and not significant at repeat level, top-5 ET not collapsed
  (|delta| < 0.10, unlike WiSE-FT's -0.27); precPD/precN within +-0.02.
* `bss` at 1e-3 is a small penalty on one direction of a 90-row batch; it barely
  moves the trajectory: same-patient prediction agreement with `ft` >= 90 %,
  every metric |delta| < 0.015, nothing significant.
* `cotune_bss` ~ `cotune`.
* Nothing significant at the PATIENT level for any arm (resolution ~+-0.03).
  Verdict: none adopted.

Run: ``python -m experiments.transfer_reg_150 check`` (bit-identity), then
``REPS=0-20 nohup python -m experiments.transfer_reg_150 &`` (resumable),
then ``python -m experiments.transfer_reg_150 report``.
"""

from __future__ import annotations

import copy
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ft", "cotune", "bss", "cotune_bss")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "transfer_reg_150_runs")
EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80
CO_W, BSS_W, BSS_K = 1.0, 1e-3, 1


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


def _head(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def relationship(m, xt, ytr):
    """R[c] = row-normalised mean pretrained softmax over training patients of class c."""
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(xt), 1).cpu().numpy().astype(np.float64)
    R = np.stack([p[ytr == c].mean(0) for c in range(3)])
    return R / R.sum(1, keepdims=True)


def _finetune(m, xt, yt, xv, yv, w, arm, R):
    """common.protocol.train's fine-tuning loop with the Co-Tuning / BSS terms."""
    co, bss = arm in ("cotune", "cotune_bss"), arm in ("bss", "cotune_bss")
    feats, hook, src = {}, None, None
    params = list(m.parameters())
    if co or bss:
        head = _head(m)
        hook = head.register_forward_hook(lambda mod, inp, out: feats.__setitem__("f", inp[0]))
    if co:
        src = copy.deepcopy(_head(m))               # pretrained PADS classifier
        params = params + list(src.parameters())
        soft = _T(R)[yt]                            # (n, 3) relationship rows
    lf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(params, lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    best, state, ep_best, bss0 = np.inf, None, -1, np.nan
    for ep in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        loss = lf(m(xt), yt)
        if co:
            loss = loss + CO_W * (-(soft * F.log_softmax(src(feats["f"]), 1)).sum(1).mean())
        if bss:
            sv = torch.linalg.svdvals(feats["f"])   # descending
            pen = (sv[-BSS_K:] ** 2).sum()
            if ep == 0:
                bss0 = float(pen.detach())
            loss = loss + BSS_W * pen
        loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lf(m(xv), yv))
        if v < best:
            best, ep_best = v, ep
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if hook is not None:
        hook.remove()
    if state:
        m.load_state_dict(state)
    m.eval()
    return m, ep_best, bss0


def train_arms(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed, arms=ARMS):
    """Pretrain once exactly as common.protocol.train, then fine-tune each arm."""
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
    rng = torch.get_rng_state()
    R = relationship(copy.deepcopy(m), xt, ytr)     # eval-mode forward: no RNG used
    assert torch.equal(rng, torch.get_rng_state())
    w = _wt(ytr)
    out, info = {}, {}
    for a in arms:                                  # ft first: same RNG as train()
        torch.set_rng_state(rng)
        mv, e, b0 = _finetune(copy.deepcopy(m), xt, yt, xv, yv, w, a, R)
        with torch.no_grad():
            out[a] = [torch.softmax(mv(_T(z)), 1).cpu().numpy() for z in Xout]
        info[a] = (e, b0)
    return out, R, info


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


def _data():
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    return spec, desc, y, own, Cp, C


def check():
    """Assert the `ft` arm equals common.protocol.train(pre=...) bit for bit."""
    import time
    from common.protocol import train
    torch.set_num_threads(1)
    spec, desc, y, own, Cp, C = _data()
    assert len(y) == 150, len(y)
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
        t0 = time.time()
        out, R, info = train_arms(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                  [f(X[va]), f(X[te])], pre, 0)
        dt = time.time() - t0
        assert np.array_equal(ref[0], out["ft"][0]) and np.array_equal(ref[1], out["ft"][1]), key
        agree = {a: float((out[a][1].argmax(1) == out["ft"][1].argmax(1)).mean()) for a in ARMS}
        print(f"{key}: ft arm bit-identical to common.protocol.train  ({dt:.0f} s, all arms)\n"
              f"  relationship P(y_PADS | y_2015) rows N/PD/ET:\n{np.round(R, 3)}\n"
              f"  best epochs {[info[a][0] for a in ARMS]}  bss0 {info['bss'][1]:.4f}\n"
              f"  test argmax agreement with ft {agree}", flush=True)
    print("CHECK_OK", flush=True)


def run(reps):
    torch.set_num_threads(1)
    spec, desc, y, own, Cp, C = _data()
    assert len(y) == 150, len(y)
    mk1, mk2 = members(desc.shape[1])
    print(f"device={DEVICE}  2015 OUT n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  arms={ARMS}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        kp = _pads(C, rep)
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        rel = np.zeros((5, 2, 3, 3, 3))
        ep = np.zeros((len(ARMS), 5, 2, 3), int)
        bss0 = np.zeros((5, 2, 3))
        for fold, tr, va, te in _folds(spec, y, rep):
            pv = {k: [] for k in ARMS}
            pt = {k: [] for k in ARMS}
            for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                per = []
                for s in (0, 1, 2):
                    o, R, info = train_arms(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                            [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]), s)
                    per.append(o)
                    rel[fold, j, s] = R
                    bss0[fold, j, s] = info["bss"][1]
                    for i, a in enumerate(ARMS):
                        ep[i, fold, j, s] = info[a][0]
                for k in ARMS:
                    pv[k].append(np.mean([r[k][0] for r in per], 0))
                    pt[k].append(np.mean([r[k][1] for r in per], 0))
            for k in ARMS:
                v, t = np.mean(pv[k], 0), np.mean(pt[k], 0)
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()}, rel=rel, ep=ep, bss0=bss0)
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {Pr[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    agree = {k: [] for k in ARMS}
    npred = {k: [] for k in ARMS}
    rel, ep, b0 = [], [], []
    for f in files:
        d = np.load(f)
        y = d["y"]
        m = y != 0
        rel.append(d["rel"]); ep.append(d["ep"]); b0.append(d["bss0"])
        for k in ARMS:
            P, _, Fs, _ = precision_recall_fscore_support(y, d[k], labels=[0, 1, 2],
                                                          zero_division=0)
            pr = d[f"p_{k}"]
            s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
            o = np.argsort(-pr[:, 2], kind="stable")
            res[k].append([P[0], P[1], P[2], P.mean(), Fs.mean(),
                           roc_auc_score((y[m] == 2).astype(int), s),
                           (y[o[:5]] == 2).mean()])
            agree[k].append(np.mean(d[k] == d["ft"]))
            npred[k].append((d[k] == 2).sum())
    n = len(files)
    R = {k: np.array(v) for k, v in res.items()}
    print(f"2015 OUT (n={len(y)}), transfer regularisers, {n} repeats\n")
    print(f"{'arm':>11}" + "".join(f"{c:>9}" for c in NAMES)
          + "  sd(precET)  ET preds/rep  agree w/ ft")
    for k in ARMS:
        print(f"{k:>11}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0))
              + f"  {R[k][:, 2].std():>9.3f}  {np.mean(npred[k]):>12.1f}"
              f"  {np.mean(agree[k]):>11.3f}")
    rel = np.concatenate([r[None] for r in rel])          # (n, 5, 2, 3, 3, 3)
    print("\nrelationship P(y_PADS | y_2015), mean over repeats/folds/seeds "
          "(rows 2015 N/PD/ET, cols PADS N/PD/ET):")
    for j, nm in enumerate(("two-stream", "TCN")):
        print(f"  {nm}:\n" + "\n".join("    " + " ".join(f"{v:.3f}" for v in row)
                                        for row in rel[:, :, j].mean((0, 1, 2))))
    ep = np.concatenate([e[None] for e in ep])            # (n, arms, 5, 2, 3)
    print("\nchosen fine-tune epoch (0-based of 80), median [two-stream, TCN]:")
    for i, k in enumerate(ARMS):
        print(f"  {k:>11}: {np.median(ep[:, i, :, 0]):.0f}, {np.median(ep[:, i, :, 1]):.0f}")
    b0 = np.concatenate([b[None] for b in b0])
    print(f"\nBSS penalty at fine-tune epoch 0 (smallest sigma^2, before x{BSS_W}): "
          f"two-stream median {np.median(b0[:, :, 0]):.4f}, TCN {np.median(b0[:, :, 1]):.4f}")
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
