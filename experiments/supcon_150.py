"""Supervised contrastive PADS pretraining for the 2015 OUT transfer model.

Question. `ft` pretrains the reported recipe (two-stream + ResidualTCN x seeds
0-2) on PADS StretchHold with class-weighted cross-entropy only, then
fine-tunes everything on 2015. Does adding a supervised contrastive term
(SupCon; Khosla et al., NeurIPS 2020) to the PADS pretraining -- pulling
same-class PADS rows together and pushing other classes apart in the
penultimate feature space -- give 2015 a better starting representation, and in
particular raise per-class precision (ET above all)?

How this differs from the closed self-supervised work (`ssl_retraction.md`,
`closed_families.md`): that study was LABEL-FREE (masked-spectrum
reconstruction on 3,081 recordings), used a FROZEN encoder + linear head, and
its apparent gain was frozen-vs-fine-tuned confounding. Here the pretraining is
SUPERVISED with the same PADS labels `ft` already uses (only the loss geometry
on the penultimate layer changes), and every arm is fine-tuned in full with the
identical recipe -- so pretraining is the only difference between arms.

Arms (per fold x member x seed; 200 pretraining epochs, AdamW lr 3e-3 wd 1e-3,
cosine over 200 -- exactly `common.protocol.train`'s pretraining; then the
standard fine-tune: AdamW lr 1e-3, cosine 80 epochs, best 2015-validation-loss
checkpoint):

    ft          CE pretraining (the current recipe; asserted bit-identical to
                `common.protocol.train(pre=...)` for every member and seed of the
                first fold of the first repeat run by each process).
    supcon      CE + 0.5 x SupCon (tau 0.1) on the L2-normalised penultimate
                features (input of the final nn.Linear, captured with a forward
                hook, i.e. after the head's dropout in train mode), full batch
                over all PADS rows; positives = every other row of the same class.
    supcon_aug  as `supcon`, but the batch holds every PADS row twice: the second
                copy has Gaussian noise (sd 0.05, z-scored units, own
                torch.Generator) added to the input, so each anchor has its noisy
                view as a positive in addition to the same-class rows. CE is
                applied to the clean copies only (so the CE term is the same
                quantity as in the other arms; the noisy copies enter through the
                contrastive term and the BatchNorm batch).

Same initial weights in every arm (torch.manual_seed(seed) before building the
model). Fine-tuning is identical across arms, including its dropout stream: the
torch RNG state at the end of the `ft` pretraining is restored before every
arm's fine-tune. SupCon anchors are averaged unweighted (Khosla); the CE term
keeps the class weights. Hyper-parameters (0.5, tau 0.1, sd 0.05) are fixed in
advance from the task specification; nothing is tuned.

Protocol: `transfer_2015` -- corrected 150-patient 2015 OUT table (61 N / 74 PD /
15 ET), StratifiedKFold(5, shuffle, random_state=rep), stratified 25 % inner
validation (random_state=rep*10+fold), PADS StretchHold capped 90/class redrawn
per repeat (rng(rep)), per-domain `zfit`, `transfer_2015.members`, soft vote of
2 members x 3 seeds, `tune_offsets` on validation. Repeats (seeds) 0-19, CPU,
one thread. No test-fold data is used for any choice.

## Prediction, recorded before the run

* The CE pretraining already separates the three PADS classes at the head
  (`pretrain_stop_150.md`: the head becomes over-confident on PADS by epoch
  200). SupCon at weight 0.5 tightens the same class clusters one layer
  earlier; it adds no new label information and no 2015 information, and the
  full fine-tune then moves the whole network anyway. PADS ET also differ from
  2015 ET (2x amplitude, 2x sharpness), so a tighter PADS-ET cluster is not
  obviously a better 2015-ET starting point.
* supcon vs ft: null. precET within +-0.04 with the repeat-level CI spanning 0;
  macroP and macroF1 within +-0.015; PD-vs-ET AUC within +-0.02; precN, precPD
  within +-0.02. If anything a slight ET loss (more PADS-specific geometry).
* supcon_aug vs ft: same as supcon; supcon_aug vs supcon |diff| < 0.02 on every
  column (sd 0.05 noise on unit-variance inputs is a near-duplicate view, and
  same-class positives already dominate each anchor's positive set: ~27-89 vs 1).
* Patient level: no arm's interval excludes 0 on any column. Expected verdict:
  not adopted; CE-only `ft` stays.

Run (resumable, one repeat per file; then the report):
    nohup python -m experiments.supcon_150 > <scratchpad>/supcon_150.log 2>&1 &
    python -m experiments.supcon_150 report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, train, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, members, zfit

EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80   # = protocol.train
LAM, TAU, NOISE_SD = 0.5, 0.1, 0.05
ARMS = ("ft", "supcon", "supcon_aug")
MEMBERS = ("two", "tcn")
SEEDS = (0, 1, 2)
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "supcon_150_runs")


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


def _head(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


def supcon_loss(feat, y, tau=TAU):
    """Khosla et al. (2020) L_out on L2-normalised features, full batch.

    For anchor i: -1/|P(i)| sum_{p in P(i)} log softmax_{a != i}(z_i.z_a / tau)[p],
    P(i) = rows of the same label except i; averaged over anchors with |P(i)| > 0.
    """
    z = F.normalize(feat, dim=1)
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = (z @ z.T / tau).masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~eye
    npos = pos.sum(1)
    per = -logp.masked_fill(~pos, 0.0).sum(1) / npos.clamp(min=1)
    keep = npos > 0
    return per[keep].mean()


def pretrain(m, xp, ypt, arm, seed):
    """200-epoch PADS pretraining. `ft` is protocol.train's loop verbatim."""
    lf = nn.CrossEntropyLoss(weight=_wt(ypt.cpu().numpy()))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    n = xp.shape[0]
    if arm == "supcon_aug":
        g = torch.Generator(device=xp.device).manual_seed(10_000 + seed)
        y2 = torch.cat([ypt, ypt])
    feats = {}
    hook = None
    if arm != "ft":
        hook = _head(m).register_forward_hook(
            lambda mod, inp, out: feats.__setitem__("f", inp[0]))
    ce_last = sc_last = np.nan
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad()
        if arm == "ft":
            loss = lf(m(xp), ypt)
            ce_last = float(loss)
        elif arm == "supcon":
            ce = lf(m(xp), ypt)
            sc = supcon_loss(feats["f"], ypt)
            loss = ce + LAM * sc
            ce_last, sc_last = float(ce), float(sc)
        else:
            noise = torch.randn(xp.shape, generator=g, device=xp.device) * NOISE_SD
            out = m(torch.cat([xp, xp + noise]))
            ce = lf(out[:n], ypt)
            sc = supcon_loss(feats["f"], y2)
            loss = ce + LAM * sc
            ce_last, sc_last = float(ce), float(sc)
        loss.backward(); op.step(); sp.step()
    if hook is not None:
        hook.remove()
    return ce_last, sc_last


def finetune(m, xt, yt, xv, yv, Xout, ytr):
    """protocol.train's fine-tuning loop (pre path, no head-only / L2-SP)."""
    lw = nn.CrossEntropyLoss(weight=_wt(ytr))
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    best, state = np.inf, None
    for _ in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        lw(m(xt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lw(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return [torch.softmax(m(_T(z)), 1).cpu().numpy() for z in Xout]


def fit_all(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed):
    """All arms for one (member, seed). Returns {arm: probs list}, {arm: (ce, sc)}."""
    xt, yt, xv, yv = _T(Xtr), _L(ytr), _T(Xva), _L(yva)
    xp, ypt = _T(pre[0]), _L(pre[1])
    res, losses, rng_ft = {}, {}, None
    for arm in ARMS:                      # ft first: its RNG state is reused
        torch.manual_seed(seed)
        m = model_fn().to(DEVICE)
        losses[arm] = pretrain(m, xp, ypt, arm, seed)
        if arm == "ft":
            rng_ft = torch.get_rng_state()
        else:
            torch.set_rng_state(rng_ft)
        res[arm] = finetune(m, xt, yt, xv, yv, Xout, ytr)
    return res, losses


# ---------------------------------------------------------------- run
def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    assert len(y) == 150, f"expected the corrected 150-patient table, got {len(y)}"
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk = dict(zip(MEMBERS, members(desc.shape[1])))
    key = {"two": "packed", "tcn": "spec"}
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEVICE}  2015 n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  PADS N/PD/ET="
          f"{[int((C[3] == k).sum()) for k in (0, 1, 2)]}", flush=True)
    checked = False
    for rep in reps:
        out = f"{OUT_DIR}/rep{rep:02d}.npz"
        if os.path.exists(out):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {a: np.zeros((len(y), 3)) for a in ARMS}
        pred = {a: np.full(len(y), -1) for a in ARMS}
        log = {f"{w}_{a}_{n}": [] for n in MEMBERS for a in ARMS for w in ("ce", "sc")}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            pv_m, pt_m = {a: [] for a in ARMS}, {a: [] for a in ARMS}
            for name in MEMBERS:
                X, P = own[key[name]], pads[key[name]]
                f = zfit(X[tr])
                pre = (zfit(P)(P), C[3][kp])
                res = []
                for s in SEEDS:
                    r, ls = fit_all(mk[name], f(X[tr]), y[tr], f(X[va]), y[va],
                                    [f(X[va]), f(X[te])], pre, s)
                    if not checked:
                        ref = train(mk[name], f(X[tr]), y[tr], f(X[va]), y[va],
                                    [f(X[va]), f(X[te])], seed=s, pre=pre)
                        assert all(np.array_equal(a, b) for a, b in zip(ref, r["ft"])), \
                            "ft arm does not reproduce common.protocol.train"
                        print(f"  ft arm == common.protocol.train bit-exactly "
                              f"({name}, seed {s})", flush=True)
                    res.append(r)
                    for a in ARMS:
                        log[f"ce_{a}_{name}"].append(ls[a][0])
                        log[f"sc_{a}_{name}"].append(ls[a][1])
                for a in ARMS:
                    pv_m[a].append(np.mean([r[a][0] for r in res], 0))
                    pt_m[a].append(np.mean([r[a][1] for r in res], 0))
            checked = True                 # every member x seed of the first fold
            for a in ARMS:
                pv, pt = np.mean(pv_m[a], 0), np.mean(pt_m[a], 0)
                prob[a][te] = pt
                pred[a][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            print(f"  rep {rep} fold {fold} done", flush=True)
        np.savez(out, y=y, **pred, **{f"p_{a}": v for a, v in prob.items()},
                 **{k: np.array(v) for k, v in log.items()})
        Pr = {a: precision_recall_fscore_support(y, pred[a], labels=[0, 1, 2],
                                                 zero_division=0)[0] for a in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{a} ET {Pr[a][2]:.2f} mP {Pr[a].mean():.3f}"
                                           for a in ARMS), flush=True)


# ---------------------------------------------------------------- report
def _metrics(y, pred, pr):
    P, _, Fs, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2],
                                                  zero_division=0)
    m = y != 0
    s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
    o = np.argsort(-pr[:, 2])
    return [P[0], P[1], P[2], P.mean(), Fs.mean(),
            roc_auc_score((y[m] == 2).astype(int), s), (y[o[:5]] == 2).mean()]


def _paired(dd, n):
    for i, nm in enumerate(NAMES):
        bs = [np.random.default_rng(s).choice(dd[:, i], n).mean() for s in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"    {nm:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"{'*' if lo > 0 or hi < 0 else ' '} win {np.mean(dd[:, i] > 0):.2f}"
              f" tie {np.mean(dd[:, i] == 0):.2f}")


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    n = len(files)
    R = {a: [] for a in ARMS}
    etp = {a: 0 for a in ARMS}
    L = {}
    for f in files:
        d = np.load(f)
        y = d["y"]
        for a in ARMS:
            R[a].append(_metrics(y, d[a], d[f"p_{a}"]))
            etp[a] += (d[a] == 2).sum()
        for k in d.files:
            if k.startswith(("ce_", "sc_")):
                L.setdefault(k, []).append(d[k])
    R = {a: np.array(v) for a, v in R.items()}
    print(f"2015 OUT, {n} repeats ({[os.path.basename(f) for f in files[:1]]}..), "
          f"{len(y)} patients (N/PD/ET = {[int((y == c).sum()) for c in (0, 1, 2)]})\n")
    print(f"{'arm':>11}" + "".join(f"{c:>9}" for c in NAMES) + "  ET preds/rep")
    for a in ARMS:
        print(f"{a:>11}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0))
              + f"  {etp[a] / n:.1f}")
    print("\nrepeat-level paired bootstrap (4000 draws over repeats), [95% CI], win rate")
    for a, b in (("supcon", "ft"), ("supcon_aug", "ft"), ("supcon_aug", "supcon")):
        print(f"  {a} - {b}")
        _paired(R[a] - R[b], n)
    print("\nfinal-epoch PADS pretraining losses (mean over fold x seed x repeat)")
    for name in MEMBERS:
        for a in ARMS:
            ce = np.concatenate(L[f"ce_{a}_{name}"])
            sc = np.concatenate(L[f"sc_{a}_{name}"])
            print(f"  {name:>4} {a:<11} CE {np.nanmean(ce):.4f}"
                  + (f"  SupCon {np.nanmean(sc):.4f}" if a != "ft" else ""))
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    print("\npatient-level (experiments.patient_bootstrap_2015.bootstrap_dir)")
    for a, b in (("supcon", "ft"), ("supcon_aug", "ft"), ("supcon_aug", "supcon")):
        bootstrap_dir(OUT_DIR, a, b, tag="supcon_150")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
