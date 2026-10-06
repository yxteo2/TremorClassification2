"""Is the "TCN epoch ~20" effect real when the epoch is chosen inside the fold?

Background (`loss_curves_2015.md`, `checkpoint_rule_2015`): a rule read off
TEST curves -- two-stream keeps its best-validation-loss checkpoint, TCN keeps
fine-tuning epoch 20 -- gave PD-vs-ET AUC +0.012 / +0.017 \\* and top-5 ET
precision +0.046 / +0.090 \\* over `ft`. In 5-fold CV every patient is a test
patient, so that choice saw every label and its gain is biased upward.
Single-seed validation-loss curves (~30 patients) are flat after ~40 epochs, so
the per-seed best epoch is near-random (`ckpt_ensemble_150.md`: two-stream
median 34, IQR 14-67; TCN 22, IQR 5-56).

Here the epoch is chosen INSIDE each training fold, from the inner validation
split only. For each fold and member the 3 seeds are trained once
(`common.protocol.train` recipe: PADS pretrain 200 epochs, fine-tune 80) and
every fine-tuning epoch's validation and test softmax outputs are recorded.
Arms, all from the same trajectories (rules fixed here, before the run):

    ft           per-seed best-validation-loss epoch (current recipe; first minimum)
    pooled_loss  per member, the single epoch minimising the class-weighted CE
                 (training-fold inverse-frequency weights, as in fine-tuning) of
                 the SEED-AVERAGED validation softmax; applied to all 3 seeds
    pooled_f1    per member, the single epoch maximising validation macro-F1 of
                 the seed-averaged softmax at argmax (ties -> earliest)
    tcn_pooled   two-stream as `ft`, TCN as `pooled_loss` -- the in-fold
                 analogue of the biased rule
    tcn_e20      REFERENCE ONLY, not a candidate: two-stream as `ft`, TCN at
                 fine-tuning epoch 20 (1-based; index 19) -- the biased rule
                 itself, re-measured on these partitions. Its number is still
                 contaminated by the earlier test-curve choice.

Members are averaged over seeds, then over the two members; per-class logit
offsets (`tune_offsets`) are tuned on each arm's own validation probabilities.
Protocol = `transfer_2015` / `ckpt_ensemble_150`: StratifiedKFold(5, shuffle,
random_state=rep), 25 % stratified inner validation (random_state rep*10+fold),
PADS capped 90/class redrawn per repeat, `zfit` per domain. 150 patients
(61 N / 74 PD / 15 ET). `python -m experiments.infold_epoch_150 check` asserts
the `ft` arm is bit-identical to `common.protocol.train(pre=...)` on one fold.

## Prediction, recorded before the run

* Chosen epochs: the seed-averaged validation loss is smoother than a single
  seed's but still flat after ~40, so `pooled_loss` picks a TCN epoch with
  median >= 25 and a wide IQR (it does NOT concentrate near 20); two-stream
  similar. `pooled_f1` (macro-F1 on ~30 patients, ~3 ET, step-valued, ties ->
  earliest) picks EARLIER epochs than `pooled_loss` (TCN median <= 20).
* `pooled_loss` vs ft: a variance reduction of the checkpoint choice, not a new
  rule. PD-vs-ET AUC |delta| < 0.01, precET |delta| < 0.03, both n.s.
* `pooled_f1` vs ft: noisier selector; AUC 0 to -0.02, precET n.s.
* `tcn_pooled` vs ft: AUC within +-0.01 and n.s. -- does NOT reproduce the
  biased +0.012/+0.017; top-5 ET n.s.
* `tcn_e20` (reference) reproduces part of the biased effect: AUC +0.005 to
  +0.015 at the repeat level.
* Nothing significant at the patient level (resolution ~+-0.03); no arm adopted.
  The epoch-20 finding will read as UNSUPPORTED by in-fold selection.

Run: ``python -m experiments.infold_epoch_150 check``, then
``REPS=0-20 python -m experiments.infold_epoch_150`` (resumable), then
``python -m experiments.infold_epoch_150 report``. Set OUT_DIR to keep the
per-repeat files outside the repo.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from common.protocol import DEVICE, tune_offsets
from experiments.ckpt_ensemble_150 import (EPOCHS, FT_EPOCHS, FT_LR, LR, WD, _L,
                                           _T, _folds, _pads, _wt)
from experiments.own_data_10et import build
from experiments.transfer_2015 import members, zfit

ARMS = ("ft", "pooled_loss", "pooled_f1", "tcn_pooled", "tcn_e20")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "infold_epoch_150_runs")
E20 = 19                     # fine-tuning epoch 20, 0-based index
SEEDS = (0, 1, 2)


def trajectory(model_fn, Xtr, ytr, Xva, yva, Xte, pre, seed):
    """common.protocol.train(pre=...) with every fine-tuning epoch's outputs logged.

    Returns val loss (E,), val probs (E, nva, 3), test probs (E, nte, 3).
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
    vl, pv, pt = [], [], []
    for _ in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        lw(m(xt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            zv = m(xv)
            vl.append(float(lw(zv, yv)))
            pv.append(torch.softmax(zv, 1).cpu().numpy())
            pt.append(torch.softmax(m(xte), 1).cpu().numpy())
    return np.array(vl), np.array(pv), np.array(pt)


def wce(p, yv, ytr):
    """Class-weighted CE (training-fold weights, PyTorch weighted mean) of probs p."""
    c = np.bincount(ytr, minlength=3).astype(float)
    w = (c.sum() / (3 * np.maximum(c, 1)))[yv]
    lp = np.log(np.clip(p[np.arange(len(yv)), yv].astype(np.float64), 1e-12, None))
    return float(-(w * lp).sum() / w.sum())


def mf1(p, yv):
    return precision_recall_fscore_support(yv, p.argmax(1), labels=[0, 1, 2],
                                           zero_division=0)[2].mean()


def choose(VL, PV, yv, ytr):
    """Epoch choices for one member from validation data only.

    VL (S, E), PV (S, E, nva, 3). Returns per-seed ft epochs, pooled-loss epoch,
    pooled-F1 epoch, and the pooled curves.
    """
    ep_ft = VL.argmin(1)                                  # first minimum per seed
    PM = PV.mean(0)                                       # seed-averaged (E, nva, 3)
    cl = np.array([wce(PM[e], yv, ytr) for e in range(len(PM))])
    cf = np.array([mf1(PM[e], yv) for e in range(len(PM))])
    return ep_ft, int(np.argmin(cl)), int(np.argmax(cf)), cl, cf


def member_outputs(VL, PV, PT, yv, ytr):
    """{rule: (val probs, test probs)} for one member, plus the choices."""
    ep_ft, e_pl, e_pf, cl, cf = choose(VL, PV, yv, ytr)
    s = np.arange(len(VL))
    out = {"ft": (PV[s, ep_ft].mean(0), PT[s, ep_ft].mean(0)),
           "pl": (PV[:, e_pl].mean(0), PT[:, e_pl].mean(0)),
           "pf": (PV[:, e_pf].mean(0), PT[:, e_pf].mean(0)),
           "e20": (PV[:, E20].mean(0), PT[:, E20].mean(0))}
    return out, ep_ft, e_pl, e_pf, cl, cf


# arm -> (two-stream rule, TCN rule)
RULES = {"ft": ("ft", "ft"), "pooled_loss": ("pl", "pl"), "pooled_f1": ("pf", "pf"),
         "tcn_pooled": ("ft", "pl"), "tcn_e20": ("ft", "e20")}


def _setup():
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    return spec, desc, y, own, Cp, C


def check():
    """Assert the `ft` arm equals common.protocol.train(pre=...) bit for bit."""
    from common.protocol import train
    spec, desc, y, own, Cp, C = _setup()
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
        vl, pv, pt = trajectory(mk, f(X[tr]), y[tr], f(X[va]), y[va], f(X[te]), pre, 0)
        e = int(np.argmin(vl))
        assert np.array_equal(ref[0], pv[e]) and np.array_equal(ref[1], pt[e]), key
        print(f"{key}: ft arm bit-identical to common.protocol.train (epoch {e})",
              flush=True)
    print("CHECK_OK", flush=True)


def run(reps):
    spec, desc, y, own, Cp, C = _setup()
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
        ep_ft = np.zeros((5, 2, 3), int)
        ep_pl = np.zeros((5, 2), int); ep_pf = np.zeros((5, 2), int)
        c_vl = np.zeros((5, 2, len(SEEDS), FT_EPOCHS))
        c_pl = np.zeros((5, 2, FT_EPOCHS)); c_pf = np.zeros((5, 2, FT_EPOCHS))
        # descriptive only (never used for a choice): seed-averaged test probs per epoch
        ptest = {}
        for fold, tr, va, te in _folds(spec, y, rep):
            mo = []
            for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                VL, PV, PT = [], [], []
                for s in SEEDS:
                    vl, pv, pt = trajectory(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                            f(X[te]), (zfit(P)(P), C[3][kp]), s)
                    VL.append(vl); PV.append(pv); PT.append(pt)
                VL, PV, PT = np.array(VL), np.array(PV), np.array(PT)
                o, ef, epl, epf, cl, cf = member_outputs(VL, PV, PT, y[va], y[tr])
                mo.append(o)
                ep_ft[fold, j], ep_pl[fold, j], ep_pf[fold, j] = ef, epl, epf
                c_vl[fold, j], c_pl[fold, j], c_pf[fold, j] = VL, cl, cf
                ptest[f"pt_f{fold}_m{j}"] = PT.mean(0).astype(np.float32)
            for k, (r1, r2) in RULES.items():
                v = (mo[0][r1][0] + mo[1][r2][0]) / 2
                t = (mo[0][r1][1] + mo[1][r2][1]) / 2
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
            ptest[f"te_f{fold}"] = te
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()},
                 ep_ft=ep_ft, ep_pl=ep_pl, ep_pf=ep_pf, vl=c_vl, cpl=c_pl, cpf=c_pf,
                 **ptest)
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {Pr[k][2]:.2f}" for k in ARMS)
              + f"  | TCN ep ft med {np.median(ep_ft[:, 1]):.0f}"
              f" pl {list(ep_pl[:, 1])} pf {list(ep_pf[:, 1])}", flush=True)


def _boot(dd, n):
    bs = [np.random.default_rng(s).choice(dd, n).mean() for s in range(4000)]
    return np.percentile(bs, [2.5, 97.5])


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    E = {"ep_ft": [], "ep_pl": [], "ep_pf": []}
    curves = {"cpl": [], "cpf": []}
    tf1 = []                      # descriptive: test macro-F1 of seed-avg TCN per epoch
    for f in files:
        d = np.load(f)
        y = d["y"]
        m = y != 0
        for k in E:
            E[k].append(d[k])
        for k in curves:
            curves[k].append(d[k])
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[k], labels=[0, 1, 2],
                                                         zero_division=0)
            pr = d[f"p_{k}"]
            s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
            o = np.argsort(-pr[:, 2])
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean(),
                           roc_auc_score((y[m] == 2).astype(int), s),
                           (y[o[:5]] == 2).mean()])
        for j in (0, 1):
            pt = np.zeros((FT_EPOCHS, len(y), 3))
            for fold in range(5):
                pt[:, d[f"te_f{fold}"]] = d[f"pt_f{fold}_m{j}"]
            tf1.append((j, [mf1(pt[e], y) for e in range(FT_EPOCHS)]))
    n = len(files)
    R = {k: np.array(v) for k, v in res.items()}
    print(f"2015 OUT (n={len(y)}), in-fold epoch selection, {n} repeats\n")
    print(f"{'arm':>12}" + "".join(f"{c:>9}" for c in NAMES) + "  sd(precET)  ET preds/rep")
    for k in ARMS:
        npred = np.mean([(np.load(f)[k] == 2).sum() for f in files])
        print(f"{k:>12}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0))
              + f"  {R[k][:, 2].std():.3f}       {npred:.1f}")
    print("\nchosen fine-tuning epochs (1-based, of 80), per member:")
    for k in E:
        a = np.concatenate([x[None] for x in E[k]]) + 1
        for j, nm in enumerate(("two-stream", "TCN")):
            v = a[:, :, j].ravel()
            h = np.histogram(v, bins=[1, 11, 21, 31, 41, 61, 81])[0] / len(v)
            print(f"  {k:>6} {nm:>10}: median {np.median(v):.0f}  IQR "
                  f"{np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}  "
                  f"share in 1-10/11-20/21-30/31-40/41-60/61-80: "
                  + "/".join(f"{x:.2f}" for x in h)
                  + f"  | within 15-25: {np.mean((v >= 15) & (v <= 25)):.2f}")
    print("\nmean seed-averaged VALIDATION curves (fold x repeat), epochs 1,5,10,15,20,25,30,40,60,80:")
    ix = np.array([1, 5, 10, 15, 20, 25, 30, 40, 60, 80]) - 1
    for k in curves:
        a = np.concatenate([x[None] for x in curves[k]])     # (n, 5, 2, E)
        for j, nm in enumerate(("two-stream", "TCN")):
            print(f"  {k} {nm:>10}: " + " ".join(f"{v:.3f}" for v in a[:, :, j].mean((0, 1))[ix]))
    print("\ndescriptive (TEST, not used for any choice) seed-avg member macro-F1 at argmax:")
    for j, nm in enumerate(("two-stream", "TCN")):
        a = np.array([c for jj, c in tf1 if jj == j]).mean(0)
        print(f"  {nm:>10}: " + " ".join(f"{v:.3f}" for v in a[ix])
              + f"  (argmax epoch {int(np.argmax(a)) + 1})")
    for a in ARMS[1:]:
        dd = R[a] - R["ft"]
        print(f"\n{a} - ft  (paired over repeats, {n})")
        for i, c in enumerate(NAMES):
            lo, hi = _boot(dd[:, i], n)
            star = "*" if lo > 0 or hi < 0 else " "
            extra = ""
            if c == "top5ET":
                extra = (f"  up/tie/down {np.sum(dd[:, i] > 0)}/{np.sum(dd[:, i] == 0)}"
                         f"/{np.sum(dd[:, i] < 0)}")
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}{extra}")
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    for a in ARMS[1:]:
        bootstrap_dir(OUT_DIR, a, "ft", tag=a)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
