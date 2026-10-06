"""The best combined 2015 system, on the corrected 150-patient table.

The components that survived patient resampling were measured one at a time,
on the old 151-row table (`patient_bootstrap_2015.md`): NewData pooled into
fine-tuning (PD-vs-ET AUC +0.044), WING as a third fused task (precPD +0.03),
PADS transfer (precET +0.08; n.s. at patient level on 150 rows,
`verify_transfer_150.md`). Here they are combined, per fold, on the SAME
partitions:

    ft      PADS pretrain -> 2015 fine-tune                 (`transfer_2015` ft)
    ft_nd   PADS pretrain -> fine-tune on 2015 training patients + ALL NewData
            (NewData z-scored on its own statistics, asymmetry columns and
            have-flag zeroed; = `newdata_2015` nd_pool)
    sys     ft_nd fused with 0.25 x REST scratch + 0.25 x WING scratch
            (log-probability weights 0.5 / 0.25 / 0.25, a term dropped for a
            patient without that task; = `fusion3_2015` "+ WING" on ft_nd)
    sys_ft  the same fusion on ft (attributes NewData's share inside the system)

`ft` and `ft_nd` branch from the same PADS-pretrained weights per member and
seed (`newdata_2015.branches`: ft first, RNG state restored before each
branch). Offsets (`tune_offsets`) are tuned on each arm's own (fused)
validation probabilities. Protocol = `transfer_2015`: StratifiedKFold(5,
shuffle, random_state=rep), 25 % stratified inner validation (random_state
rep*10+fold), PADS StretchHold capped 90/class redrawn per repeat, members
`transfer_2015.members` x 3 seeds, `zfit` per domain. Seeds 0-19.
``python -m experiments.combined_150 check`` asserts the `ft` branch is
bit-identical to `common.protocol.train(pre=...)` on one fold; the run also
asserts the `ft` predictions equal `ckpt_ensemble_150_runs` (same protocol,
150 rows) wherever that file exists.

High-confidence rule (from the saved out-of-fold probabilities): the NESTED
rule of `high_precision_2015.evaluate` (request 0.90, MIN_FLAG 3) for N, PD,
ET and the tremor-vs-none screen (P(PD) + P(ET)), all four arms reported, the
comparison of interest `sys` vs `ft`.

## Prediction, recorded before the run

* `ft` reproduces ckpt_ensemble_150's ft (precET ~0.33, precPD ~0.71, AUC ~0.61).
* `ft_nd` - `ft`: PD-vs-ET AUC +0.03 to +0.05 (repeat level \\*, and patient
  level \\*, as on 151 rows); precisions flat (|delta| < 0.03, n.s.), macroF1
  -0.02 to 0.
* `sys` - `ft`: precPD +0.03 to +0.06 \\* (REST + WING); precN +0.00 to +0.03;
  precET within +-0.04, n.s.; AUC +0.06 to +0.09 \\* (nd + REST ranking ~0.69
  vs ~0.61). Patient level: AUC survives; precPD borderline (lower bound near
  0); precET does not.
* `sys` - `sys_ft`: AUC +0.02 to +0.04 \\* at repeat level (fuse_nd vs fuse was
  +0.032 \\*), precision n.s.; at patient level AUC borderline.
* High confidence (request 0.90): `sys` PD held-out precision >= `ft`'s
  (~0.86-0.88 vs ~0.85) at higher recall; N roughly equal (~0.80-0.84; WING
  lowered N under this rule before); ET: < 1 flag per repeat for both, no 0.80
  claim; tremor screen ~0.89 for `sys` vs ~0.87 for `ft`.

Run: ``python -m experiments.combined_150 check`` then
``REPS=0-20 python -m experiments.combined_150`` (resumable; skips existing rep
files), then ``python -m experiments.combined_150 report``.
"""

from __future__ import annotations

import copy
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.training_methods_2015 import EPOCHS, LR, WD, _T, _finetune, _wt
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ft", "ft_nd", "sys", "sys_ft")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "combined_150_runs")
REF_DIR = "ckpt_ensemble_150_runs"
_TASK = re.compile(r"_(OUT|REST|WING)$")
TARGET, MIN_FLAG = 0.90, 3


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def branches(model_fn, Xtr, ytr, Xva, yva, Xout, pre, nd, seed):
    """PADS-pretrain once (as common.protocol.train), then ft and ft_nd branches."""
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
    for arm in ("ft", "ft_nd"):                   # ft first: same RNG as train()
        torch.set_rng_state(rng)
        mv = copy.deepcopy(m)
        if arm == "ft":
            mv = _finetune(mv, xt, yt, xv, yv, _wt(ytr), "ft", pre_state)
        else:
            yy = np.concatenate([ytr, yn])
            mv = _finetune(mv, torch.cat([xt, _T(Xn)]), _L(yy), xv, yv, _wt(yy),
                           "ft", pre_state)
        with torch.no_grad():
            out[arm] = [torch.softmax(mv(_T(z)), 1).cpu().numpy() for z in Xout]
    return out


def task_block(action, ids, y):
    """2015 REST / WING feature block and its row map onto the OUT patients."""
    from common.quaternion_data import load_quaternion_recordings
    from experiments.rest_2015 import block
    P, S, Y, pats, nd = block(load_quaternion_recordings("Data", action=action),
                              slice(3, 6))
    ri = {_TASK.sub("", p): k for k, p in enumerate(pats)}
    rmap = np.array([ri.get(p, -1) for p in ids])
    ok = rmap >= 0
    assert np.array_equal(Y[rmap[ok]], y[ok]), f"{action} / OUT labels disagree"
    return {"packed": P, "spec": S, "nd": nd}, rmap


def fuse(parts):
    """parts: list of (probs, has_mask, weight); weights renormalised per row."""
    z = np.zeros_like(parts[0][0])
    wsum = np.zeros(len(z))
    for p, has, w in parts:
        ww = w * has
        z += ww[:, None] * np.log(p + 1e-12)
        wsum += ww
    z /= wsum[:, None]
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _data():
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table
    A, B, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    bd = B[1].copy()
    bd[:, 10:] = 0.0                              # asymmetry + have-flag, as 2015
    ndp = {"packed": np.hstack([B[0], bd, B[2]]), "spec": B[0]}
    Cp = np.hstack([C[0], C[1], C[2]])
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = np.array([_TASK.sub("", p) for p in spectrum_table(rA, ch=slice(3, 6))[2]])
    return A, B, C, own, ndp, Cp, ids


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
    """Assert the ft branch equals common.protocol.train(pre=...) bit for bit."""
    from common.protocol import train
    torch.set_num_threads(1)
    A, B, C, own, ndp, Cp, _ = _data()
    spec, desc, _, y = A
    assert len(y) == 150, len(y)
    mk1, mk2 = members(desc.shape[1])
    kp = _pads(C, 0)
    pads = {"packed": Cp[kp], "spec": C[0][kp]}
    _, tr, va, te = next(_folds(spec, y, 0))
    for key, mk in (("packed", mk1), ("spec", mk2)):
        X, P, N = own[key], pads[key], ndp[key]
        f = zfit(X[tr])
        pre = (zfit(P)(P), C[3][kp])
        ref = train(mk, f(X[tr]), y[tr], f(X[va]), y[va], [f(X[va]), f(X[te])],
                    seed=0, pre=pre)
        br = branches(mk, f(X[tr]), y[tr], f(X[va]), y[va], [f(X[va]), f(X[te])],
                      pre, (zfit(N)(N), B[3]), 0)
        assert np.array_equal(ref[0], br["ft"][0]) and np.array_equal(ref[1], br["ft"][1]), key
        assert not np.array_equal(br["ft"][1], br["ft_nd"][1]), key
        print(f"{key}: ft branch bit-identical to common.protocol.train", flush=True)
    print("CHECK_OK", flush=True)


def run(reps):
    from experiments.rest_2015 import _scratch
    torch.set_num_threads(1)
    A, B, C, own, ndp, Cp, ids = _data()
    spec, desc, _, y = A
    yn = B[3]
    assert len(y) == 150, len(y)
    mk1, mk2 = members(desc.shape[1])
    T = {a: task_block(a, ids, y) for a in ("REST", "WING")}
    print(f"device={DEVICE}  2015 OUT n={len(y)} {np.bincount(y).tolist()}  "
          f"NewData n={len(yn)} {np.bincount(yn).tolist()}  with REST "
          f"{int((T['REST'][1] >= 0).sum())}  with WING {int((T['WING'][1] >= 0).sum())}",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    one = np.ones(len(y), bool)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        kp = _pads(C, rep)
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        side_te = {a: np.full((len(y), 3), 1 / 3) for a in T}
        for fold, tr, va, te in _folds(spec, y, rep):
            pv = {k: [] for k in ("ft", "ft_nd")}
            pt = {k: [] for k in ("ft", "ft_nd")}
            for key, mk in (("packed", mk1), ("spec", mk2)):
                X, P, N = own[key], pads[key], ndp[key]
                f = zfit(X[tr])
                per = [branches(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]),
                                (zfit(N)(N), yn), s)
                       for s in (0, 1, 2)]
                for k in ("ft", "ft_nd"):
                    pv[k].append(np.mean([r[k][0] for r in per], 0))
                    pt[k].append(np.mean([r[k][1] for r in per], 0))
            base = {k: (np.mean(pv[k], 0), np.mean(pt[k], 0)) for k in ("ft", "ft_nd")}
            side = {}
            for a, (blk, rmap) in T.items():
                has = rmap >= 0
                yr = np.full(len(blk["spec"]), -1)
                yr[rmap[has]] = y[has]
                r_tr, r_va, r_te = (rmap[i][rmap[i] >= 0] for i in (tr, va, te))
                sv = np.full((len(va), 3), 1 / 3)
                st = np.full((len(te), 3), 1 / 3)
                v, t = _scratch(blk, yr, r_tr, r_va, r_te)
                sv[has[va]], st[has[te]] = v, t
                side[a] = (sv, st, has[va], has[te])
                side_te[a][te] = st
            R, W = side["REST"], side["WING"]
            outs = dict(base)
            for arm, b in (("sys", "ft_nd"), ("sys_ft", "ft")):
                bv, bt = base[b]
                outs[arm] = (fuse([(bv, one[va], 0.5), (R[0], R[2], 0.25), (W[0], W[2], 0.25)]),
                             fuse([(bt, one[te], 0.5), (R[1], R[3], 0.25), (W[1], W[3], 0.25)]))
            for k, (v, t) in outs.items():
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        ref = f"{REF_DIR}/rep{rep:02d}.npz"
        if os.path.exists(ref) and len(np.load(ref)["y"]) == len(y):
            assert np.array_equal(np.load(ref)["ft"], pred["ft"]), \
                f"rep {rep}: ft does not reproduce {REF_DIR}"
            print(f"rep {rep}: ft reproduces {REF_DIR} ft exactly", flush=True)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()},
                 p_rest=side_te["REST"], p_wing=side_te["WING"],
                 has_rest=T["REST"][1] >= 0, has_wing=T["WING"][1] >= 0)
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} PD {Pr[k][1]:.2f} ET {Pr[k][2]:.2f}"
                                            for k in ARMS), flush=True)


def _boot(dd, n):
    bs = [np.random.default_rng(s).choice(dd, n).mean() for s in range(4000)]
    return np.percentile(bs, [2.5, 97.5])


def _hc(files):
    """Nested high-confidence rule per repeat: flags per (score, arm)."""
    from experiments import high_precision_2015 as hp
    hp.MIN_FLAG = MIN_FLAG
    out = {}
    for r, f in enumerate(files):
        d = np.load(f)
        y = d["y"]
        rep = int(re.search(r"rep(\d+)", f).group(1))
        for arm in ARMS:
            p = d[f"p_{arm}"]
            scores = {"N": (p[:, 0], y == 0), "PD": (p[:, 1], y == 1),
                      "ET": (p[:, 2], y == 2), "tremor": (p[:, 1] + p[:, 2], y != 0)}
            for c, (s, pos) in scores.items():
                flag = np.zeros(len(s), bool)
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(s, y):
                    flag[te] = s[te] >= hp.threshold(s[tr], pos[tr], TARGET)
                # cross-check against high_precision_2015.evaluate
                e = hp.evaluate(s, pos, rep, y, TARGET)
                assert e[2] == flag.sum()
                out.setdefault((c, arm), []).append((flag, pos))
    return out


def _hc_metrics(rows, w=None):
    P, Rc, K = [], [], []
    for flag, pos in rows:
        ww = np.ones(len(flag)) if w is None else w
        k = ww[flag].sum()
        tp = ww[flag & pos].sum()
        P.append(tp / k if k > 0 else np.nan)
        Rc.append(tp / ww[pos].sum())
        K.append(flag.sum())
    return np.array(P), np.array(Rc), np.array(K)


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
    print(f"2015 OUT (n={len(y)}, N/PD/ET={np.bincount(y).tolist()}), combined system, "
          f"{n} repeats\n")
    print(f"{'arm':>7}" + "".join(f"{c:>9}" for c in NAMES) + "  ET preds/rep")
    for k in ARMS:
        npred = np.mean([(np.load(f)[k] == 2).sum() for f in files])
        print(f"{k:>7}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)) + f"  {npred:.1f}")
    pairs = (("sys", "ft"), ("ft_nd", "ft"), ("sys", "sys_ft"), ("sys_ft", "ft"))
    for a, b in pairs:
        dd = R[a] - R[b]
        print(f"\n{a} - {b}  (paired over repeats, {n})")
        for i, c in enumerate(NAMES):
            lo, hi = _boot(dd[:, i], n)
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}")
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    for a, b in pairs:
        bootstrap_dir(OUT_DIR, a, b)

    # --- nested high-confidence rule (request TARGET, MIN_FLAG) ---
    H = _hc(files)
    print(f"\n=== nested high-confidence rule, request {TARGET:.2f}, MIN_FLAG {MIN_FLAG} "
          f"({n} repeats; repeat-bootstrap CI of the mean held-out precision) ===")
    for c in ("N", "PD", "ET", "tremor"):
        for arm in ARMS:
            p, rc, k = _hc_metrics(H[(c, arm)])
            ok = ~np.isnan(p)
            lo, hi = (_boot(p[ok], ok.sum()) if ok.sum() > 1 else (np.nan, np.nan))
            print(f"  {c:>6} {arm:<7} precision {np.nanmean(p):6.3f} [{lo:.3f}, {hi:.3f}]"
                  f"  recall {rc.mean():.3f}  flags/rep {k.mean():5.1f}"
                  f"  (no flags in {int((~ok).sum())})")
    # paired sys - ft on the rule, repeat level and patient level
    rng = np.random.default_rng(0)
    yy = np.load(files[0])["y"]
    groups = [np.flatnonzero(yy == c) for c in range(3)]
    draws = []
    for _ in range(2000):
        pick = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws.append(np.bincount(pick, minlength=len(yy)).astype(float))
    print("\nsys - ft under the rule (held-out precision / recall; repeat-level and "
          "patient-level 95 %; repeats where both arms flag)")
    for c in ("N", "PD", "ET", "tremor"):
        pa, ra, _ = _hc_metrics(H[(c, "sys")])
        pb, rb, _ = _hc_metrics(H[(c, "ft")])
        ok = ~np.isnan(pa) & ~np.isnan(pb)
        if ok.sum() < 2:
            print(f"  {c:>6}: too few repeats with flags in both arms ({ok.sum()})")
            continue
        dp, dr = (pa - pb)[ok], ra - rb
        lp, hp_ = _boot(dp, len(dp))
        lr, hr = _boot(dr, len(dr))
        pp, pr_ = [], []
        for w in draws:
            a1, a2, _ = _hc_metrics(H[(c, "sys")], w)
            b1, b2, _ = _hc_metrics(H[(c, "ft")], w)
            o2 = ~np.isnan(a1) & ~np.isnan(b1)
            pp.append(np.mean((a1 - b1)[o2]) if o2.any() else np.nan)
            pr_.append(np.mean(a2 - b2))
        plo, phi = np.nanpercentile(pp, [2.5, 97.5])
        rlo, rhi = np.percentile(pr_, [2.5, 97.5])
        print(f"  {c:>6}: precision {dp.mean():+.3f} rep [{lp:+.3f}, {hp_:+.3f}] "
              f"patient [{plo:+.3f}, {phi:+.3f}]  ({ok.sum()} reps) | recall "
              f"{dr.mean():+.3f} rep [{lr:+.3f}, {hr:+.3f}] patient [{rlo:+.3f}, {rhi:+.3f}]")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
