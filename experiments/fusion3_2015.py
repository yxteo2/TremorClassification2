"""A third task: does a 2015 WING model add to OUT + 0.25 REST?

`fusion_2015.md`: `ft` (OUT) + 0.25 x REST scratch is the recommended 2015 model
-- precN +0.016 \\*, AUC +0.023-0.029 \\* on both partition sets, ET unchanged.
2015 also records WING (wing-beating posture). At feature level WING carries no
PD-vs-ET signal (6 characteristics: AUC 0.526, null p = 0.39) but detects tremor
as well as OUT (N vs tremor 0.893). Its use is the N/PD side: the controls `ft`
calls PD have tremor at OUT only -- at control level in WING (log RMS −2.00 vs
−2.08, `stack_2015.md`).

## Arms (one action per model; log-probabilities averaged with fixed weights)

    base        ft(OUT) + 0.25 REST                      (= fusion_2015 fuse_w25)
    + WING      ft(OUT) + 0.25 REST + 0.25 WING scratch
    + OUT2      ft(OUT) + 0.25 REST + 0.25 OUT scratch   -- size-matched control:
                a third model of the same recipe, on the task already used

Weights are fixed in advance (normalised: 0.5 / 0.25 / 0.25 for three models).
A patient with no recording for a task drops that term. Partitions, validation
splits and seeds are `transfer_2015`'s; offsets are tuned on the fused
validation probabilities exactly as there.

## Prediction, recorded before the run

**+ WING beats + OUT2 on precPD (+0.01 to +0.03) and precN, ET flat** -- WING
repeats REST's job (tremor presence in a second posture) more weakly, because a
posture task shares OUT's false positives more than rest does.

Run: ``REPS=0-14 python -m experiments.fusion3_2015`` ... then
``python -m experiments.fusion3_2015 report``.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, fit

ARMS = ("base", "+ WING", "+ OUT2")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET")
OUT_DIR = os.environ.get("OUT_DIR", "fusion3_2015_runs")
_TASK = re.compile(r"_(OUT|REST|WING)$")


def task_block(action, ids, y):
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


def run(reps):
    from common.quaternion_data import load_quaternion_recordings
    from experiments.rest_2015 import _scratch
    from frequency.tables import spectrum_table
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = np.array([_TASK.sub("", p) for p in spectrum_table(rA, ch=slice(3, 6))[2]])
    T = {a: task_block(a, ids, y) for a in ("REST", "WING")}
    Cp = np.hstack([C[0], C[1], C[2]])
    print(f"device={DEVICE}  2015 OUT n={len(y)}  with REST "
          f"{int((T['REST'][1] >= 0).sum())}  with WING {int((T['WING'][1] >= 0).sum())}",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    one = np.ones(len(y), bool)
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
            pv_o, pt_o = fit("ft", own, pads, y, tr, va, te, C[3][kp])
            pv_2, pt_2 = fit("scratch", own, pads, y, tr, va, te, C[3][kp])
            side = {}
            for a, (blk, rmap) in T.items():
                has = rmap >= 0
                yr = np.full(len(blk["spec"]), -1)
                yr[rmap[has]] = y[has]
                r_tr, r_va, r_te = (rmap[i][rmap[i] >= 0] for i in (tr, va, te))
                pv = np.full((len(va), 3), 1 / 3)
                pt = np.full((len(te), 3), 1 / 3)
                v, t = _scratch(blk, yr, r_tr, r_va, r_te)
                pv[has[va]], pt[has[te]] = v, t
                side[a] = (pv, pt, has[va], has[te])
            R, W = side["REST"], side["WING"]
            outs = {
                "base": (fuse([(pv_o, one[va], 0.75), (R[0], R[2], 0.25)]),
                         fuse([(pt_o, one[te], 0.75), (R[1], R[3], 0.25)])),
                "+ WING": (fuse([(pv_o, one[va], 0.5), (R[0], R[2], 0.25),
                                 (W[0], W[2], 0.25)]),
                           fuse([(pt_o, one[te], 0.5), (R[1], R[3], 0.25),
                                 (W[1], W[3], 0.25)])),
                "+ OUT2": (fuse([(pv_o, one[va], 0.5), (R[0], R[2], 0.25),
                                 (pv_2, one[va], 0.25)]),
                           fuse([(pt_o, one[te], 0.5), (R[1], R[3], 0.25),
                                 (pt_2, one[te], 0.25)]))}
            for k, (pv, pt) in outs.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        # assert first: base must be fusion_2015's fuse_w25, bit for bit
        ref = f"fusion_2015_runs/rep{rep:02d}.npz"
        if os.path.exists(ref):
            assert np.array_equal(np.load(ref)["fuse_w25"], pred["base"]), \
                f"rep {rep}: base does not reproduce fusion_2015 fuse_w25"
            print(f"rep {rep}: base reproduces fusion_2015 fuse_w25 exactly", flush=True)
        np.savez(f"{OUT_DIR}/rep{rep:03d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>3}  " + "  ".join(f"{k} N {P[k][0]:.2f} PD {P[k][1]:.2f} "
                                            f"ET {P[k][2]:.2f}" for k in ARMS), flush=True)


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
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean(),
                           roc_auc_score((y[m] == 2).astype(int), s)])
    n = len(files)
    R = {k: np.array(v) for k, v in res.items()}
    print(f"2015, OUT + REST (+ WING / + OUT2), {n} repeats\n")
    print(f"{'arm':>8}" + "".join(f"{c:>9}" for c in NAMES))
    for k in ARMS:
        print(f"{k:>8}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)))
    for a, b in (("+ WING", "+ OUT2"), ("+ WING", "base"), ("+ OUT2", "base")):
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
