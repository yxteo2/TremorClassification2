"""L2-SP strength sweep for the 2015 OUT transfer model, then a fresh confirmation.

`transfer_2015_explore2` (40 repeats) found L2-SP fine-tuning helps ET-vs-PD
ranking with a dose-response in the anchor strength lambda, and head-only
fine-tuning (backbone frozen, ~lambda -> infinity) gives it back:

    lambda        0 (ft)   0.1     1.0     inf (head-only, round 1)
    PD-vs-ET AUC  0.627    0.631   0.645   0.628
    precET        0.338    0.341   0.369   (0.256)

So there is an interior optimum somewhere above 1. Two stages:

``sweep``    partitions 0-39 (the same as every earlier 2015 transfer run),
             arms ft, l2sp_1, l2sp_3, l2sp_10, l2sp_30.
``confirm``  the chosen lambda vs ft on 40 FRESH partitions (seeds 100-139),
             because choosing lambda on the sweep partitions and then quoting
             its numbers there is selection bias.

PREDICTION (before the sweep): an inverted U -- PD-vs-ET AUC peaks at lambda 3
or 10, at least +0.005 above lambda 1; lambda 30 falls back toward head-only
(AUC below lambda 1, precPD below ft). The confirmation keeps a significant
AUC gain over ft (measurement-derived: the explore2 dose-response), while
precET gains of ~+0.03 may not reach significance.

Run:  TREMOR_DEVICE=cuda STAGE=sweep REPS=0-8 python -m experiments.transfer_2015_l2sp
      TREMOR_DEVICE=cuda STAGE=confirm LAMBDA=<best> REPS=100-108 python -m ...
      STAGE=sweep python -m experiments.transfer_2015_l2sp report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import NAMES
from experiments.transfer_2015_explore2 import fit

STAGE = os.environ.get("STAGE", "sweep")
if STAGE == "sweep":
    LAMBDAS = {"ft": None, "l2sp_1": 1.0, "l2sp_3": 3.0, "l2sp_10": 10.0, "l2sp_30": 30.0}
else:
    lam = float(os.environ["LAMBDA"])
    LAMBDAS = {"ft": None, f"l2sp_{lam:g}": lam}
ARMS = tuple(LAMBDAS)
CAP = 90
OUT_DIR = os.environ.get("OUT_DIR", f"transfer_2015_l2sp_{STAGE}_runs")


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    print(f"device={DEVICE}  stage={STAGE}  arms={ARMS}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = (Cp[kp], C[0][kp], C[3][kp])
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            for k, lam in LAMBDAS.items():
                kw = {} if lam is None else {"ft_l2sp": lam}
                pv, pt = fit(own, pads, y, tr, va, te, **kw)
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:03d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>3}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    cols = NAMES + ("PDvsET AUC",)
    out = {k: [] for k in ARMS}
    for f in files:
        d = np.load(f); y = d["y"]; m = y != 0
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[f"pred_{k}"],
                                                         labels=[0, 1, 2], zero_division=0)
            p = d[f"prob_{k}"]
            a = roc_auc_score(y[m] == 2, np.log(p[m, 2] + 1e-12) - np.log(p[m, 1] + 1e-12))
            out[k].append([P[0], P[1], P[2], P.mean(), F.mean(), a])
    n = len(files)
    print(f"2015 OUT, stage={STAGE}, {n} CV repeats, all 151 patients, ET prevalence 0.099\n")
    print(f"{'arm':>9}" + "".join(f"{c:>11}" for c in cols))
    for k in ARMS:
        out[k] = np.array(out[k])
        print(f"{k:>9}" + "".join(f"{v:>11.3f}" for v in out[k].mean(0)))
    for k in ARMS[1:]:
        print(f"\n{k} - ft  (paired, {n} repeats)")
        for i, c in enumerate(cols):
            dlt = out[k][:, i] - out["ft"][:, i]
            b = [np.random.default_rng(s).choice(dlt, len(dlt)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>10} {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dlt > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
