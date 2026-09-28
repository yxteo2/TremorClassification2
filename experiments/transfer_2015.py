"""Transfer learning from PADS for the 2015 cohort, OUT task only.

Scope (user, 2026-09-27): develop on 2015 only, one action per model, OUT
first, never combine actions. PADS StretchHold is also a postural task, so
using it as a transfer source does not combine actions.

Why transfer at all: `_inhouse_transfer_diagnostic.py` found a PADS-fitted
linear model transfers N-vs-tremor to in-house patients perfectly (AUC 0.850 vs
0.844 own CV) but PD-vs-ET only to 0.589 (inside its null). The earlier
negative (`cohort_strategies.md`, precET -0.188 \\*) was scored on a merged test
set containing PADS patients; on 2015 patients it was never measured.

Protocol. 2015 holds 61 N / 75 PD / 15 ET -- too few ET for a held-out test
set, so: stratified 5-fold CV, every patient predicted once out-of-fold, per-
class precision computed on all 151 at natural prevalence (ET 0.099); repeated
over 20 partitions, arms paired on the same partitions. Inside each training
fold a stratified 25 % validation split drives early stopping and the logit
offsets. PADS (capped 90/class, redrawn per repeat) never enters validation or
test. Each domain is z-scored on its own statistics (label-free, every class
present in both -- invariant 13). The reported recipe (two-stream +
ResidualTCN x 3 seeds), GPU via ``TREMOR_DEVICE=cuda``.

Arms (from ``inhouse_transfer``): scratch, pool, ft, ft_gentle, ft_NT, ft_shuf.

PREDICTION (before the run): precET null for every transfer arm vs scratch
(|mean| < 0.05, CI spanning 0), because PD-vs-ET does not transfer; precN up
for ``ft`` and ``ft_NT`` (+0.02 to +0.05) because N-vs-tremor does;
``ft_shuf`` ~ scratch.

Run in parts (parallel on one GPU), then report:
    TREMOR_DEVICE=cuda REPS=0-5 python -m experiments.transfer_2015
    ...
    python -m experiments.transfer_2015 report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.inhouse_transfer import ARMS, NAMES, fit
from experiments.own_data_10et import build

CAP = 90
OUT_DIR = os.environ.get("OUT_DIR", "transfer_2015_runs")


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    print(f"device={DEVICE}  2015 n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  PADS n={len(C[3])}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            for arm in ARMS:
                pv, pt = fit(arm, own, pads, y, tr, va, te, C[3][kp])
                pred[arm][te] = (np.log(pt + 1e-12)
                                 + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred)
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    cm = {k: np.zeros((3, 3), int) for k in ARMS}
    for f in files:
        d = np.load(f)
        y = d["y"]
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[k], labels=[0, 1, 2],
                                                         zero_division=0)
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean()])
            cm[k] += confusion_matrix(y, d[k], labels=[0, 1, 2])
    n = len(files)
    print(f"2015 OUT, {n} CV repeats, all 151 patients per repeat "
          f"(N/PD/ET = {[int((y == c).sum()) for c in (0, 1, 2)]}, ET prevalence "
          f"{(y == 2).mean():.3f})\n")
    print(f"{'arm':>10}" + "".join(f"{nm:>9}" for nm in NAMES) + "  | ET preds/rep")
    out = {}
    for k in ARMS:
        a = np.array(res[k]); out[k] = a
        print(f"{k:>10}" + "".join(f"{v:>9.3f}" for v in a.mean(0))
              + f"  | {cm[k][:, 2].sum() / n:.1f}")
    for base, arm in [("scratch", k) for k in ARMS[1:]] + [("ft_shuf", "ft"),
                                                           ("ft_shuf", "ft_NT")]:
        d = out[arm] - out[base]
        print(f"\n{arm} - {base}  (paired, {n} repeats)")
        for i, nm in enumerate(NAMES):
            b = [np.random.default_rng(s).choice(d[:, i], len(d)).mean()
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {nm:>8} {d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(d[:, i] > 0):.2f}")
    print("\nconfusion matrices, mean per repeat (rows true N/PD/ET, cols predicted)")
    for k in ARMS:
        m = cm[k] / n
        print(f"  {k:>10}  " + "  |  ".join(" ".join(f"{v:5.1f}" for v in row) for row in m))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
