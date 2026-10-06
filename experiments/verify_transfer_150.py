"""Verification: does the 2015 PADS-transfer headline survive the PD 12 fix?

Every saved 2015 run used the pre-fix 151-row OUT table, in which "PD 12_OUT"
and "PD 12_OUT N LOAD" (one person) were two subjects that could land in train
and test of the same fold. `common.quaternion_data.canonical_subject` now merges
them: 2015 OUT is **150 patients (61 N / 74 PD / 15 ET)**. This re-runs the
headline contrast -- `ft` (PADS StretchHold pretrain -> 2015 fine-tune) vs
`scratch` (2015 only) -- on the corrected table.

## Protocol -- `transfer_2015.py`'s, reused, not copied

2015 OUT only, stratified 5-fold CV (``StratifiedKFold(5, shuffle=True,
random_state=rep)``), every patient predicted once out-of-fold, a stratified
25 % validation split inside each training fold (``random_state=rep*10+fold``)
for early stopping and logit offsets; PADS capped 90/class, redrawn per repeat
with ``default_rng(rep)`` exactly as `transfer_2015.run`; `transfer_2015.fit`
unchanged (reported recipe, two-stream + ResidualTCN x 3 seeds). 40 repeats,
seeds 0-39. CPU, one thread (the 151-row runs were on GPU, so even without the
fix the numbers would not be bit-identical).

Saved per repeat (``verify_transfer_150_runs/repNN.npz``): ``scratch``, ``ft``
(out-of-fold predicted class), ``p_scratch``, ``p_ft`` (out-of-fold
probabilities, before the validation-tuned offsets), ``y``.

Reported both as the per-repeat paired bootstrap (re-partitioning noise only)
and at the patient level with `patient_bootstrap_2015.bootstrap_dir`
(resamples patients -- the generalisation test).

## Prediction, recorded before the run

151-row reference: precET 0.249 -> 0.333 / 0.350, ft - scratch +0.084..+0.094
(repeat-level \\*), patient-level +0.092 [+0.012, +0.211]; PD-vs-ET AUC +0.053
(patient-level n.s.); precPD about -0.04.

Removing one duplicated PD row moves no ET patient, so **the sign survives**:
precET ft - scratch **+0.06 to +0.10, repeat-level significant**; precPD cost
-0.02 to -0.05; PD-vs-ET AUC +0.04 to +0.06 repeat-level \\*. The patient-level
precET interval was only just clear of 0 (lower bound +0.012), so with CPU vs GPU
noise **I expect its lower bound to sit within +-0.02 of 0 -- a coin flip
whether it still excludes 0**; patient-level AUC stays n.s.

Run: ``nohup python -m experiments.verify_transfer_150 > verify_transfer_150.log 2>&1 &``
(resumable: skips repeats whose file exists; ``REPS=a-b`` selects seeds), then
``python -m experiments.verify_transfer_150 report``.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, fit

ARMS = ("scratch", "ft")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET")
OUT_DIR = os.environ.get("OUT_DIR", "verify_transfer_150_runs")


def check(y):
    counts = [int((y == c).sum()) for c in (0, 1, 2)]
    assert len(y) == 150 and counts == [61, 74, 15], (len(y), counts)
    return counts


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    counts = check(y)
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    print(f"device={DEVICE}  2015 n={len(y)} N/PD/ET={counts}  PADS n={len(C[3])}",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        out = f"{OUT_DIR}/rep{rep:02d}.npz"
        if os.path.exists(out):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            for arm in ARMS:
                pv, pt = fit(arm, own, pads, y, tr, va, te, C[3][kp])
                prob[arm][te] = pt
                pred[arm][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        assert all((pred[k] >= 0).all() for k in ARMS)
        np.savez(out + ".tmp.npz", y=y, **pred, **{f"p_{k}": v for k, v in prob.items()})
        os.replace(out + ".tmp.npz", out)
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep??.npz"))
    res = {k: [] for k in ARMS}
    cm = {k: np.zeros((3, 3), int) for k in ARMS}
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
            cm[k] += confusion_matrix(y, d[k], labels=[0, 1, 2])
    n = len(files)
    counts = check(y)
    print(f"2015 OUT, {n} CV repeats, {len(y)} patients each "
          f"(N/PD/ET = {counts}, ET prevalence {(y == 2).mean():.3f})\n")
    print(f"{'arm':>10}" + "".join(f"{c:>9}" for c in NAMES) + "  | ET preds/rep")
    R = {k: np.array(v) for k, v in res.items()}
    for k in ARMS:
        print(f"{k:>10}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0))
              + f"  | {cm[k][:, 2].sum() / n:.1f}")
    d = R["ft"] - R["scratch"]
    print(f"\nft - scratch  (repeat-level paired bootstrap, {n} repeats, 4000 draws)")
    for i, c in enumerate(NAMES):
        b = [np.random.default_rng(s).choice(d[:, i], len(d)).mean() for s in range(4000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {c:>8} {d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
              f"  win {np.mean(d[:, i] > 0):.2f}")
    print("\nconfusion matrices, mean per repeat (rows true N/PD/ET, cols predicted)")
    for k in ARMS:
        mm = cm[k] / n
        print(f"  {k:>10}  " + "  |  ".join(" ".join(f"{v:5.1f}" for v in row) for row in mm))
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    bootstrap_dir(OUT_DIR, "ft", "scratch", tag="patient level (150 rows)")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
