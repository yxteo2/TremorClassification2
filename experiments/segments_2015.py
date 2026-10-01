"""Multi-segment tremor coupling in the 2015 OUT model.

`multisegment.py` found, on 2015 OUT, that eight coupling features between the
hand, lower-arm and upper-arm sensors separate PD from ET at CV AUC 0.686
against a shuffled-label null of [0.307, 0.697] (p = 0.045), where the six
spectral characteristics read 0.29-0.41. It is driven by hand-forearm
coherence, lower in ET: PD tremor is largely forearm rotation, which carries the
hand with it; ET is more wrist flexion-extension, the hand moving against the
forearm (IEEE TNSRE 2023, multi-segment PD-vs-ET). The pipeline reads only the
lower arm. Borderline at 15 ET, so this asks the model.

## Protocol -- `transfer_2015.py`'s, reused, not copied

2015 OUT only, 151 patients, stratified 5-fold, every patient predicted once
out-of-fold, 25 % validation inside each training fold; `transfer_2015.fit` is
called unchanged. The coupling block (8 features) is appended to the descriptor
stream, so only the two-stream member sees it; the TCN reads the spectrum.

    scratch    2015 only, reported recipe                 (= transfer_2015)
    seg        + coupling block
    seg_shuf   + coupling block, rows permuted among the 151 patients,
               re-drawn every repeat                       (attribution)
    ft         PADS StretchHold pretrain -> 2015 fine-tune (= transfer_2015)
    ft_seg     ft + coupling block; PADS has one wrist sensor, so its block is
               zero during pretraining and the fine-tune learns it

Saved per repeat: out-of-fold predictions and probabilities, so PD-vs-ET AUC is
computed from the same fits.

## Prediction, recorded before the run

**seg beats seg_shuf on PD-vs-ET AUC by +0.02 to +0.05 (significant), but not on
ET precision** -- the feature-level signal is real but small at 15 ET, and this
project's record is that feature-level signals move ranking before they move a
thresholded precision. **ft_seg vs ft: same pattern, smaller**, because the block
enters only at fine-tuning.

Run (CPU, in parallel parts): ``REPS=0-5 python -m experiments.segments_2015``
... then ``python -m experiments.segments_2015 report``.
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
from experiments.multisegment import FEATS, seg_table
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, fit

ARMS = ("scratch", "seg", "seg_shuf", "ft", "ft_seg")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET")
OUT_DIR = os.environ.get("OUT_DIR", "segments_2015_runs")


def tables():
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table
    A, _, C = build()
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = spectrum_table(rA, ch=slice(3, 6))[2]
    S, ys, sp = seg_table(rA)
    idx = {p: i for i, p in enumerate(sp)}
    assert all(p in idx for p in ids), "a 2015 patient has no segment features"
    seg = S[[idx[p] for p in ids]]
    assert np.array_equal(ys[[idx[p] for p in ids]], A[3]), "label order mismatch"
    return A, C, seg


def run(reps):
    torch.set_num_threads(1)
    A, C, seg = tables()
    spec, desc, traj, y = A
    nd, ns = desc.shape[1], seg.shape[1]
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": nd}
    Cp = np.hstack([C[0], C[1], C[2]])
    Cs = np.hstack([C[0], C[1], np.zeros((len(C[3]), ns)), C[2]])
    print(f"device={DEVICE}  2015 n={len(y)}  coupling block {ns} features",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        shuf = seg[np.random.default_rng(5000 + rep).permutation(len(y))]
        o_seg = {"packed": np.hstack([spec, desc, seg, traj]), "spec": spec,
                 "nd": nd + ns}
        o_shf = {"packed": np.hstack([spec, desc, shuf, traj]), "spec": spec,
                 "nd": nd + ns}
        p0 = {"packed": Cp[kp], "spec": C[0][kp]}
        ps = {"packed": Cs[kp], "spec": C[0][kp]}
        plan = {"scratch": ("scratch", own, p0), "seg": ("scratch", o_seg, p0),
                "seg_shuf": ("scratch", o_shf, p0), "ft": ("ft", own, p0),
                "ft_seg": ("ft", o_seg, ps)}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            for k, (mode, o, p) in plan.items():
                pv, pt = fit(mode, o, p, y, tr, va, te, C[3][kp])
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, **pred,
                 **{f"p_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


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
    print(f"2015 OUT, {n} CV repeats, 151 patients each\n")
    print(f"{'arm':>10}" + "".join(f"{c:>9}" for c in NAMES))
    R = {k: np.array(v) for k, v in res.items()}
    for k in ARMS:
        print(f"{k:>10}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)))
    for arm, base, tag in (("seg", "scratch", "adoption"),
                           ("seg", "seg_shuf", "attribution"),
                           ("seg_shuf", "scratch", "control vs baseline"),
                           ("ft_seg", "ft", "adoption on top of transfer"),
                           ("ft", "scratch", "transfer (reproduces transfer_2015)")):
        d = R[arm] - R[base]
        print(f"\n{arm} - {base}  ({tag}, paired, {n} repeats)")
        for i, c in enumerate(NAMES):
            b = [np.random.default_rng(s).choice(d[:, i], len(d)).mean()
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(d[:, i] > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
