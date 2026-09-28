"""Exploring the 2015 OUT transfer model: which part of "PADS pretrain -> 2015
fine-tune" carries the gain, and can it be pushed further?

`transfer_2015.md` (40 repeats): ``ft`` lifts precET 0.249 -> 0.333 (+0.084 \\*),
macroP +0.023 \\*, at precPD -0.040 \\*. Its gentle variant (fine-tune lr 1e-4,
keeping more of PADS) gave precPD -0.080 \\* and less ET -- so fine-tuning
strength is a live axis. Every arm below changes ONE thing about ``ft``
(scope: 2015 only, OUT only; PADS StretchHold and NewData OUT are postural,
so no action is combined).

``scratch``   2015 only (baseline)
``ft``        adopted: pretrain PADS capped 90/class, fine-tune all, lr 1e-3, 80 ep
``ft_long``   fine-tune 200 epochs instead of 80 (stronger re-fit to 2015)
``ft_head``   fine-tune the classifier only; BatchNorm frozen (linear probe)
``ft_uncap``  pretrain on all 383 PADS instead of 90/class
``ft_pn``     pretrain on PADS (capped) + NewData OUT (56, 6 ET)
``ens``       soft vote of ``ft`` and ``scratch`` (12 fits, no new training)
``ft_x2``     ensemble-size control for ``ens``: ``ft`` seeds 0-2 + seeds 3-5

Protocol, splits and seeds as `transfer_2015` (40 CV partitions, OOF on all
151, per-fold validation for early stopping + offsets, GPU). Probabilities are
saved so later analyses need no refit.

PREDICTIONS (before the run):
  * ft_long lies between ft and scratch: precPD above ft, precET below ft --
    the gentle -> ft -> scratch ordering (PD 0.671 / 0.732 / 0.771,
    ET 0.286 / 0.333 / 0.249) says fine-tune strength trades PD against
    PADS's ET knowledge. (measurement-derived)
  * ft_head is closest to ft_gentle: PD below ft (the PADS representation
    kept whole carries PADS's PD/ET frequency rule, which runs opposite to
    in-house REST and not at all in-house OUT).
  * ft_uncap: precET within +-0.03 of ft, precPD <= ft (more PADS PD).
  * ft_pn: null vs ft (NewData ET look like PD at OUT; 6 ET).
  * ens vs ft_x2: tie on macroP (|d| < 0.015); ens precPD above ft_x2,
    precET below -- it inherits scratch's PD strength and ET weakness.

Run (parts in parallel), then report:
    TREMOR_DEVICE=cuda REPS=0-10 python -m experiments.transfer_2015_explore
    python -m experiments.transfer_2015_explore report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, train, tune_offsets
from experiments.inhouse_transfer import NAMES, members, zfit
from experiments.own_data_10et import build

ARMS = ("scratch", "ft", "ft_long", "ft_head", "ft_uncap", "ft_pn", "ens", "ft_x2")
CAP = 90
OUT_DIR = os.environ.get("OUT_DIR", "transfer_2015_explore_runs")


def fit(own, pre_sets, y, tr, va, te, seeds=(0, 1, 2), **kw):
    """Reported recipe on 2015; ``pre_sets`` = list of (packed, spec, y) domains
    stacked for pretraining (each z-scored on itself), or None for scratch."""
    mk1, mk2 = members(own["nd"])
    pv_l, pt_l = [], []
    for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
        X = own[key]
        f = zfit(X[tr])
        Xtr, Xva, Xte = f(X[tr]), f(X[va]), f(X[te])
        extra = dict(kw)
        if pre_sets is not None:
            P = np.vstack([zfit(d[j])(d[j]) for d in pre_sets])
            yp = np.concatenate([d[2] for d in pre_sets])
            extra["pre"] = (P, yp)
        r = [train(mk, Xtr, y[tr], Xva, y[va], [Xva, Xte], seed=s, **extra)
             for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def run(reps):
    torch.set_num_threads(1)
    A, B, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    newd = (np.hstack([B[0], B[1], B[2]]), B[0], B[3])
    print(f"device={DEVICE}  2015 n={len(y)}  PADS n={len(C[3])}  NewData n={len(B[3])}",
          flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)          # identical draw to transfer_2015
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads_cap = (Cp[kp], C[0][kp], C[3][kp])
        pads_all = (Cp, C[0], C[3])
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            out = {
                "scratch": fit(own, None, y, tr, va, te),
                "ft": fit(own, [pads_cap], y, tr, va, te),
                "ft_long": fit(own, [pads_cap], y, tr, va, te, ft_epochs=200),
                "ft_head": fit(own, [pads_cap], y, tr, va, te, ft_head_only=True),
                "ft_uncap": fit(own, [pads_all], y, tr, va, te),
                "ft_pn": fit(own, [pads_cap, newd], y, tr, va, te),
            }
            ft2 = fit(own, [pads_cap], y, tr, va, te, seeds=(3, 4, 5))
            out["ens"] = tuple((a + b) / 2 for a, b in zip(out["ft"], out["scratch"]))
            out["ft_x2"] = tuple((a + b) / 2 for a, b in zip(out["ft"], ft2))
            for k, (pv, pt) in out.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} {P[k].mean():.2f}" for k in ARMS)
              + "  (macroP)", flush=True)


def paired(d, n):
    b = [np.random.default_rng(s).choice(d, len(d)).mean() for s in range(4000)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return d.mean(), lo, hi, ("*" if lo > 0 or hi < 0 else " "), np.mean(d > 0)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    cm = {k: np.zeros((3, 3), int) for k in ARMS}
    for f in files:
        d = np.load(f); y = d["y"]
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[f"pred_{k}"],
                                                         labels=[0, 1, 2], zero_division=0)
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean()])
            cm[k] += confusion_matrix(y, d[f"pred_{k}"], labels=[0, 1, 2])
    n = len(files)
    print(f"2015 OUT, {n} CV repeats, all 151 patients per repeat, ET prevalence 0.099\n")
    print(f"{'arm':>9}" + "".join(f"{nm:>9}" for nm in NAMES) + "  | ET preds/rep")
    out = {}
    for k in ARMS:
        a = np.array(res[k]); out[k] = a
        print(f"{k:>9}" + "".join(f"{v:>9.3f}" for v in a.mean(0))
              + f"  | {cm[k][:, 2].sum() / n:.1f}")
    pairs = [("scratch", "ft")] + [("ft", k) for k in ARMS[2:]] + [("ft_x2", "ens")]
    for base, arm in pairs:
        print(f"\n{arm} - {base}  (paired, {n} repeats)")
        for i, nm in enumerate(NAMES):
            m, lo, hi, s, w = paired(out[arm][:, i] - out[base][:, i], n)
            print(f"  {nm:>8} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] {s}  win {w:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
