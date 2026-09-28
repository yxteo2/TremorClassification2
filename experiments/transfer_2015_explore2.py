"""Transfer-learning methods for the 2015 OUT model, round 2.

Round 1 (`transfer_2015_explore`) left the recipe at a local optimum on
fine-tune *strength* -- gentle (lr 1e-4) and head-only too weak, 200 epochs too
long -- and one clue: fine-tuning longer lost ET (precET -0.021 \\*) without
recovering PD, i.e. drifting away from the pretrained weights costs what PADS
taught. This round tests transfer methods aimed at that, one change each vs
``ft`` (scope: 2015 only, OUT only; PADS StretchHold is postural):

``ft``          adopted: pretrain PADS capped 90/class for 200 epochs,
                fine-tune all weights lr 1e-3 for 80 epochs
``l2sp_lo``     L2-SP fine-tuning, lambda 0.1  (Li et al., ICML 2018)
``l2sp_hi``     L2-SP fine-tuning, lambda 1.0
``pre100``      pretrain 100 epochs instead of 200
``pre400``      pretrain 400 epochs instead of 200
``ft_rec``      pretrain on PADS *recordings* (both wrists as separate
                examples, same capped patients) instead of patient averages

Lambda scale was measured before choosing it: unconstrained fine-tuning drifts
sum||w - w0||^2 = 0.67 / 0.76 (two-stream / TCN) while its CE ends at ~0.7, so
0.1 is a mild anchor (~10 % of the loss at full drift) and 1.0 a strong one.

Protocol identical to `transfer_2015` / `_explore` (40 CV partitions, OOF on all
151, per-fold validation for early stopping + offsets, same PADS draw per
repeat, GPU); probabilities saved.

PREDICTIONS (before the run):
  * l2sp_lo ~ ft on every column (|d| < 0.02): the anchor is too weak to bind
    within 80 epochs.
  * l2sp_hi moves toward head-only: precPD below ft, precET not above ft --
    head-only (the limit of a strong anchor) was precET -0.094 \\*,
    precPD -0.052 \\*. So L2-SP does NOT beat ft at either strength.
    (measurement-derived: the ft_long / ft_head bracket)
  * pre100 / pre400 within +-0.03 precET of ft (pretraining has converged on
    197 patients by 200 full-batch epochs).
  * ft_rec: precET within +-0.03 of ft -- two wrists of one patient add
    little new ET information (28 ET either way).

Run in parts, then report:
    TREMOR_DEVICE=cuda REPS=0-8 python -m experiments.transfer_2015_explore2
    python -m experiments.transfer_2015_explore2 report
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, train, tune_offsets
from experiments.transfer_2015 import NAMES, TL, members, zfit
from experiments.own_data_10et import build

ARMS = ("ft", "l2sp_lo", "l2sp_hi", "pre100", "pre400", "ft_rec")
CAP = 90
OUT_DIR = os.environ.get("OUT_DIR", "transfer_2015_explore2_runs")


def pads_per_recording():
    """PADS StretchHold tables with one row per RECORDING, plus each row's patient.

    Features match `own_data_10et.block` exactly (multitaper logbin spectrum,
    descriptors + bilateral asymmetry + availability, IF trajectory); the
    asymmetry block is a patient property, so both wrists get their patient's.
    """
    from common.cohorts import asym_for, desc_table, logbin
    from common.loaders import load_pads_extracted
    from experiments.final_model import method_table
    from frequency.tables import spectrum_table
    from signal_processing.stability import trajectory_table
    ch = slice(0, 3)
    side = lambda r: ("left" if "LeftWrist" in str(r.path)
                      else ("right" if "RightWrist" in str(r.path) else None))
    recs = load_pads_extracted("pads_stretchhold")
    pats = spectrum_table(recs, ch=ch)[2]
    a, h = asym_for(recs, side, ch, pats)
    asym = {p: np.r_[a[i], h[i]] for i, p in enumerate(pats)}
    count = {}
    rr = []
    for r in recs:
        k = count.get(r.subject, 0); count[r.subject] = k + 1
        rr.append(replace(r, subject=f"{r.subject}#{k}"))
    S, y, keys = method_table(rr, "multitaper", ch)
    spec = logbin(S)
    traj, _, tk = trajectory_table(rr, ch=ch, n_out=TL)
    assert (tk == keys).all()
    d = desc_table(rr, ch)
    owner = np.array([k.split("#")[0] for k in keys])
    desc = np.hstack([d, np.stack([asym[p] for p in owner])])
    packed = np.hstack([spec, desc, traj.reshape(len(traj), -1)])
    return packed, spec, y, owner, pats


def fit(own, pre, y, tr, va, te, seeds=(0, 1, 2), **kw):
    """Reported recipe on 2015, pretrained on ``pre`` = (packed, spec, y)."""
    mk1, mk2 = members(own["nd"])
    pv_l, pt_l = [], []
    for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
        X = own[key]
        f = zfit(X[tr])
        Xtr, Xva, Xte = f(X[tr]), f(X[va]), f(X[te])
        P = pre[j]
        r = [train(mk, Xtr, y[tr], Xva, y[va], [Xva, Xte], seed=s,
                   pre=(zfit(P)(P), pre[2]), **kw) for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    Rp, Rs, Ry, owner, pats = pads_per_recording()
    assert Rp.shape[1] == Cp.shape[1], (Rp.shape, Cp.shape)
    pidx = {q: i for i, q in enumerate(pats)}
    assert len(pats) == len(C[3]) and all(C[3][pidx[o]] == yy for o, yy in zip(owner, Ry)), \
        "per-recording labels disagree with the patient table"
    print(f"device={DEVICE}  2015 n={len(y)}  PADS patients {len(C[3])}, "
          f"recordings {len(Ry)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)          # identical draw to transfer_2015
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = (Cp[kp], C[0][kp], C[3][kp])
        rows = np.flatnonzero(np.isin(owner, pats[kp]))
        pads_rec = (Rp[rows], Rs[rows], Ry[rows])
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            out = {
                "ft": fit(own, pads, y, tr, va, te),
                "l2sp_lo": fit(own, pads, y, tr, va, te, ft_l2sp=0.1),
                "l2sp_hi": fit(own, pads, y, tr, va, te, ft_l2sp=1.0),
                "pre100": fit(own, pads, y, tr, va, te, epochs=100),
                "pre400": fit(own, pads, y, tr, va, te, epochs=400),
                "ft_rec": fit(own, pads_rec, y, tr, va, te),
            }
            for k, (pv, pt) in out.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


def report():
    from sklearn.metrics import roc_auc_score
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    res = {k: [] for k in ARMS}
    auc = {k: [] for k in ARMS}
    for f in files:
        d = np.load(f); y = d["y"]; m = y != 0
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[f"pred_{k}"],
                                                         labels=[0, 1, 2], zero_division=0)
            res[k].append([P[0], P[1], P[2], P.mean(), F.mean()])
            p = d[f"prob_{k}"]
            auc[k].append(roc_auc_score(y[m] == 2, np.log(p[m, 2] + 1e-12)
                                        - np.log(p[m, 1] + 1e-12)))
    n = len(files)
    print(f"2015 OUT, {n} CV repeats, all 151 patients per repeat, ET prevalence 0.099\n")
    print(f"{'arm':>9}" + "".join(f"{nm:>9}" for nm in NAMES) + "   PDvsET AUC")
    out = {}
    for k in ARMS:
        a = np.array(res[k]); out[k] = np.c_[a, auc[k]]
        print(f"{k:>9}" + "".join(f"{v:>9.3f}" for v in a.mean(0))
              + f"   {np.mean(auc[k]):.3f}")
    for k in ARMS[1:]:
        print(f"\n{k} - ft  (paired, {n} repeats)")
        for i, nm in enumerate(NAMES + ("PDvsET AUC",)):
            dlt = out[k][:, i] - out["ft"][:, i]
            b = [np.random.default_rng(s).choice(dlt, len(dlt)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {nm:>10} {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dlt > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
