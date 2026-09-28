"""A separate 2015 REST model (never combined with OUT), with transfer from PADS Relaxed.

Why REST. Rest tremor is the PD-specific clinical sign, and 2015 REST is the only
in-house task where PD vs ET is above chance on simple characteristics (AUC
0.65-0.70 on all three sensors, `inhouse_pd_vs_et.md`); OUT is at chance there.
The earlier deep REST test (`inhouse_rest_deep`, git history) pooled 2015 with
NewData and used no transfer, so this is untested in scope (2015 only, one
action per model).

The complication. PADS ET are *slower* than PD at rest (Relaxed max_freq ET 4.83
vs PD 6.05 Hz) while 2015 ET are *faster* (6.05 vs 5.57): the two sources teach
opposite frequency rules. So transfer may carry N-vs-tremor and hurt PD-vs-ET.

Two stages, same file:

``diag``   (CPU, no deep fits) linear PADS Relaxed -> 2015 REST transfer AUC,
           per axis, vs 2015's own CV AUC and a label-permutation null.
``deep``   40 CV partitions on 2015 REST, OOF on every patient, arms:
           scratch, ft (pretrain PADS Relaxed capped 90/class, fine-tune all),
           ft_shuf (control: pretrain with PADS labels permuted); plus an
           L2-SP arm when ``LAMBDA`` is set (the OUT sweep's choice).

Features are the OUT pipeline's, built the same way (multitaper 16-bin log
spectrum, 10 descriptors, bilateral asymmetry + availability, IF trajectory),
lower-arm sensor for 2015 (the pipeline convention; not re-chosen here).
2015's four `N 2` accelerometer files are skipped by the loader.

PREDICTIONS (before either stage runs):
  * diag: PADS -> 2015 N-vs-tremor AUC > 0.70 (tremor presence transfers, as at
    OUT); PD-vs-ET transfer AUC < 0.5 or inside the null (opposite frequency
    rule), while 2015's own PD-vs-ET CV AUC is above its null.
  * deep: scratch precET above chance (0.106) but modest; ft does NOT beat
    scratch on precET or PD-vs-ET AUC (the opposite rule), unlike OUT; ft
    beats ft_shuf on precN.

Run:  python -m experiments.rest_2015 diag
      TREMOR_DEVICE=cuda STAGE=deep REPS=0-8 python -m experiments.rest_2015
      STAGE=deep python -m experiments.rest_2015 report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.transfer_2015 import NAMES, TL, zfit
from experiments.transfer_2015_explore2 import fit

CAP = 90
LAMBDA = os.environ.get("LAMBDA")
ARMS = ("scratch", "ft", "ft_shuf") + ((f"l2sp_{float(LAMBDA):g}",) if LAMBDA else ())
OUT_DIR = os.environ.get("OUT_DIR", "rest_2015_runs")


def block(recs, ch, side_fn=None):
    """OUT-pipeline features for any task: (packed, spec, y, patients)."""
    from common.cohorts import N_ASYM, asym_for, desc_table, logbin
    from experiments.final_model import method_table
    from signal_processing.stability import trajectory_table
    S, y, pats = method_table(recs, "multitaper", ch)
    spec = logbin(S)
    traj, _, tp = trajectory_table(recs, ch=ch, n_out=TL)
    assert (tp == pats).all()
    d = desc_table(recs, ch)
    if side_fn is not None:
        a, h = asym_for(recs, side_fn, ch, pats)
    else:
        a, h = np.zeros((len(pats), N_ASYM)), np.zeros(len(pats))
    desc = np.hstack([d, a, h[:, None]])
    return np.hstack([spec, desc, traj.reshape(len(traj), -1)]), spec, y, pats, desc.shape[1]


def load():
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    side = lambda r: ("left" if "LeftWrist" in str(r.path)
                      else ("right" if "RightWrist" in str(r.path) else None))
    own = block(load_quaternion_recordings("Data", action="REST"), slice(3, 6))
    pads = block(load_pads_extracted("pads_relaxed"), slice(0, 3), side)
    return own, pads


def diag():
    own, pads = load()
    Xo, yo, Xp, yp = own[0], own[2], pads[0], pads[2]
    Xo, Xp = zfit(Xo)(Xo), zfit(Xp)(Xp)            # each domain on its own stats
    print(f"2015 REST n={len(yo)} N/PD/ET={np.bincount(yo).tolist()}   "
          f"PADS Relaxed n={len(yp)} N/PD/ET={np.bincount(yp).tolist()}\n")
    mdl = lambda: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")

    def own_cv(X, y, reps=10):
        a = []
        for r in range(reps):
            s = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
                s[te] = mdl().fit(X[tr], y[tr]).decision_function(X[te])
            a.append(roc_auc_score(y, s))
        return float(np.mean(a))

    for axis in ("N vs tremor", "PD vs ET"):
        if axis == "N vs tremor":
            sel = lambda X, y: (X, (y != 0).astype(int))
        else:
            sel = lambda X, y: (X[y != 0], (y[y != 0] == 2).astype(int))
        Xt, yt = sel(Xp, yp); Xi, yi = sel(Xo, yo)
        s = mdl().fit(Xt, yt).decision_function(Xi)
        tr_auc = roc_auc_score(yi, s)
        rng = np.random.default_rng(0)
        null = [roc_auc_score(rng.permutation(yi), s) for _ in range(1000)]
        lo, hi = np.percentile(null, [2.5, 97.5])
        own_auc = own_cv(Xi, yi)
        onull = [own_cv(Xi, rng.permutation(yi), reps=2) for _ in range(100)]
        olo, ohi = np.percentile(onull, [2.5, 97.5])
        print(f"{axis:>12}: PADS->2015 transfer AUC {tr_auc:.3f} (null [{lo:.3f}, {hi:.3f}])"
              f"   2015 own CV AUC {own_auc:.3f} (null [{olo:.3f}, {ohi:.3f}])", flush=True)
    print("MARKER_DONE", flush=True)


def run(reps):
    torch.set_num_threads(1)
    own_b, pads_b = load()
    own = {"packed": own_b[0], "spec": own_b[1], "nd": own_b[4]}
    y = own_b[2]
    Cp, Cs, Cy = pads_b[0], pads_b[1], pads_b[2]
    print(f"device={DEVICE}  2015 REST n={len(y)} N/PD/ET={np.bincount(y).tolist()}  "
          f"PADS Relaxed n={len(Cy)}  arms={ARMS}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(Cy == c),
                                                min(CAP, int((Cy == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = (Cp[kp], Cs[kp], Cy[kp])
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(own["spec"], y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            out = {"scratch": _scratch(own, y, tr, va, te),
                   "ft": fit(own, pads, y, tr, va, te),
                   "ft_shuf": fit(own, (pads[0], pads[1],
                                        np.random.default_rng(len(tr)).permutation(pads[2])),
                                  y, tr, va, te)}
            if LAMBDA:
                out[ARMS[-1]] = fit(own, pads, y, tr, va, te, ft_l2sp=float(LAMBDA))
            for k, (pv, pt) in out.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:03d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>3}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS),
              flush=True)


def _scratch(own, y, tr, va, te, seeds=(0, 1, 2)):
    """Reported recipe on 2015 REST only, no pretraining."""
    from common.protocol import train
    from experiments.transfer_2015 import members
    mk1, mk2 = members(own["nd"])
    pv_l, pt_l = [], []
    for key, mk in (("packed", mk1), ("spec", mk2)):
        X = own[key]; f = zfit(X[tr])
        r = [train(mk, f(X[tr]), y[tr], f(X[va]), y[va], [f(X[va]), f(X[te])], seed=s)
             for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    arms = [k[5:] for k in np.load(files[0]).files if k.startswith("pred_")]
    cols = NAMES + ("PDvsET AUC",)
    out = {k: [] for k in arms}
    for f in files:
        d = np.load(f); y = d["y"]; m = y != 0
        for k in arms:
            P, _, F, _ = precision_recall_fscore_support(y, d[f"pred_{k}"],
                                                         labels=[0, 1, 2], zero_division=0)
            p = d[f"prob_{k}"]
            a = roc_auc_score(y[m] == 2, np.log(p[m, 2] + 1e-12) - np.log(p[m, 1] + 1e-12))
            out[k].append([P[0], P[1], P[2], P.mean(), F.mean(), a])
    n = len(files)
    print(f"2015 REST, {n} CV repeats, all {len(y)} patients, N/PD/ET="
          f"{np.bincount(y).tolist()}, ET prevalence {(y == 2).mean():.3f}\n")
    print(f"{'arm':>9}" + "".join(f"{c:>11}" for c in cols))
    for k in arms:
        out[k] = np.array(out[k])
        print(f"{k:>9}" + "".join(f"{v:>11.3f}" for v in out[k].mean(0)))
    pairs = [("scratch", k) for k in arms if k != "scratch"] + [("ft_shuf", "ft")]
    for base, k in pairs:
        print(f"\n{k} - {base}  (paired, {n} repeats)")
        for i, c in enumerate(cols):
            dlt = out[k][:, i] - out[base][:, i]
            b = [np.random.default_rng(s).choice(dlt, len(dlt)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>10} {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dlt > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["diag"]:
        diag()
    elif sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
