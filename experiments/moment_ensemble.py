"""MOMENT embeddings vs and with the 2015 OUT transfer model, under one decision rule.

`pretrained_moment` showed frozen MOMENT-1-small embeddings beat the spectrum +
descriptor features under a plain linear classifier (PD-vs-ET AUC 0.701 vs
0.621, +0.080, win 1.00), beat three weight-permuted MOMENT controls by
0.08-0.17 AUC, and clear a label-permutation null (binary AUC 0.710, null top
0.686, p 0.020). But that comparison used argmax decisions, while the adopted
deep model tunes per-class logit offsets on validation. This puts everything
under the same protocol as `transfer_2015` (same 40 partitions, folds,
validation splits, PADS draws, offset tuning) so precision is comparable:

``ft``         adopted 2015 OUT model (PADS pretrain -> fine-tune), GPU
``mom_lr``     L2 logistic regression on patient-mean MOMENT embeddings
               (band-passed; fixed features, fitted per fold)
``ens``        soft vote of ft and mom_lr (mean of probabilities)
``ens_ctrl``   soft vote of ft and the same logistic on weight-PERMUTED MOMENT
               (seed 0) -- controls for adding any second member

PREDICTIONS (before the run):
  * mom_lr PD-vs-ET AUC above ft by >= +0.04 (0.70 linear vs ~0.63 deep);
    precET within +-0.05 of ft (precision hinges on the offsets).
  * ens beats ft on PD-vs-ET AUC and macroP; ens beats ens_ctrl on PD-vs-ET AUC.

Needs the embeddings from `pretrained_moment` (``WORK`` folder).
Run:  TREMOR_DEVICE=cuda WORK=... REPS=0-8 python -m experiments.moment_ensemble
      WORK=... python -m experiments.moment_ensemble report
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
from sklearn.preprocessing import StandardScaler

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.pretrained_moment import _patient_mean
from experiments.transfer_2015 import NAMES, fit

WORK = os.environ.get("WORK", "moment_runs")
OUT_DIR = os.environ.get("OUT_DIR", f"{WORK}/ensemble_runs")
ARMS = ("ft", "mom_lr", "ens", "ens_ctrl")
CAP = 90


def _embeddings(y):
    w = np.load(f"{WORK}/windows.npz")
    lab = dict(zip(w["subj"], w["lab"]))
    E, keys = _patient_mean(np.load(f"{WORK}/embeddings.npz")["bp"], w["rec_of"], w["subj"])
    Ep, keys_p = _patient_mean(np.load(f"{WORK}/embeddings_perm0.npz")["bp"], w["rec_of"], w["subj"])
    assert (keys == keys_p).all() and len(keys) == len(y)
    assert all(lab[k] == yy for k, yy in zip(keys, y)), "patient order/labels differ from build()"
    return E, Ep


def _lr(E, y, tr, va, te):
    sc = StandardScaler().fit(E[tr])
    m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(sc.transform(E[tr]), y[tr])
    return m.predict_proba(sc.transform(E[va])), m.predict_proba(sc.transform(E[te]))


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    E, Ep = _embeddings(y)
    print(f"device={DEVICE}  2015 n={len(y)}  MOMENT dim {E.shape[1]}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        rng = np.random.default_rng(rep)              # identical PADS draw to transfer_2015
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            ft = fit("ft", own, pads, y, tr, va, te, C[3][kp])
            mo = _lr(E, y, tr, va, te)
            mp = _lr(Ep, y, tr, va, te)
            out = {"ft": ft, "mom_lr": mo,
                   "ens": ((ft[0] + mo[0]) / 2, (ft[1] + mo[1]) / 2),
                   "ens_ctrl": ((ft[0] + mp[0]) / 2, (ft[1] + mp[1]) / 2)}
            for k, (pv, pt) in out.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:03d}.npz", y=y,
                 **{f"pred_{k}": v for k, v in pred.items()},
                 **{f"prob_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>3}  " + "  ".join(f"{k} ET {P[k][2]:.2f}" for k in ARMS), flush=True)


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    cols = NAMES + ("PDvsET AUC",)
    out = {k: [] for k in ARMS}
    for f in files:
        d = np.load(f); y = d["y"]; m = y != 0
        for k in ARMS:
            P, _, F, _ = precision_recall_fscore_support(y, d[f"pred_{k}"], labels=[0, 1, 2],
                                                         zero_division=0)
            p = d[f"prob_{k}"]
            a = roc_auc_score(y[m] == 2, np.log(p[m, 2] + 1e-12) - np.log(p[m, 1] + 1e-12))
            out[k].append([P[0], P[1], P[2], P.mean(), F.mean(), a])
    n = len(files)
    print(f"2015 OUT, {n} CV repeats, all 151 patients, ET prevalence 0.099, offsets tuned on validation\n")
    print(f"{'arm':>9}" + "".join(f"{c:>11}" for c in cols))
    for k in ARMS:
        out[k] = np.array(out[k])
        print(f"{k:>9}" + "".join(f"{v:>11.3f}" for v in out[k].mean(0)))
    for base, k in (("ft", "mom_lr"), ("ft", "ens"), ("ens_ctrl", "ens"), ("ft", "ens_ctrl")):
        print(f"\n{k} - {base}  (paired, {n} repeats)")
        for i, c in enumerate(cols):
            dlt = out[k][:, i] - out[base][:, i]
            b = [np.random.default_rng(s).choice(dlt, len(dlt)).mean() for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"  {c:>10} {dlt.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{'*' if lo > 0 or hi < 0 else ' '}  win {np.mean(dlt > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-40").split("-"))
        run(range(a, b))
