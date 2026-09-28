"""Transfer learning from PADS, scored on in-house patients only.

`cohort_strategies.md` measured PADS-pretrain -> in-house-finetune at precET
-0.188 \\*, "the worst thing tried" -- but on a MERGED test set containing PADS
patients, which fine-tuning on in-house data is bound to forget. On in-house
patients against an in-house-only baseline it had never been measured.
`_inhouse_transfer_diagnostic.py` (run first) found that a PADS-fitted linear
model transfers N-vs-tremor to in-house patients perfectly (AUC 0.850 vs 0.844
in-house own CV) but not PD-vs-ET (0.589, inside its null) -- so pretraining
has something to give the N/tremor part of the network and nothing obvious for
ET.

Protocol: `own_data_10et` exactly (2015 + NewData, 10 ET per test set at
natural prevalence, 20 repeats, OUT, lower arm, reported recipe: two-stream +
ResidualTCN x 3 seeds, validation-tuned offsets). PADS (capped 90/class,
redrawn per repeat) never enters validation or test. Each domain is z-scored
with its own statistics (PADS on itself, in-house on its training fold): a
label-free normaliser over units that contain every class (invariant 13).

Arms, all on the same in-house test patients:

``scratch``      in-house only (baseline; published 0.652 / 0.769 / 0.193)
``pool``         PADS added to the training fold (own_data_10et's arm)
``ft``           pretrain 3-class on PADS, fine-tune all weights in-house
                 (lr 1e-3, 80 epochs -- `train(pre=...)` defaults)
``ft_gentle``    same, fine-tune lr 1e-4: keeps more of the PADS solution
``ft_NT``        pretrain N-vs-tremor only on PADS (the axis that transfers),
                 fine-tune 3-class in-house
``ft_shuf``      control: pretrain on PADS with labels permuted -- same data,
                 same steps, no label knowledge. ``ft`` vs ``ft_shuf`` decides
                 whether PADS *labels* transfer; vs ``scratch`` decides adoption.

PREDICTION (from the diagnostic, before the run): precET null for every
transfer arm vs scratch (|mean| < 0.05, CI spanning 0); precN up for ``ft`` and
``ft_NT`` (+0.02 to +0.05), since N-vs-tremor transfers; ``ft_NT`` >= ``ft`` on
precPD, because it cannot import PADS's reversed PD/ET frequency rule;
``ft_shuf`` ~ scratch.

Run: ``TREMOR_DEVICE=cuda python -m experiments.inhouse_transfer``
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common.protocol import DEVICE, NBIN, train, tune_offsets
from experiments.own_data_10et import build
from models.architectures import ResidualTCN, Spectrum1DCNN, TRUNKS, TwoStreamNet

REPEATS = int(os.environ.get("REPEATS", 20))
TL, ET_TEST, CAP = 64, 10, 90
ARMS = ("scratch", "pool", "ft", "ft_gentle", "ft_NT", "ft_shuf")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1")


def zfit(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    return lambda Z: (Z - mu) / sd


def members(nd):
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    return mk1, mk2


def fit(arm, own, pads, y, tr, va, te, yp, seeds=(0, 1, 2)):
    """own / pads: (packed, spec) per domain. Returns val / test probs."""
    mk1, mk2 = members(own["nd"])
    pv_l, pt_l = [], []
    for key, mk in (("packed", mk1), ("spec", mk2)):
        X, P = own[key], pads[key]
        f = zfit(X[tr])
        Xtr, Xva, Xte = f(X[tr]), f(X[va]), f(X[te])
        kw = {}
        if arm == "pool":
            Xtr = np.vstack([Xtr, zfit(P)(P)])
            ytr = np.concatenate([y[tr], yp])
        else:
            ytr = y[tr]
        if arm.startswith("ft"):
            lab = {"ft_NT": (yp != 0).astype(int),
                   "ft_shuf": np.random.default_rng(len(tr)).permutation(yp)
                   }.get(arm, yp)
            kw = dict(pre=(zfit(P)(P), lab))
            if arm == "ft_gentle":
                kw["ft_lr"] = 1e-4
        r = [train(mk, Xtr, ytr, Xva, y[va], [Xva, Xte], seed=s, **kw) for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def main():
    torch.set_num_threads(1)
    A, B, C = build()
    spec = np.vstack([A[0], B[0]]); desc = np.vstack([A[1], B[1]])
    traj = np.vstack([A[2], B[2]]); y = np.concatenate([A[3], B[3]])
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    print(f"device={DEVICE}  in-house n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  PADS n={len(C[3])}")

    frac = ET_TEST / int((y == 2).sum())
    n_te = {0: int(round(frac * (y == 0).sum())),
            1: int(round(frac * (y == 1).sum())), 2: ET_TEST}
    print(f"test per repeat: N={n_te[0]} PD={n_te[1]} ET={n_te[2]}  ET prevalence "
          f"{n_te[2] / sum(n_te.values()):.3f};  {REPEATS} repeats\n", flush=True)

    res = {k: [] for k in ARMS}
    cm = {k: np.zeros((3, 3), int) for k in ARMS}
    for rep in range(REPEATS):
        rng = np.random.default_rng(rep)
        te, rest = [], []
        for c in (0, 1, 2):
            idx = np.flatnonzero(y == c); rng.shuffle(idx)
            te.extend(idx[:n_te[c]]); rest.extend(idx[n_te[c]:])
        te = np.array(sorted(te)); rest = np.array(sorted(rest)); rng.shuffle(rest)
        n_va = max(int(0.25 * len(rest)), 12)
        va, tr = np.sort(rest[:n_va]), np.sort(rest[n_va:])
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        for arm in ARMS:
            pv, pt = fit(arm, own, pads, y, tr, va, te, C[3][kp])
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            res[arm].append([P[0], P[1], P[2], P.mean(), F.mean()])
            cm[arm] += confusion_matrix(y[te], pred, labels=[0, 1, 2])
        print(f"rep {rep:>2}  " + "  ".join(f"{k} {res[k][-1][3]:.2f}" for k in ARMS)
              + "   (macroP)", flush=True)

    print(f"\n{'arm':>10}" + "".join(f"{n:>9}" for n in NAMES) + "  |  sd precET")
    out = {}
    for k in ARMS:
        a = np.array(res[k]); out[k] = a
        print(f"{k:>10}" + "".join(f"{v:>9.3f}" for v in a.mean(0))
              + f"  |  {a[:, 2].std():.3f}")
    for base, arm in [("scratch", k) for k in ARMS[1:]] + [("ft_shuf", "ft"),
                                                           ("ft_shuf", "ft_NT")]:
        d = out[arm] - out[base]
        print(f"\n{arm} - {base}  (paired, {REPEATS} repeats)")
        for i, nm in enumerate(NAMES):
            b = [np.random.default_rng(s).choice(d[:, i], len(d)).mean()
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {nm:>8} {d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(d[:, i] > 0):.2f}")
    print("\nconfusion matrices summed over repeats (rows true N/PD/ET, cols predicted)")
    for k in ARMS:
        print(f"  {k:>10}  " + "   ".join(" ".join(f"{v:>4}" for v in row) for row in cm[k]))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
