"""Late fusion of the 2015 OUT transfer model and a 2015 REST model.

## Why

`stack_2015.md` found 5 of `ft`'s 27 consistent errors are controls with tremor
at OUT but none at REST, and that adding REST tremor amplitude to `ft`'s
probabilities raised precN / precPD by +0.02 \\* with ET flat -- the one
remaining lever with a measured positive effect. `inhouse_rest_deep` (git
history) had OUT+REST score fusion at precPD +0.059 \\* (20 repeats, merged
in-house). Each model here sees ONE action; only their outputs are combined, so
the "one action per model" scope holds.

## Protocol -- `transfer_2015`'s, with its code reused

Partitions over the 151 OUT patients exactly as `transfer_2015.run`
(StratifiedKFold(5, rep), 25 % validation, same seeds), so `ft` reproduces it.
The REST model (`rest_2015._scratch`, 2015 REST only -- its PADS transfer
reverses the ET ranking, `rest_2015.md`) is trained on the REST recordings of
the same training patients, tuned on the same validation patients, and scores the
same test patients. Patients with no REST recording keep the OUT prediction.

    ft         OUT, PADS pretrain -> fine-tune, 3 seeds          (= transfer_2015)
    ft_x2      ft with 6 seeds -- the size-matched control: fusion adds six
               networks (2 members x 3 seeds), so does this
    rest       REST scratch alone
    fuse       mean of ft and rest log-probabilities, offsets tuned on validation
    fuse_w25   the same with REST weighted 0.25 (fixed in advance; no weight
               search on ~30 validation patients)

## Prediction, recorded before the run

**fuse beats ft_x2 on precPD (+0.02 to +0.05) and precN, with precET flat or
lower** -- REST carries tremor presence at rest (fixing N->PD controls) but no
2015 ET signal (`rest_2015.md`: ET precision at chance), so averaging can only
dilute the ET ranking. **fuse_w25 keeps more of the ET precision than fuse.**

Run (CPU, parts in parallel): ``REPS=0-14 python -m experiments.fusion_2015`` ...
then ``python -m experiments.fusion_2015 report``.
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

ARMS = ("ft", "ft_x2", "rest", "fuse", "fuse_w25")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET")
OUT_DIR = os.environ.get("OUT_DIR", "fusion_2015_runs")
TOPK = (3, 5, 10, 15)
_TASK = re.compile(r"_(OUT|REST|WING)$")


def tables():
    from common.quaternion_data import load_quaternion_recordings
    from experiments.rest_2015 import block
    from frequency.tables import spectrum_table
    A, _, C = build()
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = np.array([_TASK.sub("", p) for p in spectrum_table(rA, ch=slice(3, 6))[2]])
    Rp, Rs, Ry, Rpats, Rnd = block(load_quaternion_recordings("Data", action="REST"),
                                   slice(3, 6))
    ri = {_TASK.sub("", p): k for k, p in enumerate(Rpats)}
    rmap = np.array([ri.get(p, -1) for p in ids])          # OUT row -> REST row
    ok = rmap >= 0
    assert np.array_equal(Ry[rmap[ok]], A[3][ok]), "REST / OUT labels disagree"
    return A, C, (Rp, Rs, Rnd), rmap


def fuse(p_out, p_rest, has, w):
    """Weighted mean of log-probabilities; OUT alone where REST is missing."""
    lo, lr = np.log(p_out + 1e-12), np.log(p_rest + 1e-12)
    z = np.where(has[:, None], (1 - w) * lo + w * lr, lo)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def run(reps):
    torch.set_num_threads(1)
    A, C, (Rp, Rs, Rnd), rmap = tables()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec, "nd": desc.shape[1]}
    Cp = np.hstack([C[0], C[1], C[2]])
    has = rmap >= 0
    print(f"device={DEVICE}  2015 OUT n={len(y)}  with REST {int(has.sum())}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
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
            pv1, pt1 = fit("ft", own, pads, y, tr, va, te, C[3][kp], seeds=(0, 1, 2))
            pv2, pt2 = fit("ft", own, pads, y, tr, va, te, C[3][kp], seeds=(3, 4, 5))
            # REST model on the same patients' REST rows
            rtr, rva, rte = (rmap[i][rmap[i] >= 0] for i in (tr, va, te))
            pv_r = np.tile(pv1.mean(0), (len(va), 1))     # placeholders (no REST)
            pt_r = np.tile(pt1.mean(0), (len(te), 1))
            from experiments.rest_2015 import _scratch
            yr = np.full(len(Rp), -1)
            yr[rmap[has]] = y[has]
            rv, rt = _scratch({"packed": Rp, "spec": Rs, "nd": Rnd}, yr, rtr, rva, rte)
            pv_r[has[va]], pt_r[has[te]] = rv, rt
            outs = {"ft": (pv1, pt1),
                    "ft_x2": ((pv1 + pv2) / 2, (pt1 + pt2) / 2),
                    "rest": (pv_r, pt_r),
                    "fuse": (fuse(pv1, pv_r, has[va], 0.5), fuse(pt1, pt_r, has[te], 0.5)),
                    "fuse_w25": (fuse(pv1, pv_r, has[va], 0.25),
                                 fuse(pt1, pt_r, has[te], 0.25))}
            for k, (pv, pt) in outs.items():
                prob[k][te] = pt
                pred[k][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        np.savez(f"{OUT_DIR}/rep{rep:02d}.npz", y=y, has=has, **pred,
                 **{f"p_{k}": v for k, v in prob.items()})
        P = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} PD {P[k][1]:.2f} ET {P[k][2]:.2f}"
                                           for k in ARMS), flush=True)


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
    print(f"2015, OUT ft + REST scratch late fusion, {n} repeats\n")
    print(f"{'arm':>10}" + "".join(f"{c:>9}" for c in NAMES))
    for k in ARMS:
        print(f"{k:>10}" + "".join(f"{v:>9.3f}" for v in R[k].mean(0)))
    for a, b, tag in (("fuse", "ft_x2", "attribution: REST vs more OUT seeds"),
                      ("fuse", "ft", "adoption"),
                      ("fuse_w25", "ft_x2", "attribution, REST weight 0.25"),
                      ("fuse_w25", "fuse", "weight"),
                      ("ft_x2", "ft", "seeds")):
        dd = R[a] - R[b]
        print(f"\n{a} - {b}  ({tag}, paired, {n} repeats)")
        for i, c in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}")
    # ranking use (transfer_2015.md's framing): ET precision among the k
    # patients a repeat ranks most ET-like, per repeat, by P(ET) from the
    # out-of-fold probabilities over all 151 patients
    print("\nET precision among the k most ET-like patients (mean over repeats; "
          "15 ET of 151)")
    print(f"{'arm':>10}" + "".join(f"{'k=' + str(k):>8}" for k in TOPK))
    top = {}
    for k_ in ARMS:
        rows = []
        for f in files:
            d = np.load(f)
            o = np.argsort(-d[f"p_{k_}"][:, 2])
            rows.append([(d["y"][o[:k]] == 2).mean() for k in TOPK])
        top[k_] = np.array(rows)
        print(f"{k_:>10}" + "".join(f"{v:>8.3f}" for v in top[k_].mean(0)))
    for a, b in (("fuse", "ft"), ("fuse", "ft_x2"), ("fuse_w25", "ft")):
        dd = top[a] - top[b]
        cells = []
        for i in range(len(TOPK)):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{dd[:, i].mean():+.3f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{a + ' - ' + b:>16}  " + "  ".join(cells))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
