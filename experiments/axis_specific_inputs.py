"""Different inputs for the two decisions, because they want opposite things.

`task_averaging.md` measured a clean split. Averaging every task into the
per-patient spectrum, on the reported model:

    precN   +0.047 [+0.009, +0.088] *      more recordings, same question
    precET  -0.104 [-0.170, -0.035] *      the PD-vs-ET contrast is averaged away
    macroP  -0.012                          the two cancel

So the extra recordings are not useless and they are not useful — they are useful
for **one of the two decisions** and harmful for the other. A single flat 3-class
model must pick one input and pay the other cost. A model that makes the two
decisions separately does not have to.

  N vs Tremor      "is there tremor at all" -- benefits from every recording
  PD vs ET         "which tremor" -- needs the postural condition kept clean

**The control that makes this interpretable.** A two-stage hierarchy was tried
before and lost (macroP 0.568 vs 0.583), so any gain here could be the hierarchy
rather than the inputs. Arm 2 runs the identical hierarchy with **postural inputs
at both stages**, which is that earlier experiment reproduced inside this one.
Only the arm-3-minus-arm-2 difference is attributable to the input choice.

This is the same discipline the SSL retraction was written about: when an arm
changes two things, the baseline has to change with it.

Arms:

  1. flat 3-class, postural            the reported model
  2. two-stage, postural / postural    hierarchy alone, no input change
  3. two-stage, ALL-tasks / postural   hierarchy plus axis-specific inputs

Composition is the ordinary chain rule, P(PD) = P(tremor) * P(PD | tremor), and
validation-tuned priors are applied to the composed 3-class vector exactly as in
the flat model, so the calibration step is identical across arms.

Run: ``python -m experiments.axis_specific_inputs``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import desc_table, logbin
from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import aligned, all_task_recs, paired
from experiments.final_model import NBIN, SPLITS, TL, build, method_table
from frequency.tables import spectrum_table
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SEEDS = (0, 1, 2)


def fit_stage(spec, desc, traj, y2, tr, va, te, nc=2):
    """Two-stream + ResidualTCN soft vote, as in the reported model."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj]) if traj is not None else \
        np.hstack([spec, desc])
    tl = TL if traj is not None else 0
    # TwoStreamNet has its OWN num_classes defaulting to 3; passing nc only to
    # the inner Spectrum1DCNN leaves the head at 3 outputs while the trainer
    # builds 2-class weights. The same mismatch bit DescriptorFusion earlier.
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, nc, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, tl, num_classes=nc)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=nc, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y2[tr], (X[va] - mu) / sd, y2[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s, nc=nc)
             for s in SEEDS]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D_post = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    S_post = d["SPEC"]["multitaper"]

    # all-task tables on the same patient order
    cohorts = all_task_recs()
    A, B_, C = (spectrum_table(cohorts[0][0], ch=cohorts[0][2]),
                spectrum_table(cohorts[1][0], ch=cohorts[1][2]),
                spectrum_table(cohorts[2][0], ch=cohorts[2][2]))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C[1][keep]]), y), \
        "patient order does not match build()"

    parts, pats, dparts = [], [], []
    for (post, alls, ch) in cohorts:
        X, _, p = method_table(alls, "multitaper", ch)
        parts.append(X); pats.append(p)
        dparts.append(desc_table(alls, ch))
    pall = np.concatenate(pats)
    S_all = logbin(aligned((np.vstack(parts), pall), order))
    D_all = np.hstack([aligned((np.vstack(dparts), pall), order),
                       d["ASYM"], d["HAVE"]])

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits\n", flush=True)

    y_tre = (y != 0).astype(int)          # stage A target
    ARMS = ("flat 3-class, postural",
            "two-stage, postural/postural",
            "two-stage, ALL-tasks/postural")
    res = {a: [] for a in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        def score(pv3, pt3):
            pred = (np.log(pt3 + 1e-12) + tune_offsets(pv3, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            return [P[0], P[1], P[2], P.mean(), F.mean()]

        # ---- arm 1: flat 3-class -------------------------------------- #
        pv, pt = fit_stage(S_post, D_post, traj, y, tr, va, te, nc=3)
        res["flat 3-class, postural"].append(score(pv, pt))

        # ---- stage B: PD vs ET, postural, trained on tremor patients --- #
        trT = tr[y[tr] != 0]
        vaT = va[y[va] != 0]
        yb = (y == 2).astype(int)
        pvB, ptB = fit_stage(S_post, D_post, traj, yb, trT, vaT, te, nc=2)

        # ---- stage A: N vs Tremor, two input choices ------------------- #
        for lab, SA, DA in (("two-stage, postural/postural", S_post, D_post),
                            ("two-stage, ALL-tasks/postural", S_all, D_all)):
            pvA, ptA = fit_stage(SA, DA, traj, y_tre, tr, va, te, nc=2)
            # compose on val and test alike; stage B on val needs the tremor rows
            pvB_full = np.zeros((len(va), 2))
            idx = {p: i for i, p in enumerate(vaT)}
            for j, p in enumerate(va):
                pvB_full[j] = pvB[idx[p]] if p in idx else np.array([0.5, 0.5])
            comp = lambda pA, pB: np.column_stack(
                [pA[:, 0], pA[:, 1] * pB[:, 0], pA[:, 1] * pB[:, 1]])
            res[lab].append(score(comp(pvA, pvB_full), comp(ptA, ptB)))
        print(f"  split {sp+1}/{SPLITS} done", flush=True)

    for k in res:
        res[k] = np.array(res[k])

    print(f"\n{'arm':>32}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        m = res[a].mean(0)
        print(f"{a:>32}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[a][:, 3].std():>12.3f}")

    print("\npaired vs the flat 3-class model:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], res[ARMS[0]]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\npaired vs the SAME hierarchy with postural inputs "
          "(isolates the input choice):")
    for (dd, lo, hi), c in zip(paired(res[ARMS[2]], res[ARMS[1]]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
