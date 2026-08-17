"""2015 + NewData with at least 10 ET patients in every test set.

The two in-house cohorts hold **21 ET between them** (2015 15, NewData 6). Fixing
10 in the test set leaves 11 for training and validation -- a more trustworthy
test estimate bought with a much weaker model. Both halves of that trade are
reported.

Three arms, all scored on the **same** test patients so the comparison is
paired:

``own``        train on 2015 + NewData only (11 ET available)
``own+pads``   same test patients, PADS added to TRAINING only (11 + 28 ET)
``own+pads_cap`` PADS capped at 90/class, as in the merged pipeline

The middle arm is the one that matters. `merge_design.md` found that dropping
PADS collapses ET precision from 0.519 to 0.065, but that was measured on a
merged test set. This asks the question on in-house patients only, with enough
ET in test to read the answer.

Test prevalence is held at the cohorts' natural ratio (10 ET among ~99
patients, ET prevalence 0.10) and printed with every result, because precision
is not comparable across differently-composed test sets.

Run: ``python -m experiments.own_data_10et``
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import N_ASYM, NBIN, train, tune_offsets
from experiments.final_model import method_table
from frequency.tables import spectrum_table
from models.architectures import ResidualTCN, Spectrum1DCNN, TRUNKS, TwoStreamNet
from signal_processing.stability import trajectory_table

REPEATS, TL, ET_TEST = 20, 64, 10


def build():
    from common.load_2025 import SIDE, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    side_pads = lambda r: ("left" if "LeftWrist" in str(r.path)
                           else ("right" if "RightWrist" in str(r.path) else None))
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    A0, B0, C0 = (spectrum_table(rA, ch=slice(3, 6)),
                  spectrum_table(rB, ch=slice(3, 6)),
                  spectrum_table(rC, ch=slice(0, 3)))

    def block(recs, sp_tbl, ch, side_fn, has_bilat):
        spec = logbin(method_table(recs, "multitaper", ch)[0])
        traj = trajectory_table(recs, ch=ch, n_out=TL)[0]
        traj = traj.reshape(len(traj), -1)
        d = desc_table(recs, ch)
        if has_bilat:
            a, h = asym_for(recs, side_fn, ch, sp_tbl[2])
        else:
            a = np.zeros((len(sp_tbl[1]), N_ASYM))
            h = np.zeros(len(sp_tbl[1]))
        return spec, np.hstack([d, a, h[:, None]]), traj, sp_tbl[1]

    A = block(rA, A0, slice(3, 6), None, False)
    B = block(rB, B0, slice(3, 6), side_new, True)
    C = block(rC, C0, slice(0, 3), side_pads, True)
    return A, B, C


def fit_eval(spec, desc, traj, y, tr, va, te, seeds=(0, 1, 2)):
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s) for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
    pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    A, B, C = build()
    own_spec = np.vstack([A[0], B[0]])
    own_desc = np.vstack([A[1], B[1]])
    own_traj = np.vstack([A[2], B[2]])
    own_y = np.concatenate([A[3], B[3]])
    n_own = len(own_y)
    print(f"in-house cohorts: n={n_own}  N={int((own_y==0).sum())} "
          f"PD={int((own_y==1).sum())} ET={int((own_y==2).sum())}")
    print(f"PADS available for training: n={len(C[3])} "
          f"ET={int((C[3]==2).sum())}\n")

    # test set: ET_TEST ET plus N/PD at the cohorts' natural ratio
    frac = ET_TEST / int((own_y == 2).sum())
    n_te = {0: int(round(frac * (own_y == 0).sum())),
            1: int(round(frac * (own_y == 1).sum())), 2: ET_TEST}
    print(f"test set per repeat: N={n_te[0]} PD={n_te[1]} ET={n_te[2]} "
          f"(total {sum(n_te.values())}, ET prevalence "
          f"{n_te[2]/sum(n_te.values()):.3f})")
    print(f"left for train+val: {n_own - sum(n_te.values())} patients, "
          f"{int((own_y==2).sum()) - ET_TEST} ET\n")

    res = {k: [] for k in ("own", "own+pads", "own+pads_cap")}
    for rep in range(REPEATS):
        rng = np.random.default_rng(rep)
        te, rest = [], []
        for c in (0, 1, 2):
            idx = np.flatnonzero(own_y == c)
            rng.shuffle(idx)
            te.extend(idx[:n_te[c]]); rest.extend(idx[n_te[c]:])
        te = np.array(sorted(te)); rest = np.array(sorted(rest))
        rng.shuffle(rest)
        n_va = max(int(0.25 * len(rest)), 12)
        va, tr = np.sort(rest[:n_va]), np.sort(rest[n_va:])

        res["own"].append(fit_eval(own_spec, own_desc, own_traj, own_y,
                                   tr, va, te))

        for tag, cap in (("own+pads", None), ("own+pads_cap", 90)):
            if cap is None:
                kp = np.arange(len(C[3]))
            else:
                kp = []
                for c in (0, 1, 2):
                    i = np.flatnonzero(C[3] == c)
                    kp.extend(rng.choice(i, min(cap, len(i)), replace=False))
                kp = np.array(sorted(kp))
            spec = np.vstack([own_spec, C[0][kp]])
            desc = np.vstack([own_desc, C[1][kp]])
            traj = np.vstack([own_traj, C[2][kp]])
            y = np.concatenate([own_y, C[3][kp]])
            pads_idx = np.arange(n_own, len(y))
            # PADS goes into TRAINING only -- never val, never test
            res[tag].append(fit_eval(spec, desc, traj, y,
                                     np.concatenate([tr, pads_idx]), va, te))

    print(f"{'training data':>26}{'precN':>9}{'precPD':>9}{'precET':>9}"
          f"{'macroP':>9}{'macroF1':>9}  |{'  sd':>7}")
    out = {}
    for k in ("own", "own+pads_cap", "own+pads"):
        a = np.array(res[k]); out[k] = a
        m, s = a.mean(0), a.std(0)
        print(f"{k:>26}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)))

    print(f"\npaired vs own-data-only, same {REPEATS} test sets:")
    for k in ("own+pads_cap", "own+pads"):
        d = out[k] - out["own"]
        print(f"  {k}:")
        for i, nm in enumerate(("precN", "precPD", "precET", "macroP", "macroF1")):
            b = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                         replace=True))
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
