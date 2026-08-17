"""Do the rotation-invariant axis features improve the in-house model?

`four_families.md` found that on 2015 + NewData the axis family is the best
single family for PD-vs-ET (AUC 0.641), beating harmonics (0.402), amplitude
modulation (0.504) and amplitude (0.516) -- and that PD is more *linear* while ET
is more spread across axes, matching the clinical pronation-supination versus
multi-axis distinction.

`own_data_reality_check.md` established the target: on in-house patients the
current model reaches **ET precision 0.193**, not the 0.685 of the merged
cohort, and adding PADS does not help.

This puts the two together. Test sets are 2015 + NewData only with exactly
**10 ET** each, at natural prevalence, 20 draws -- identical to
`own_data_10et.py` so the numbers are directly comparable.

Four arms, paired on the same test sets:

``base``          spectrum + descriptors + asymmetry + trajectory (0.193 ET)
``+axes``         the 4 axis features appended
``axes replace``  axis features INSTEAD of the 10 descriptors -- eight feature
                  unions in this project have underperformed their best member,
                  so replacing is tested alongside appending
``axes only``     spectrum + axis features, nothing else

Run: ``python -m experiments.inhouse_axes``
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
from signal_processing.tremor_physics import FAMILIES, FEATURE_NAMES
from signal_processing.tremor_physics import patient_table as physics_table

REPEATS, TL, ET_TEST = 20, 64, 10
AXIS_IDX = [FEATURE_NAMES.index(n) for n in FAMILIES["axes"]]


def build():
    from common.load_2025 import SIDE, load_2025_all
    from common.quaternion_data import load_quaternion_recordings

    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    A0 = spectrum_table(rA, ch=slice(3, 6))
    B0 = spectrum_table(rB, ch=slice(3, 6))

    spec = logbin(np.vstack([method_table(rA, "multitaper", slice(3, 6))[0],
                             method_table(rB, "multitaper", slice(3, 6))[0]]))
    traj = np.vstack([trajectory_table(rA, ch=slice(3, 6), n_out=TL)[0],
                      trajectory_table(rB, ch=slice(3, 6), n_out=TL)[0]])
    traj = traj.reshape(len(traj), -1)
    desc = np.vstack([desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6))])
    aB, hB = asym_for(rB, side_new, slice(3, 6), B0[2])
    nA = len(A0[1])
    asym = np.vstack([np.zeros((nA, N_ASYM)), aB])
    have = np.concatenate([np.zeros(nA), hB])[:, None]
    axes = np.vstack([physics_table(rA, ch=slice(3, 6))[0][:, AXIS_IDX],
                      physics_table(rB, ch=slice(3, 6))[0][:, AXIS_IDX]])
    y = np.concatenate([A0[1], B0[1]])
    return spec, desc, asym, have, axes, traj, y


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
    spec, desc, asym, have, axes, traj, y = build()
    n = len(y)
    print(f"in-house: n={n}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   axis features: "
          f"{', '.join(FAMILIES['axes'])}")

    arms = {
        "base (desc+asym)":      np.hstack([desc, asym, have]),
        "+ axes":                np.hstack([desc, asym, have, axes]),
        "axes REPLACE desc":     np.hstack([axes, asym, have]),
        "axes only (no asym)":   axes,
    }
    frac = ET_TEST / int((y == 2).sum())
    n_te = {0: int(round(frac * (y == 0).sum())),
            1: int(round(frac * (y == 1).sum())), 2: ET_TEST}
    print(f"test per repeat: N={n_te[0]} PD={n_te[1]} ET={n_te[2]} "
          f"(prevalence {n_te[2]/sum(n_te.values()):.3f}), "
          f"train+val {n - sum(n_te.values())}\n")

    res = {k: [] for k in arms}
    for rep in range(REPEATS):
        rng = np.random.default_rng(rep)
        te, rest = [], []
        for c in (0, 1, 2):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            te.extend(idx[:n_te[c]]); rest.extend(idx[n_te[c]:])
        te = np.array(sorted(te)); rest = np.array(sorted(rest))
        rng.shuffle(rest)
        n_va = max(int(0.25 * len(rest)), 12)
        va, tr = np.sort(rest[:n_va]), np.sort(rest[n_va:])
        for k, d in arms.items():
            res[k].append(fit_eval(spec, d, traj, y, tr, va, te))

    print(f"{'features':>24}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    out = {}
    for k in arms:
        a = np.array(res[k]); out[k] = a
        m, s = a.mean(0), a.std(0)
        print(f"{k:>24}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)))

    base = out["base (desc+asym)"]
    print(f"\npaired vs base, same {REPEATS} test sets:")
    for k in list(arms)[1:]:
        d = out[k] - base
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
