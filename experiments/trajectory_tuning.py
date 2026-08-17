"""Tune the trajectory branch -- the newest component, never tuned.

The two-stream model won significantly with a trajectory built by AVERAGING the
instantaneous-frequency trajectories of the three angular-velocity axes. That
average was measured to damp the fluctuation magnitude by **1.61x** (predicted
sqrt(3) = 1.73): each axis is separately mean-centred and their fluctuations are
largely independent, so averaging cancels the quantity the Tremor Stability
Index measures.

Raw class separation in the IF fluctuation, PADS:

    axis_mode      N       PD      ET     PD-ET gap
    mean         0.517   0.499   0.405     0.094      <- and PD < N, wrong order
    dominant     0.905   0.843   0.642     0.201
    pca          0.892   0.846   0.639     0.207
    stack        0.748   0.766   0.641     0.125

So the branch has been running on roughly 40 % of its discriminative signal.
This measures whether that translates into the model, and sweeps the trajectory
length, which was fixed at 64 without justification.

Run: ``python -m experiments.trajectory_tuning``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import NBIN, N_ASYM, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.final_model import method_table
from frequency.tables import spectrum_table
from models.architectures import (DescriptorFusion, ResidualTCN, Spectrum1DCNN,
                                  TRUNKS, TwoStreamNet)
from signal_processing.stability import trajectory_table

SPLITS = 20


def assemble(axis_mode="pca", n_out=64):
    import os

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
    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C0[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    nA = len(A0[1])

    spec = logbin(np.vstack([method_table(rA, "multitaper", slice(3, 6))[0],
                             method_table(rB, "multitaper", slice(3, 6))[0],
                             method_table(rC, "multitaper", slice(0, 3))[0][keep]]))
    kw = dict(axis_mode=axis_mode, n_out=n_out)
    T = np.vstack([trajectory_table(rA, ch=slice(3, 6), **kw)[0],
                   trajectory_table(rB, ch=slice(3, 6), **kw)[0],
                   trajectory_table(rC, ch=slice(0, 3), **kw)[0][keep]])
    n_ch = T.shape[1]
    traj = T.reshape(len(T), -1)
    aB, hB = asym_for(rB, side_new, slice(3, 6), B0[2])
    aC, hC = asym_for(rC, side_pads, slice(0, 3), C0[2])
    desc = np.hstack([
        np.vstack([desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6)),
                   desc_table(rC, slice(0, 3))[keep]]),
        np.vstack([np.zeros((nA, N_ASYM)), aB, aC[keep]]),
        np.concatenate([np.zeros(nA), hB, hC[keep]])[:, None]])
    y = np.concatenate([A0[1], B0[1], C0[1][keep]])
    coh = np.concatenate([np.full(nA, "2015"), np.full(len(B0[1]), "NewData"),
                          np.full(len(keep), "PADS")])
    key = np.array([f"{c}_{l}" for c, l in zip(coh, y)])
    return spec, desc, traj, n_ch, y, key


def evaluate(name, spec, desc, traj, n_ch, n_out, y, key, splits=SPLITS):
    nd = desc.shape[1]
    packed = np.hstack([spec, desc]) if traj is None else \
        np.hstack([spec, desc, traj])
    if traj is None:
        mk1 = lambda: DescriptorFusion(Spectrum1DCNN(NBIN, 3, ch=8),
                                       TRUNKS["cnn"], NBIN, nd, 8 * 2 * 4)
    else:
        mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                   8 * 2 * 4, NBIN, nd, n_out, n_traj_ch=n_ch)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    out = []
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    a = np.array(out); m, s = a.mean(0), a.std(0)
    print(f"{name:>34}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
          + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a


def paired(a, b, name, n=4000):
    d = a - b
    print(f"  {name}:")
    for i, nm in enumerate(("precN", "precPD", "precET", "macroP", "macroF1")):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def main():
    torch.set_num_threads(1)
    print(f"multitaper spectrum, {SPLITS} splits, per-class precision\n")
    print(f"{'config':>34}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    res = {}
    print("### A. how the three axes are combined")
    for mode in ("mean", "dominant", "pca", "stack"):
        spec, desc, traj, n_ch, y, key = assemble(axis_mode=mode, n_out=64)
        res[mode] = evaluate(f"axis={mode}", spec, desc, traj, n_ch, 64, y, key)

    print("\n### B. trajectory length, best axis mode")
    best = max(("dominant", "pca"), key=lambda m: res[m][:, 3].mean())
    for L in (32, 128):
        spec, desc, traj, n_ch, y, key = assemble(axis_mode=best, n_out=L)
        res[f"{best}_{L}"] = evaluate(f"axis={best} len={L}", spec, desc, traj,
                                      n_ch, L, y, key)

    print(f"\npaired vs axis=mean (the previous default), {SPLITS} splits:")
    for k in ("dominant", "pca", "stack", f"{best}_32", f"{best}_128"):
        paired(res[k], res["mean"], k)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
