"""Stack every verified gain into one model and measure it against the baseline.

Three improvements were measured separately, against the same
welch + descriptors + asymmetry baseline, and never combined:

  transform switch   multitaper / wavelet_packet over welch
                     paired +0.042 macroP [+0.004, +0.082]
  IF trajectory      two-stream spectrum + instantaneous-frequency TCN
                     paired +0.026 macroP [-0.008, +0.068], precET +0.081
  stability features replacing the 10 spectral descriptors
                     paired +0.009 macroP, precET +0.060, sd 0.185 -> 0.120

Whether they stack is genuinely open. Five feature unions in this session have
UNDERPERFORMED their best member -- concat+asym (0.554 against 0.709 for asym
alone), rich descriptors (nothing), fusion on ResidualTCN (hurt), stability
appended to descriptors (significantly hurt precN), descriptors+stability on
PADS (0.754 against 0.807). The maximally-stacked row is included precisely so
that pattern can show itself.

All configurations share one fixed PADS subsample and one set of splits, so
every comparison is paired.

Run: ``python -m experiments.final_model``
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import N_ASYM, NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets

from models.architectures import DescriptorFusion, ResidualTCN, Spectrum1DCNN, TRUNKS, TwoStreamNet
from frequency.tables import spectrum_table

from signal_processing.stability import patient_table as stab_table
from signal_processing.stability import trajectory_table
from signal_processing.transforms import METHODS

SPLITS, TL = 20, 64
GRID = np.linspace(3.0, 15.0, 64)


def method_table(recs, meth, ch):
    """Any transform, resampled onto one common 3-15 Hz grid and normalised."""
    fn = METHODS[meth]
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = fn(x)
        f, P = np.asarray(f, float), np.asarray(P, float)
        m = np.isfinite(P)
        v = np.clip(np.interp(GRID, f[m], P[m], left=0.0, right=0.0), 0, None)
        rows[r.subject].append(v / (v.sum() + 1e-20))
        lab[r.subject] = r.y
    p = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[k], 0) for k in p])),
            np.array([lab[k] for k in p]), np.array(p))


def build():
    from common.loaders import load_pads_extracted
    from common.load_2025 import SIDE, load_2025_all
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

    d = dict(
        DESC=np.vstack([desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6)),
                        desc_table(rC, slice(0, 3))[keep]]),
        STAB=np.vstack([stab_table(rA, ch=slice(3, 6))[0],
                        stab_table(rB, ch=slice(3, 6))[0],
                        stab_table(rC, ch=slice(0, 3))[0][keep]]),
        y=np.concatenate([A0[1], B0[1], C0[1][keep]]),
    )
    T = np.vstack([trajectory_table(rA, ch=slice(3, 6), n_out=TL)[0],
                   trajectory_table(rB, ch=slice(3, 6), n_out=TL)[0],
                   trajectory_table(rC, ch=slice(0, 3), n_out=TL)[0][keep]])
    d["TRAJ"] = T.reshape(len(T), -1)
    aB, hB = asym_for(rB, side_new, slice(3, 6), B0[2])
    aC, hC = asym_for(rC, side_pads, slice(0, 3), C0[2])
    d["ASYM"] = np.vstack([np.zeros((nA, N_ASYM)), aB, aC[keep]])
    d["HAVE"] = np.concatenate([np.zeros(nA), hB, hC[keep]])[:, None]
    coh = np.concatenate([np.full(nA, "2015"), np.full(len(B0[1]), "NewData"),
                          np.full(len(keep), "PADS")])
    d["key"] = np.array([f"{c}_{l}" for c, l in zip(coh, d["y"])])
    d["SPEC"] = {m: logbin(np.vstack([method_table(rA, m, slice(3, 6))[0],
                                      method_table(rB, m, slice(3, 6))[0],
                                      method_table(rC, m, slice(0, 3))[0][keep]]))
                 for m in ("welch", "multitaper", "wavelet_packet")}
    return d


def evaluate(name, spec, desc, traj, y, key, splits=SPLITS, verbose=True):
    """``traj=None`` uses DescriptorFusion; otherwise the two-stream model."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc]) if traj is None else \
        np.hstack([spec, desc, traj])
    if traj is None:
        mk1 = lambda: DescriptorFusion(Spectrum1DCNN(NBIN, 3, ch=8),
                                       TRUNKS["cnn"], NBIN, nd, 8 * 2 * 4)
    else:
        mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                   8 * 2 * 4, NBIN, nd, TL)
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
    a = np.array(out)
    if verbose:
        m, s = a.mean(0), a.std(0)
        print(f"{name:>40}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a


def main():
    torch.set_num_threads(1)
    d = build()
    y, key, SPEC = d["y"], d["key"], d["SPEC"]
    print(f"n={len(y)}  N={int((y == 0).sum())} PD={int((y == 1).sum())} "
          f"ET={int((y == 2).sum())}   splits={SPLITS}\n")
    D_desc = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    D_stab = np.hstack([d["STAB"], d["ASYM"], d["HAVE"]])
    D_both = np.hstack([d["DESC"], d["STAB"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]

    print(f"{'config':>40}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    res = {}
    res["base"] = evaluate("welch + desc + asym (baseline)", SPEC["welch"],
                           D_desc, None, y, key)
    res["t"] = evaluate("+ trajectory", SPEC["welch"], D_desc, TR, y, key)
    res["mt"] = evaluate("multitaper + desc + asym", SPEC["multitaper"],
                         D_desc, None, y, key)
    res["mt_t"] = evaluate("multitaper + trajectory", SPEC["multitaper"],
                           D_desc, TR, y, key)
    res["mt_t_s"] = evaluate("multitaper + traj + stability(replace)",
                             SPEC["multitaper"], D_stab, TR, y, key)
    res["mt_t_b"] = evaluate("multitaper + traj + desc + stability",
                             SPEC["multitaper"], D_both, TR, y, key)
    res["wp_t"] = evaluate("wavelet_packet + trajectory",
                           SPEC["wavelet_packet"], D_desc, TR, y, key)

    print(f"\npaired vs welch baseline, same {SPLITS} splits (bootstrap 95 % CI):")
    for k, lbl in (("t", "+ trajectory"), ("mt", "multitaper"),
                   ("mt_t", "multitaper + trajectory"),
                   ("mt_t_s", "multitaper + traj + stability"),
                   ("mt_t_b", "multitaper + traj + desc + stab"),
                   ("wp_t", "wavelet_packet + trajectory")):
        diff = res[k] - res["base"]
        print(f"  {lbl}:")
        for i, nm in enumerate(("precN", "precPD", "precET", "macroP", "macroF1")):
            b = [np.mean(np.random.default_rng(s).choice(diff[:, i], len(diff),
                                                         replace=True))
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {nm:>8} {diff[:, i].mean():+.3f}  "
                  f"[{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
