"""Did fixing the multitaper frequency axis change anything? A paired A/B.

`signal_processing/transforms.py` used to rebuild the multitaper frequency axis
as ``linspace(0, F_MAX, n_freq)`` after `apply_multitaper` had cropped the true
``rfftfreq`` bins to ``f <= F_MAX``. The last kept bin is the largest multiple of
``fs/nfft`` **below** ``f_max``, not ``f_max`` itself, so the reconstruction
stretched the whole axis:

    true bins      3.1250 .. 14.8438 Hz, step 0.390625
    old (linspace) 3.1579 .. 15.0000 Hz, step 0.394737
    max error      0.1562 Hz  =  1.05 % of the band, 14 % of the
                                 N-vs-ET mean-frequency gap (8.16 vs 7.04 Hz)

`m_welch` and `m_stft` always took their axes from SciPy and were correct, and
the descriptor table is built from `stft512` — so the bug put **the reported
model's spectral input on a different frequency scale from every other branch of
its own pipeline**, including the descriptors it is concatenated with inside
`TwoStreamNet`.

## Why it survived 68 reports

The stretch is a smooth monotone reparametrisation applied **identically to every
recording, every patient and every cohort** at fs = 100. It biases no class and
no site, and a network can simply learn on the distorted axis. Nothing in the
protocol — patient-level splits, paired bootstraps, permutation nulls — is
sensitive to a global change of variable that both arms share.

## The prediction, recorded before the run

**Little or no change in macro precision.** The distortion is uniform, and the
model has been fitted and evaluated on it consistently. If the fix *does* move
the headline, the interesting quantity is the direction: a gain would say the
SPEC/DESC frequency disagreement was costing something real, since that
misalignment is the one thing the network could not absorb by relearning.

What the fix definitely buys, regardless of this result, is correctness: any
frequency quoted from the multitaper path was ~1 % high, and the two streams of
the reported model now agree about what a frequency is.

## Design

Both arms are built from the same recordings in the same process, differing only
in the frequency axis handed to the interpolation onto ``FM.GRID``. Everything
else — architecture, seeds, splits, priors, descriptors, trajectory — is
identical, so this is as tight a pairing as the repo's pooling experiments.

20 splits, paired. Run: ``python -m experiments.axis_fix_audit``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import logbin
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pooling_rules import fit_members
from signal_processing.tfd import apply_multitaper
from signal_processing.transforms import (F_MAX, _band, _kept_rfftfreq,
                                          _per_freq_mean)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
FS = 100.0


def mt(x, stretched):
    """Multitaper spectrum on either the true or the old stretched axis."""
    n = min(256, x.shape[-1])
    S = apply_multitaper(x, fs=FS, nperseg=n, nfft=n, noverlap=n * 3 // 4,
                         f_max=F_MAX)
    n_ch = np.atleast_2d(x).shape[0]
    n_freq = np.asarray(S).shape[0] // n_ch
    P = _per_freq_mean(S, n_freq, n_ch, square=True)
    f = (np.linspace(0.0, F_MAX, n_freq) if stretched
         else _kept_rfftfreq(n, FS))
    return _band(f, P)


def spec_for(stretched, recs, keep):
    def table(rs, ch):
        rows = defaultdict(list)
        for r in rs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            f, P = mt(x, stretched)
            f, P = np.asarray(f, float), np.asarray(P, float)
            m = np.isfinite(P)
            v = np.clip(np.interp(FM.GRID, f[m], P[m], left=0.0, right=0.0),
                        0, None)
            rows[r.subject].append(v / (v.sum() + 1e-20))
        p = sorted(rows)
        return np.nan_to_num(np.array([np.mean(rows[k], 0) for k in p]))
    rA, rB, rC = recs
    return logbin(np.vstack([table(rA, slice(3, 6)), table(rB, slice(3, 6)),
                             table(rC, slice(0, 3))[keep]]))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    tf = _kept_rfftfreq(256, FS)
    old = np.linspace(0.0, F_MAX, len(tf))
    print(f"true axis {tf[0]:.4f}..{tf[-1]:.4f} step {tf[1]-tf[0]:.6f}")
    print(f"old  axis {old[0]:.4f}..{old[-1]:.4f} step {old[1]-old[0]:.6f}")
    print(f"max axis error {np.abs(old - tf).max():.4f} Hz\n")
    print("prediction on record: little or no change in macro precision\n",
          flush=True)

    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    recs, keep = load_cohorts()

    SPEC = {}
    for lab, st in (("fixed (true axis)", False), ("old (stretched)", True)):
        print(f"building spectra: {lab} ...", flush=True)
        S = spec_for(st, recs, keep)
        assert len(S) == len(y), f"{lab}: {len(S)} rows, expected {len(y)}"
        SPEC[lab] = S

    # build() now uses the fixed transform, so the fixed arm must reproduce it
    dev = float(np.abs(SPEC["fixed (true axis)"] - d["SPEC"]["multitaper"]).max())
    print(f"\nfixed arm vs build()'s multitaper: max|diff| = {dev:.2e} "
          f"{'OK' if dev < 1e-6 else 'MISMATCH'}")
    assert dev < 1e-6
    diff = float(np.abs(SPEC["fixed (true axis)"]
                        - SPEC["old (stretched)"]).max())
    print(f"how much the bug moved the input: max|diff| over log-bins = "
          f"{diff:.4f}\n", flush=True)

    res = {a: [] for a in SPEC}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in SPEC:
            V, T = fit_members(SPEC[a], D, traj, y, tr, va, te)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                y[te]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in SPEC:
        print(f"{a:>20}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    print("\npaired, fixed vs the old stretched axis:")
    for (dd, lo, hi), c in zip(paired(res["fixed (true axis)"],
                                      res["old (stretched)"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
