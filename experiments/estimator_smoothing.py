"""Is the headline's transform gain really about SMOOTHING? Sweep it and find out.

The reported model's spectral input is multitaper; the baseline it beats is
welch. That gain — macroP +0.043 [+0.024, +0.062] at 40 splits — is the largest
single measured effect in the project after the validation-tuned priors. **What
nobody has asked is which property of multitaper produced it.**

Measured on a pure 6 Hz tone, whose true bandwidth is zero, the two estimators
differ enormously in how much they blur a peak. Q ceiling is the sharpest value
the estimator can report at all:

    ar16                          Q 31.00     (sharpest; never tried as an input)
    welch, nperseg 512            Q 15.00     (the baseline)
    multitaper nw 2.5 K4 n256     Q  5.33     (the reported model)

So the reported model's input is **three times blunter** than the baseline it
beats, and blunter still than an AR spectrum. Whatever multitaper is doing, it is
not preserving peak shape — and peak sharpness is the single strongest
class characteristic this project has measured (PADS: ET 12.19, PD 5.80, N 4.08).

That inverts the natural hypothesis. **The evidence says smoothing helps**, and
the obvious reading is variance reduction: at 404 patients with 49 ET, a
lower-variance estimate of a blurred peak beats a noisy estimate of a sharp one.

## The prediction, recorded before the run

**Performance should increase monotonically as the estimator gets smoother**,
until over-smoothing eventually destroys the band. Ordering the arms by measured
Q ceiling:

    ar16 (31.0)  <  welch (15.0)  <  nw2.5 (5.33)  <  nw4 (2.14)  <  nw6 (1.36)

If that ordering holds, two things follow: the headline gain is explained, and
**the current nw = 2.5 is not the optimum** — there is a real improvement sitting
one knob turn away. If ar16 wins instead, smoothing is not the axis and the
welch-to-multitaper gain is about something else entirely.

This is deliberately the *measurement-derived* kind of prediction with an
objective x-axis, not a mechanism story. `failed_predictions.md` records that the
former has by far the better record here — four mechanism stories failed this
session, one measurement-derived prediction held.

## What this isolates that the headline could not

Welch versus multitaper changes **two things at once**: the estimator family and
the amount of smoothing. Attributing the +0.043 to smoothing is therefore a
hypothesis, not something the headline established. The `nw` sweep changes
smoothing **with the family fixed** — same DPSS machinery, same window length,
same frame count, only the time-bandwidth product moves. That is the arm that
decides it.

`n_tapers` is raised with `nw` (K < 2·NW is the concentration condition), so the
arms also differ in how many tapers are averaged. That is not a confound to
remove: taper count and bandwidth are the same knob in multitaper, and untying
them would produce badly concentrated tapers.

Arms, all as the deep model's spectral input with everything else identical:

  ar16                   Q 31.00   parametric, sharpest, never used as an input
  welch nperseg 512      Q 15.00   the published baseline
  multitaper nw2.5 K4    Q  5.33   the reported model
  multitaper nw4  K7     Q  2.14   smoother
  multitaper nw6  K11    Q  1.36   smoothest, 2W = 4.69 Hz across a 12 Hz band

20 splits, paired. Run: ``python -m experiments.estimator_smoothing``
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
from experiments.pooling_rules import fit_members
from frequency.descriptors import describe
from signal_processing.tfd import apply_multitaper
from signal_processing.transforms import F_MAX, METHODS, _band, _per_freq_mean

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
FS = 100.0


def mt_variant(nperseg, nw, K):
    """A multitaper estimator with an explicit time-bandwidth product."""
    def fn(x, **kw):
        n = min(nperseg, x.shape[-1])
        S = apply_multitaper(x, fs=FS, nperseg=n, nfft=n, noverlap=n * 3 // 4,
                             f_max=F_MAX, n_tapers=K, nw=nw)
        n_ch = np.atleast_2d(x).shape[0]
        n_freq = np.asarray(S).shape[0] // n_ch
        P = _per_freq_mean(S, n_freq, n_ch, square=True)
        return _band(np.linspace(0.0, F_MAX, n_freq), P)
    return fn


ARMS = {
    "ar16": METHODS["ar16"],
    "welch n512": METHODS["welch"],
    "MT nw2.5 K4 [now]": mt_variant(256, 2.5, 4),
    "MT nw4 K7": mt_variant(256, 4.0, 7),
    "MT nw6 K11": mt_variant(256, 6.0, 11),
}


def q_ceiling(fn):
    """Sharpest Q the estimator can report, measured on a pure tone."""
    t = np.arange(int(15.5 * FS)) / FS
    f, P = fn(np.atleast_2d(np.sin(2 * np.pi * 6 * t)))
    return describe(f, P)["q_factor"]


def spec_for(fn, recs_by_cohort, keep):
    """Per-patient spectrum on the shared grid, exactly as final_model does."""
    def table(recs, ch):
        rows, lab = defaultdict(list), {}
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            f, P = fn(x)
            f, P = np.asarray(f, float), np.asarray(P, float)
            m = np.isfinite(P)
            v = np.clip(np.interp(FM.GRID, f[m], P[m], left=0.0, right=0.0),
                        0, None)
            rows[r.subject].append(v / (v.sum() + 1e-20))
            lab[r.subject] = r.y
        p = sorted(rows)
        return np.nan_to_num(np.array([np.mean(rows[k], 0) for k in p]))
    rA, rB, rC = recs_by_cohort
    return logbin(np.vstack([table(rA, slice(3, 6)), table(rB, slice(3, 6)),
                             table(rC, slice(0, 3))[keep]]))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    print("estimator smoothing, measured on a pure 6 Hz tone:")
    qs = {a: q_ceiling(f) for a, f in ARMS.items()}
    for a in ARMS:
        print(f"  {a:>20}  Q ceiling {qs[a]:>6.2f}")
    print("\nprediction on record: performance rises monotonically as Q ceiling")
    print("FALLS, i.e. smoother is better, until over-smoothing bites\n",
          flush=True)

    d = FM.build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    D = np.hstack([d["DESC"], A])
    traj = d["TRAJ"]

    # rebuild the raw recordings once so each estimator can be applied to them
    import os
    from common.loaders import load_pads_extracted
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    rA = load_quaternion_recordings("Data", action="OUT",
                                    mode="angular_velocity")
    rB = load_2025_all()
    rC = load_pads_extracted()

    SPEC = {}
    for a, fn in ARMS.items():
        print(f"building spectra: {a} ...", flush=True)
        try:
            S = spec_for(fn, (rA, rB, rC), slice(None))
            if len(S) != len(y):
                print(f"  patient count {len(S)} != {len(y)}; "
                      f"falling back to build() for this arm")
                S = None
        except Exception as e:                       # noqa: BLE001
            print(f"  failed: {type(e).__name__}: {e}")
            S = None
        if S is None and a in ("welch", "welch n512", "MT nw2.5 K4 [now]"):
            S = d["SPEC"]["welch" if "welch" in a else "multitaper"]
        SPEC[a] = S

    usable = [a for a in ARMS if SPEC[a] is not None
              and len(SPEC[a]) == len(y)]
    print(f"\nusable arms: {usable}\n", flush=True)
    if len(usable) < 2:
        print("too few usable arms; aborting\nMARKER_DONE", flush=True)
        return

    res = {a: [] for a in usable}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in usable:
            V, T = fit_members(SPEC[a], D, traj, y, tr, va, te)
            pv, pt = V.mean(0), T.mean(0)
            res[a].append(score(pt, tune_offsets(pv, y[va]), y[te]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    order = sorted(usable, key=lambda a: -qs[a])      # sharpest first
    print(f"\n{'estimator':>20}{'Qceil':>8}"
          + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in order:
        print(f"{a:>20}{qs[a]:>8.2f}"
              + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    cur = "MT nw2.5 K4 [now]"
    if cur in res:
        print(f"\npaired vs the reported model ({cur}):")
        for a in order:
            if a == cur:
                continue
            print(f"  {a}:")
            for (dd, lo, hi), c in zip(paired(res[a], res[cur]), NM):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- is macroP monotone in smoothing?")
    print(f"{'estimator':>20}{'Qceil':>8}{'macroP':>9}{'precET':>9}")
    for a in order:
        print(f"{a:>20}{qs[a]:>8.2f}{res[a][:,3].mean():>9.3f}"
              f"{res[a][:,2].mean():>9.3f}")
    xs = np.array([qs[a] for a in order])
    ys = np.array([res[a][:, 3].mean() for a in order])
    from scipy.stats import spearmanr
    print(f"\n  Spearman(Q ceiling, macroP) = {spearmanr(xs, ys).correlation:+.3f}"
          f"   (negative = smoother is better, as predicted)")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
