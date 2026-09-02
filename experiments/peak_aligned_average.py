"""Align the peaks before averaging a patient's recordings, instead of after.

`method_table` normalises each recording's spectrum and then takes a plain mean
over the patient's recordings. For PADS that means averaging the **left and right
wrist**; for 2015 it means averaging repeat trials.

Those spectra do not peak at the same frequency. Measured on all 383 PADS
patients, the between-wrist peak-frequency mismatch is:

    N   median 0.781 Hz   59 % of patients above 0.5 Hz
    PD  median 0.781 Hz   53 %
    ET  median 0.391 Hz   43 %

against a multitaper bin width of **0.391 Hz**. So a typical patient's two
spectra peak one to two bins apart, and averaging them broadens the result.

## What that costs, measured

Peak sharpness is the strongest class contrast in this project (PADS Q: ET 12.19,
PD 5.80, N 4.08). Averaging the two wrists destroys a third of it:

    | | PD | ET | ET-PD gap |
    | per-wrist Q  | 2.32 | 3.77 | 1.45 |
    | after averaging | 1.86 | 2.82 | 0.96 |   -33 %

Note the direction is the opposite of the obvious guess. PD is the asymmetric
disease, so averaging "should" blur PD more — it does not. **ET loses more in
absolute terms because it starts sharper**: a given misalignment costs a narrow
peak more than a broad one. The mechanism and the measurement agree.

## What this is not

`mil_recordings.md` established that **learned pooling over recordings is
significantly worse than the uniform mean** — attention and max both lose
(precET −0.117). This is not another pooling rule. It is a **registration step
before the same uniform mean**: shift each recording's spectrum so its peak lands
on the patient's own median peak, then average exactly as before. The aggregator
is unchanged.

Aligning to the patient's **median** peak rather than to a fixed reference is
deliberate: it removes between-recording jitter while preserving the patient's
absolute tremor frequency, which is itself discriminative (N 8.16, PD 7.51, ET
7.04 Hz). Aligning everything to a common frequency would throw that away.

## Two comparisons, answering two different questions

**Aligned vs the plain mean decides adoption.** If alignment is worse than doing
nothing, it does not go in, whatever the control says.

**Aligned vs random-shift decides attribution.** The random arm applies shifts of
the *same magnitudes* with random signs and assignment — the same resampling, the
same displacement budget, but not chosen to align anything. It is the honest
counterfactual for "were the shifts worth choosing well?", and a gain over it
attributes the effect to alignment rather than to the interpolation.

An earlier version of this docstring called the random comparison "the one that
decides it". **That was wrong** and is corrected here: beating a deliberately
scrambled arm is a low bar, and the 1-split smoke test made that obvious — random
shift cost precET −0.300 while alignment cost −0.050 against the plain mean, so
"aligned beats random by +0.250" would have read as a triumph while alignment was
in fact losing to doing nothing.

## The prediction, recorded before the run — deliberately cautious

The descriptor-level measurement says alignment recovers 33 % of the ET-PD
sharpness gap. **That is not a prediction that the model improves.**
`failed_predictions.md` #5 is exactly this trap: a prediction built from measured
sub-component gains, which still failed, with the standing conclusion that
*"sub-component gains on this dataset are not evidence about the composite
task."*

So the prediction on record is only this: **aligned should beat random-shift on
precET if the mechanism is real**, whether or not either beats the plain mean.
Adoption is decided separately, by aligned vs plain.

20 splits, paired. Run: ``python -m experiments.peak_aligned_average``
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
from signal_processing.transforms import METHODS

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20


def _shift(v, grid, d):
    """Resample spectrum ``v`` shifted by ``d`` Hz (peak moves by +d)."""
    if abs(d) < 1e-9:
        return v
    return np.clip(np.interp(grid - d, grid, v, left=0.0, right=0.0), 0, None)


def patient_spectra(recs, ch, mode, rng):
    """Per-patient spectrum under one averaging rule."""
    rows, peaks = defaultdict(list), defaultdict(list)
    grid = FM.GRID
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = METHODS["multitaper"](x)
        f, P = np.asarray(f, float), np.asarray(P, float)
        m = np.isfinite(P)
        v = np.clip(np.interp(grid, f[m], P[m], left=0.0, right=0.0), 0, None)
        v = v / (v.sum() + 1e-20)
        rows[r.subject].append(v)
        peaks[r.subject].append(grid[int(np.argmax(v))])
    out = []
    for s in sorted(rows):
        V, pk = rows[s], np.array(peaks[s])
        if mode == "plain" or len(V) < 2:
            out.append(np.mean(V, 0))
            continue
        ref = float(np.median(pk))
        d = ref - pk                       # shift each peak onto the median
        if mode == "random":
            # same shift magnitudes, random signs and assignment: identical
            # interpolation smoothing, no alignment
            d = rng.permutation(np.abs(d)) * rng.choice([-1.0, 1.0], len(d))
        out.append(np.mean([_shift(V[i], grid, d[i]) for i in range(len(V))], 0))
    return np.array(out)


def spec_for(mode, recs, keep, seed=0):
    rng = np.random.default_rng(1234 + seed)
    rA, rB, rC = recs
    return logbin(np.vstack([patient_spectra(rA, slice(3, 6), mode, rng),
                             patient_spectra(rB, slice(3, 6), mode, rng),
                             patient_spectra(rC, slice(0, 3), mode, rng)[keep]]))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    recs, keep = load_cohorts()

    ARMS = ("plain mean (current)", "peak-aligned", "random-shift (control)")
    MODE = {"plain mean (current)": "plain", "peak-aligned": "aligned",
            "random-shift (control)": "random"}
    SPEC = {}
    for a in ARMS:
        print(f"building spectra: {a} ...", flush=True)
        S = spec_for(MODE[a], recs, keep)
        assert len(S) == len(y), f"{a}: {len(S)} rows, expected {len(y)}"
        SPEC[a] = S

    ref = d["SPEC"]["multitaper"]
    dev = float(np.abs(SPEC["plain mean (current)"] - ref).max())
    print(f"\nplain arm vs build()'s multitaper: max|diff| = {dev:.2e} "
          f"{'OK' if dev < 1e-6 else 'MISMATCH -- comparisons invalid'}")
    assert dev < 1e-6
    print(f"alignment moved the input by max|diff| = "
          f"{float(np.abs(SPEC['peak-aligned'] - ref).max()):.4f} log units\n",
          flush=True)

    res = {a: [] for a in ARMS}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in ARMS:
            V, T = fit_members(SPEC[a], D, traj, y, tr, va, te)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                y[te]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>24}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>24}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["plain mean (current)"]
    print("\npaired vs the plain mean:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nATTRIBUTION -- aligned vs random-shift. Same resampling and the")
    print("same displacement budget, not chosen to align. A gain here says the")
    print("effect is alignment, not interpolation. ADOPTION is decided by the")
    print("aligned-vs-plain rows above, not by this one:")
    for (dd, lo, hi), c in zip(paired(res["peak-aligned"],
                                      res["random-shift (control)"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
