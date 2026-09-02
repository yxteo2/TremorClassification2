# Two preprocessing defects found by synthetic verification, fixed, and null

## How they were found

`experiments/verify_preprocessing.py` pushes signals with analytically known
answers through every stage that feeds a model — a known rotation through the
quaternion path, a 128 Hz sequence through the NewData resampler, pure tones
through all twelve estimators and the grid interpolation, exact partitions
through `logbin`, tones and FM tones through the descriptors, stability features
and IF trajectory, a rotated oscillation through the axis features, a known
harmonic through the harmonic ratios, a planted window through epoch selection.
The protocol's safeguards all compare arms, so a defect every arm shares is
invisible to them; this is the absolute check that the axis bug showed was
missing.

**41 checks pass, 3 skip** (optional `EMD-signal` / `vmdpy`), and two stages
that feed the reported model failed. One check also failed for the wrong
reason and is worth recording: the quaternion path read an 18.7 % error, which
turned out to be the *test's* Euler integrator lagging half a sample; an
analytic quaternion sequence gives 2.35 %, exactly the central-difference bound.
The code was right. A synthetic test's own physics has to be checked first.

Also cleared with numbers: the three cohorts are in the same units (in-band RMS
0.024 / 0.019 / 0.025 rad/s; `total_power` predicts cohort at 0.495 against a
0.488 majority) and every per-patient table aligns with an independent
reconstruction of the row order.

## Defect 1 — `describe()`'s Q-factor was not the peak's half-power width

It took the span of **every** bin above half-maximum anywhere in the band. A 6 Hz
tone with a 0.8-amplitude 12 Hz harmonic read **Q 0.94 instead of 15.0**.

On real recordings (stft512) the supra-half-max set is non-contiguous for

    PADS   N 0.85   PD 0.74   ET 0.30
    2015   N 0.82   PD 0.64   ET 0.62

so the old `q_factor` — one of the ten `DESC` inputs to `TwoStreamNet` — was
measuring "has no clear peak" (N) and "has secondary spectral content" (PD) as
much as sharpness, and doing so in a class-ordered way. Under the corrected
contiguous definition:

| | N | PD | ET | ET/PD ratio |
|---|---|---|---|---|
| PADS, old | 5.09 | 8.42 | 16.05 | **1.90** |
| PADS, new | 22.82 | 22.14 | 20.97 | **0.95** |
| 2015, old | 6.40 | 8.61 | 12.36 | 1.44 |
| 2015, new | 19.28 | 18.86 | 22.20 | 1.18 |

**On PADS there is no ET-vs-PD peak-sharpness gap at all.** The old descriptor's
class contrast was mostly definitional. Two consequences: the headline
"peak sharpness" characteristic (ET 12.19 / PD 5.80 / N 4.08) is **unaffected**,
because `characteristics.py` computes peak-over-mean power, a different and sound
quantity; and the "wrist averaging destroys 33 % of the sharpness gap" argument in
`peak_aligned_average.md` was built on the flawed Q and is withdrawn there.

Fixed to walk the contiguous half-power region around the peak, with a
single-bin fallback that never returns zero width. `Q_CONTIGUOUS = False`
reproduces the old path for the audit.

## Defect 2 — the IF trajectory's end points were filter transients

The band-pass / Hilbert chain gets the instantaneous frequency wrong for the
first and last 10–16 samples, and resampling to 64 points maps the raw ends onto
trajectory points 0 and 63 *exactly*:

    stable 6 Hz tone      point 0: 0.36 Hz of "wander"      interior: 0.017 Hz
    6 +/- 0.5 Hz FM tone  point 63: 2.73 Hz                 interior: 0.56 (truth 0.5)

Two of 64 points per channel, in every patient's `TRAJ` input, were noise of
larger magnitude than the signal. Fixed with a 0.25 s guard band — the settling
scale of the 4 Hz-wide 4th-order filter, comfortably above the measured 16
samples — costing 0.5 s of a 10–15 s recording. `guard_s = 0` reproduces the
old path. The guard changes every patient's trajectory (max |diff| 4.75).

## What the fixes do to the model — nothing, as predicted

Both arms reconstructed in one process; the new arms assert bit-exact against
`build()`. 20 splits, paired against the reconstructed pre-fix model.

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| old DESC, old TRAJ (pre-fix) | 0.650 | 0.652 | 0.654 | 0.652 | **0.602** |
| new DESC, old TRAJ (Q fix) | 0.644 | **0.660** | **0.690** | **0.664** | 0.596 |
| old DESC, new TRAJ (guard) | 0.638 | 0.655 | 0.650 | 0.648 | 0.588 |
| new DESC, new TRAJ (current) | 0.642 | 0.649 | 0.648 | 0.646 | 0.590 |

| arm vs pre-fix | precET | macroP |
|---|---|---|
| Q fix alone | +0.036 [−0.012, +0.097] | +0.012 [−0.003, +0.034] |
| guard alone | −0.004 [−0.068, +0.071] | −0.004 [−0.027, +0.021] |
| both (current defaults) | −0.006 [−0.061, +0.061] | −0.006 [−0.026, +0.019] |

Nothing significant. The predictions on record: **guard null — held**; **Q fix
small with uncertain sign — held** (it trends positive, +0.012, against the
"possibly slightly negative" lean, which was explicitly not the claim). Removing
a mislabelled but class-correlated feature cost nothing; the model had the same
information from `peak_share` and `spectral_entropy`.

## The headline, re-derived at 40 splits on the fixed inputs

| model | precN | precPD | precET | macroP |
|---|---|---|---|---|
| welch + desc + asym (baseline) | 0.640 | 0.635 | 0.550 | 0.608 |
| multitaper + desc + asym | 0.641 | 0.650 | 0.627 | 0.639 |
| **multitaper + trajectory (reported)** | 0.648 | 0.654 | **0.654** | **0.652** |

| claim | after the axis fix | after all three fixes |
|---|---|---|
| headline macroP | +0.046 [+0.023, +0.067] * | **+0.044 [+0.020, +0.068]** * |
| headline precET | +0.103 [+0.040, +0.161] * | **+0.104 [+0.041, +0.169]** * |
| transform alone, precET | +0.045 [+0.001, +0.087] * | +0.078 [+0.022, +0.132] * |
| **trajectory, precET** | +0.057 [+0.015, +0.103] * | **+0.026 [−0.009, +0.068]** |
| trajectory, macroP | +0.021 [+0.005, +0.037] * | +0.012 [−0.000, +0.027] |

**The headline survives unchanged. The trajectory stream's contribution does
not.** Credit moved to the transform; the two still sum to the whole. The
baseline row moved as well this time — correctly, since both models consume the
descriptors. **Quote precET 0.654 / macroP 0.652.** sd(precET) across splits is
0.19, so every figure between 0.65 and 0.69 quoted at any point in this project
is the same number under noise.

## The story about why the trajectory lost significance — tested and wrong

The obvious mechanism: the old trajectory's gain lived in its transient end
points, and point 0 sits on the class-ordered PADS arm-raising onset
(`pads_onset_trim.md`). Measured on every recording with the pre-fix trajectory:

    PADS  old |IF dev| at point 0:  N 1.01   PD 0.89   ET 1.01
          Spearman(point-0 magnitude, onset ratio):  +0.032 overall, +0.028 within class

The transient was class-agnostic noise of about 1 Hz for everyone and did not
track the onset at all. Failed prediction #17. What remains is the plain
reading: +0.057 [+0.015, +0.103] was a marginal effect, and removing two of 128
input values — whether noise or not — was enough to move it inside its interval.
A component whose significance depends on two corrupted inputs staying in place
should not have been called verified, and the README no longer does.

## Standing

* **Keep both fixes.** Correct on their own terms, null on performance, and now
  covered by a regression test that exits non-zero on any failure.
* **The headline is intact** at +0.044 / +0.104; **the trajectory is plausible,
  not verified**. Say so in any writeup.
* **`q_factor` now means what its name says**, and on PADS that quantity does
  not separate ET from PD. The class contrast people attribute to "ET is a purer
  tone" is carried by `peak_sharp` (peak over mean, sound) and by
  `spectral_entropy` / `peak_share`, not by half-power Q.
* **Run `python -m experiments.verify_preprocessing` after touching
  `signal_processing/`, `frequency/` or `common/cohorts.py`**, and add a check
  when a stage is added. Three defects in inputs to the reported model survived
  every relative safeguard this project has; only the absolute check found them.
