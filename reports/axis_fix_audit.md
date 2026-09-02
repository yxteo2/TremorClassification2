# A 1 % frequency-axis stretch in the multitaper spectrum — found, fixed, null

## The defect

`m_multitaper` and `m_sst` rebuilt their frequency axis as
`linspace(0, F_MAX, n_freq)` after `apply_*` had cropped the true `rfftfreq`
bins to `f <= f_max`. **The last kept bin is the largest multiple of `fs/nfft`
below `f_max`, not `f_max` itself**, so the reconstruction stretched the whole
axis:

    true bins       3.1250 .. 14.8438 Hz, step 0.390625
    old (linspace)  3.1579 .. 15.0000 Hz, step 0.394737
    max error       0.1562 Hz

That is **1.05 % of the band**, and **14 % of the N-vs-ET mean-frequency gap**
(8.16 vs 7.04 Hz). A component truly at 12.11 Hz was reported at 12.24 Hz.

`m_welch` and `m_stft` take their axes from SciPy and were always correct, and
the descriptor table is built from `stft512`. So the bug put **the reported
model's spectral input on a different frequency scale from every other branch of
its own pipeline** — including the descriptors it is concatenated with inside
`TwoStreamNet`, where a `DESC` "peak at 6.25 Hz" and a `SPEC` bin nominally at
6.25 Hz referred to physically different frequencies.

## Why 68 reports did not catch it

The stretch is a smooth monotone reparametrisation applied **identically to every
recording, patient and cohort** at fs = 100. It biases no class and no site.
Patient-level splits, paired bootstraps and permutation nulls are all insensitive
to a global change of variable that both arms of every comparison share — the
protocol was working exactly as designed and could not have seen this.

It is worth stating plainly what does catch a bug of this shape: **checking a
reconstructed axis against the one the estimator actually used.** The fix carries
an `assert len(f) == n_freq` so the two can never silently diverge again.

## The effect on performance: null, as predicted

Both arms built in one process from the same recordings, differing only in the
frequency axis handed to the interpolation. 20 splits, paired.

| arm | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| fixed (true axis) | 0.650 | 0.652 | 0.654 | 0.652 | **0.602** | 0.068 |
| old (stretched) | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 | 0.068 |

paired, fixed vs old:

    precN   +0.010 [-0.006, +0.028]
    precPD  -0.003 [-0.020, +0.014]
    precET  -0.031 [-0.087, +0.015]
    macroP  -0.008 [-0.029, +0.010]
    macroF1 +0.009 [-0.006, +0.026]

**Nothing significant.** The prediction recorded before the run — "little or no
change in macro precision, because the distortion is uniform and the model was
fitted and evaluated on it consistently" — holds. That is a measurement-derived
prediction, and `failed_predictions.md` records that this kind has by far the
better track record here.

The distortion was not cosmetic at the input: it moved the log-binned spectrum by
up to **3.54 log units**. The model simply relearned around it.

## The consequence that does matter

**Every reported multitaper number was computed on the buggy axis.** The point
estimates move by more than nothing even though the paired difference does not
clear zero:

    macroP  0.660 -> 0.652
    precET  0.685 -> 0.654

That is the difference between quoting 0.685 and quoting 0.654 for ET precision.
The paired interval says the two are statistically indistinguishable, so **this
is not evidence that the fix hurt** — but the headline must be re-derived on the
corrected pipeline before anything is published from it. `headline_audit.py` has
been re-run at 40 splits against the fixed transform for exactly that reason.

**Keep the fix.** A known-wrong frequency axis cannot be carried into a paper to
preserve a point estimate whose difference is inside the noise, and any physical
frequency quoted from the multitaper path was ~1 % high.

## Standing

* **Fixed and asserted.** `_kept_rfftfreq` reconstructs the axis the estimators
  actually used; `m_multitaper` and `m_sst` both assert its length matches the
  returned spectrum.
* **Performance effect is null** — macroP −0.008 [−0.029, +0.010].
* **All multitaper-path numbers predating this commit must be re-derived**,
  including the headline and anything quoting a physical frequency from that
  path. `DESC`-derived frequencies (`stft512`) and welch numbers are unaffected.
* The wider lesson for this repo: the protocol's safeguards are all *relative* —
  they compare arms. A defect shared by every arm is invisible to all of them,
  and only an absolute check against ground truth finds it.
