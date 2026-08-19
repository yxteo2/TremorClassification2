# Log-frequency bins help; the principal-eigenvalue spectrum does not

Two untested choices in how the spectrum is built, each with a physical argument
against the current default. Run:
`python -m experiments.spectral_representation`, merged cohort, n=404, 20 splits,
reported model, changing only the spectrum table.

## Result

| representation | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| axis mean, linear bins (current) | 0.626 | 0.639 | 0.658 | 0.641 | 0.582 | 0.065 |
| **axis mean, LOG-freq bins** | **0.657** | 0.647 | **0.675** | **0.660** | **0.600** | **0.052** |
| principal eigenvalue, linear bins | 0.611 | **0.660** | 0.650 | 0.640 | 0.587 | 0.069 |
| principal eigenvalue, LOG-freq bins | 0.604 | 0.651 | 0.659 | 0.638 | 0.592 | 0.060 |
| polarisation spectrum, linear bins | 0.592 | 0.620 | 0.651 | 0.621 | 0.552 | 0.089 |

paired vs the current representation:

| arm | precN | precPD | macroP |
|---|---|---|---|
| axis mean, LOG-freq bins | **+0.031 [+0.006, +0.057]** * | +0.008 | +0.019 [−0.005, +0.043] |
| principal eigenvalue, linear | −0.015 | **+0.021 [+0.004, +0.040]** * | −0.000 |
| principal eigenvalue, LOG | −0.021 | +0.012 | −0.003 |
| polarisation spectrum | −0.033 | −0.019 | −0.020 |

## Log-frequency binning: a real, modest gain

Significant on precN (+0.031 *), positive but not significant on macro precision
(+0.019 [−0.005, +0.043]) and macro F1 (+0.018). It also **reduces variance** —
sd(macroP) 0.065 → 0.052, the lowest of any arm.

The mechanism it was built on: `ResidualTCN` convolves along the frequency axis.
On a **linear** axis a tremor's harmonics sit at f, 2f, 3f — distances that
change with the fundamental, so no fixed kernel matches them across patients
whose tremor frequencies differ. On a **log** axis the same harmonics sit
log 2 apart for everyone, so one kernel serves all patients and the convolution
becomes equivariant to a *scaling* of frequency rather than a shift. Log spacing
also concentrates resolution where the classes sit (PD 4–6 Hz, ET 4–12 Hz)
instead of spreading 16 uniform bins over 3–15 Hz.

The variance reduction is consistent with that story: a representation in which
the same kernel works for every patient should be less sensitive to which
patients land in the training fold.

## The principal eigenvalue: correct physics, discarded by the pipeline

`spectrum_table` and `method_table` both average the three gyroscope axes
(`P.mean(0)`). Tremor is close to a **linear** oscillation — this repo measured
linearity 0.997 — so the signal lives on roughly one spatial axis while the other
two carry noise. The rotation-invariant alternative is the largest eigenvalue of
the per-frequency cross-spectral matrix S(f) = ⟨X(f)X(f)ᴴ⟩, which is the power
along the dominant oscillation direction. Under a rotation S → RSRᵀ has identical
eigenvalues.

**The mechanism was verified synthetically before touching real data**: a 6 Hz
linear oscillation on an arbitrary axis in isotropic noise gives

| | axis mean | λ₁ | ratio |
|---|---|---|---|
| at 6 Hz (signal) | 0.0642 | 0.1902 | **2.96×** |
| at 12 Hz (noise) | 0.0016 | 0.0024 | 1.46× |

**SNR 39.7 → 80.5**, and rotation invariance holds to 6 × 10⁻¹⁶. The physics is
exactly as predicted, and it still buys nothing (macroP −0.000).

**Why: the pipeline is scale-invariant by design, and throws the gain away.**
Every spectrum is sum-normalised per patient before binning, and the models
standardise on top of that. A uniform 3× enhancement of the signal band is a
*scale* change, so normalisation removes almost all of it — the measured mean
relative difference between λ₁ and the axis mean after normalisation is only
**0.075**. What survives is a 7.5 % change in spectral *shape*, which is not
enough to move a model.

This is worth keeping as a general lesson: **an SNR improvement that lives in
absolute amplitude is invisible to a scale-invariant pipeline.** The same
normalisation that makes the three cohorts poolable at all
(`merge_design.md`) is what discards it.

The one significant effect is precPD +0.021 * with linear bins, which is a
genuine but isolated result — the same arm's precN is −0.015 and macroP is
exactly zero.

## The polarisation spectrum is the weakest arm

λ₁/trace per frequency — the degree of linear polarisation as a function of
frequency — is worse than the current representation on every column, none
significantly, with by far the highest variance (sd 0.089).

**Caveat on this arm specifically**: it was sum-normalised like the others, which
for a bounded ratio is the wrong operation. It therefore tests the *profile
shape* of polarisation across frequency, not its absolute level. A correctly
scaled version is untested; given the arm's magnitude and variance, it is not a
priority.

## Standing

* **Log-frequency binning is a cheap, low-risk change** worth carrying: one
  significant per-class gain, a positive macro point estimate, and lower
  variance. It is tested against the reported model in
  `experiments/combined_best.py`.
* **Do not re-try the principal-eigenvalue or polarisation spectrum** under the
  current normalisation. Re-open only if a pipeline variant keeps absolute
  amplitude — and note that `rest_postural_contrast.md` found the same
  normalisation deletes the within-patient amplitude ratio, so this is the second
  quantity it has removed.
