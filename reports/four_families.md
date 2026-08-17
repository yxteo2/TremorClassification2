# The four clinical signal families, tested properly

An earlier pass dismissed orientation and under-implemented harmonics and
amplitude modulation. Re-done with the confounds removed, **the orientation
family is the best single family on the in-house cohort** -- and its direction
matches the clinical description.

## What was wrong before

| family | earlier treatment | why it was inadequate |
|---|---|---|
| harmonics | only 2f | the clinical claim names 2nd *and* 3rd harmonics |
| orientation | log map, body-frame gravity, per-axis fusion | all **mount-dependent**; wrist orientation is unrecorded, so those features partly measure how the sensor was strapped on |
| amplitude change | one scalar (`amp_cv`) | the claim is *waxing and waning*, which needs the modulation spectrum -- the rate at which amplitude fluctuates |

The fix for orientation is the **eigenvalues of the 3x3 cross-spectral matrix**.
Under a rotation R the matrix becomes `R S R^T`, which has identical
eigenvalues, so the features are invariant to mounting. Verified numerically:
a linear oscillation gives linearity 0.997, and the same signal rotated 40
degrees gives 0.997; isotropic noise gives 0.400.

## PD vs ET, AUC by family

| cohort | n | ET | harmonic | **axes** | ampmod | amplitude | all |
|---|---|---|---|---|---|---|---|
| 2015 OUT | 90 | 15 | 0.460 | **0.648** | 0.561 | 0.554 | 0.628 |
| NewData OUT | 29 | 6 | 0.290 | 0.587 | 0.109 | 0.283 | 0.217 |
| PADS StretchHold | 304 | 28 | **0.736** | 0.558 | 0.703 | 0.673 | 0.733 |
| **in-house (2015+NewData)** | 119 | 21 | 0.402 | **0.641** | 0.504 | 0.516 | 0.594 |

## The orientation result, and its direction

In-house PD-vs-ET, univariate:

| feature | AUC | higher in |
|---|---|---|
| `planarity` | 0.678 | **PD** |
| `linearity` | 0.674 | **PD** |
| `axis_entropy` | 0.661 | **ET** |
| `sphericity` | 0.624 | **ET** |

`linearity` is how far the oscillation is confined to a single axis. **PD is more
linear, ET more spread across axes** -- exactly the clinical distinction between
pronation-supination dominance in PD and multi-axis action tremor in ET. The
mechanism is real and measurable; the earlier failure was the formulation, not
the idea.

## Other findings

* **Harmonics work on PADS (0.736) and are below chance in-house (0.402).** The
  families split by cohort: harmonics on PADS, axes in-house. A single
  cross-cohort claim for either would be wrong.
* **Dilution, eighth instance.** In-house, the combined 14 features (0.594) are
  worse than the 4 axis features alone (0.641).
* **Raw amplitude is weak for PD-vs-ET (0.516) but strong for N-vs-Tremor**
  (0.835 on 2015, 0.794 in-house). "Amplitude indexes severity, not diagnosis"
  holds for the differential, not for detecting tremor at all.
* **Amplitude modulation is weaker than claimed** in-house (0.504). Its best
  single feature is `mod_peak_hz` (0.625) -- the *rate* of fluctuation, not its
  size, which is why a single `amp_cv` scalar missed it.

## Consequence

The in-house PD-vs-ET model should be built on **axis-shape features**, not
frequency. That is the opposite of the merged/PADS pipeline, and it is consistent
with `own_data_reality_check.md`: the two cohorts want different features, and
pooling them serves neither.

Reproduce: `signal_processing/tremor_physics.py`, `scratch/physics_test.py`.

## Do the axis features improve the in-house MODEL?

Directionally yes, not confirmably. Same test sets as
`own_data_reality_check.md` (2015 + NewData, exactly 10 ET each, natural
prevalence 0.101, 20 draws):

| features | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| base (desc + asym) | 0.652 | **0.769** | 0.193 | 0.538 | 0.471 |
| **+ axes** | 0.681 | 0.729 | **0.245** | **0.552** | 0.479 |
| axes replace desc | 0.614 | 0.782 | 0.242 | 0.546 | 0.463 |
| axes only | 0.653 | 0.754 | 0.200 | 0.536 | 0.482 |

Paired against base:

| | precN | precPD | precET | macroP |
|---|---|---|---|---|
| + axes | **+0.029 [+0.003, +0.058]** * | **-0.039 [-0.067, -0.013]** * | +0.052 [-0.049, +0.153] | +0.014 [-0.019, +0.048] |
| axes replace desc | -0.038 [-0.085, +0.006] | +0.013 [-0.025, +0.059] | +0.049 [-0.075, +0.173] | +0.008 [-0.039, +0.053] |

**ET precision 0.193 -> 0.245 is the largest in-house ET improvement measured**
(+27 % relative), but its interval spans zero and its sd widens from 0.186 to
0.273. What *is* significant is a trade rather than a gain: precN +0.029 and
precPD -0.039, both clearing zero -- appending axes shifts predictions toward N
and away from PD.

### Why the feature-level advantage does not convert

The axis family separates PD from ET at AUC 0.641 against harmonics' 0.402 on
these same patients. That advantage does not become a significant model gain,
and the arithmetic explains it: **with 10 ET per test set, an ET-precision
difference must exceed roughly +/-0.10 to clear the noise.** A +0.052 effect is
below what 21 ET patients can resolve.

This is not a statement about the feature. It is a statement about the cohort,
and it is the recurring shape of this project: population-level separability and
model improvement are different things once the minority class is this small.
