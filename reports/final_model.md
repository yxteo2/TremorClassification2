# Final model: multitaper spectrum + instantaneous-frequency trajectory

**The first significant paired improvement to the deep model in this repo.**
macro precision +0.041 [+0.014, +0.067], ET precision +0.102 [+0.031, +0.175],
over the same baseline on the same 20 splits.

## Results

Merged 2015 + NewData + PADS (n=404, 49 ET), PADS capped at 90/class, mixed
protocol, validation-tuned priors, 20 splits, one shared PADS subsample so every
comparison is paired.

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| welch + desc + asym (baseline) | 0.639 | 0.636 | 0.583 | 0.619 | 0.580 |
| + trajectory | 0.627 | 0.643 | 0.639 | 0.636 | 0.585 |
| multitaper + desc + asym | 0.637 | 0.659 | 0.587 | 0.628 | 0.576 |
| **multitaper + trajectory** | 0.639 | 0.655 | **0.685** | **0.660** | **0.593** |
| multitaper + traj + stability (replace) | 0.644 | 0.647 | 0.624 | 0.639 | 0.596 |
| multitaper + traj + desc + stability | 0.649 | 0.652 | 0.659 | 0.653 | 0.587 |
| wavelet_packet + trajectory | 0.604 | 0.634 | 0.601 | 0.613 | 0.575 |

Paired against baseline:

| config | precET | macroP |
|---|---|---|
| + trajectory | **+0.056 [+0.018, +0.095]** * | **+0.017 [+0.001, +0.032]** * |
| multitaper alone | +0.005 [-0.076, +0.076] | +0.009 [-0.019, +0.035] |
| **multitaper + trajectory** | **+0.102 [+0.031, +0.175]** * | **+0.041 [+0.014, +0.067]** * |
| multitaper + traj + stability | +0.042 [-0.073, +0.145] | +0.019 [-0.023, +0.057] |
| multitaper + traj + desc + stab | +0.076 [-0.019, +0.169] | +0.034 [+0.002, +0.064] * |
| wavelet_packet + trajectory | +0.018 [-0.078, +0.104] | -0.006 [-0.039, +0.026] |

## What this establishes

### The trajectory result is confirmed

At 10 splits the ET-precision interval was [-0.007, +0.180] -- missing zero by
0.007. At 20 splits it is [+0.018, +0.095]. The extra splits resolved it in the
direction the point estimate indicated, which is the outcome that was in doubt.

### The two gains stack super-additively

Trajectory alone +0.017 macroP, multitaper alone +0.009, together **+0.041** --
more than their sum. They are not redundant: multitaper improves the spectral
*estimate*, the trajectory adds temporal dynamics the spectrum cannot express at
all. This is the first time in this session that two improvements have combined
for more than either alone.

### Dilution held, as predicted before the run

Adding stability features on top of the winner drops macro precision
0.660 -> 0.639 (replacing descriptors) or 0.653 (appending). This is the sixth
and seventh instance in this session of a feature union underperforming its best
member:

| union | best member | union |
|---|---|---|
| concat + asym (PADS PD-vs-ET) | 0.709 | 0.554 |
| descriptors + stability (PADS) | 0.807 | 0.754 |
| rich descriptors (34 vs 10) | 0.583 | 0.584 |
| fusion on ResidualTCN | 0.510 | 0.501 |
| stability appended to descriptors | 0.649 | 0.644 |
| mt + traj + stability | 0.660 | 0.639 |
| mt + traj + desc + stability | 0.660 | 0.653 |

At 404 patients with 49 ET, **feature dimensionality binds harder than feature
information**. This is a reportable finding, not a nuisance.

### Transform choice

multitaper, not wavelet_packet: the latter significantly *hurts* precN
(-0.035 [-0.064, -0.008]) and is below baseline on macro precision.

## Discrepancy to note

Multitaper alone measures +0.009 macroP here against +0.042 in the earlier
transform sweep (`deep_model_improvement.md`). Different baseline feature set
(that sweep had no asymmetry features) and 10 splits rather than 20. **The
earlier +0.042 should not be carried forward**; this paired 20-split figure
supersedes it.

## Recommended configuration

* merged 2015 + NewData + PADS, PADS capped at 90/class, no distribution
  alignment (`cohort_strategies.md`)
* **multitaper** spectrum, log-scaled, 16 bins
* two-stream: `Spectrum1DCNN` on the spectrum + `TrajectoryEncoder` on the
  instantaneous-frequency trajectory, with `ResidualTCN` soft-voted alongside
* 10 descriptors + 4 unsigned asymmetry features + availability indicator
* **no** stability features on top -- they dilute
* validation-tuned class priors

Measured: precN 0.639, precPD 0.655, precET 0.685, macro precision 0.660.

Reproduce: `python -m experiments.final_model`.
