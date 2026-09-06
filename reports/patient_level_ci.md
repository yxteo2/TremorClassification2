# Patient-level uncertainty: current-pipeline rerun pending

The results below predate the corrected frequency axis, descriptors and trajectory.
They **do not validate** the current ET precision 0.654 / macro precision 0.652
reported in `descriptor_trajectory_fix.md`. Historical significance claims are
withdrawn as support for the current pipeline; updated intervals must be computed.

## Current implementation

`python -m experiments.patient_level_ci --output patient_ci_current.json`

Both arms now call `final_model.evaluate`, using the current preprocessing and
40 shared splits. The same patient draw is reused across arms and splits, with
multiplicity preserved. Ordered test identities must match exactly. Missing-class
draws use the same fixed-label, zero-division metric policy as the original
scores. The script exports patient identities, predictions, metrics, source
hashes and feature fingerprints for auditability.

`python -m experiments.headline_audit --output headline_current.json` performs
this calculation for all three headline comparisons in one training run.

## Historical output only

| arm | precN | precPD | precET | macroP |
|---|---|---|---|---|
| welch + desc + asym (baseline) | 0.639 | 0.636 | 0.583 | 0.619 |
| multitaper + trajectory | 0.639 | 0.655 | **0.685** | **0.660** |

| | diff | split-level 95 % | patient-level 95 % | width ratio |
|---|---|---|---|---|
| precN | +0.001 | [−0.029, +0.029] | [−0.019, +0.020] | 0.7 |
| precPD | +0.019 | [−0.005, +0.044] | [−0.006, +0.045] | 1.1 |
| **precET** | **+0.102** | [+0.031, +0.175] * | **[+0.005, +0.163] *** | 1.1 |
| **macroP** | **+0.041** | [+0.014, +0.067] * | **[+0.005, +0.066] *** | 1.1 |

**Both significant results hold.** macro precision +0.041 [+0.005, +0.066] and ET
precision +0.102 [+0.005, +0.163] with the patient as the sampling unit.

## Interpretation limits

Both procedures condition on already fitted models. Patient resampling captures
evaluation-patient composition, while split resampling describes sensitivity to
split assignment on the fixed dataset. Overlapping splits are not independent
patient evidence. Neither accounts for refitting on new training patients or the
many model/feature choices already explored in this repository. Pairing does not
prove those omitted sources cancel, nor is one interval guaranteed to be wider.

Do not infer that the two methods are equivalent from similar historical widths.
Neither interval establishes performance at an unseen site.

For a future confirmatory analysis, freeze the candidate set and use patient-grouped
nested cross-validation, with preprocessing, early stopping, offset tuning and
model selection confined to outer-training data. Keep each outer test fold out
of every choice. Report cohort-specific results and identify this as a new
analysis on previously explored data; external validation remains stronger.
See [scikit-learn's nested CV example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
