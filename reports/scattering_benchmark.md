# Scattering candidate: paired patient-level experiment

## Result: do not replace the reference

Completed the prespecified five-fold run on 383 PADS patients. All metrics below
are pooled out-of-fold patient metrics, not averages of split scores.

| Arm | Accuracy | Macro-F1 | ET precision | ET recall |
|---|---:|---:|---:|---:|
| Current architecture, retrained reference | 0.679 | 0.538 | 0.325 | 0.464 |
| First-order scattering + logistic regression | 0.650 | 0.529 | 0.267 | 0.429 |
| First+second-order scattering + logistic regression | 0.619 | 0.465 | 0.175 | 0.250 |
| Fixed 50:50 reference/scattering fusion | 0.642 | 0.511 | 0.306 | 0.393 |

The primary fusion difference in macro-F1 was **-0.027**, with a conditional
paired patient-bootstrap 95% interval **[-0.071, +0.016]**. There is no evidence
of improvement from this fusion in this run. Second-order scattering alone
was -0.073 macro-F1 **[-0.133, -0.017]** versus the reference. Its ET true
positives fell from 13/28 to 7/28; fusion detected 11/28.

These findings reject deployment of this particular candidate on this evidence;
they do not rule out all scattering representations or all action models.
No seed sweep or post-result retuning was performed. Outer-split variability
has not been estimated. The existing baseline implementation is unchanged.

All 12 new synthetic/protocol tests passed. Existing preprocessing verification
passed 39 checks with no failures; HHT, HHT-IMF2+, VMD and wavelet-packet checks
were skipped for missing optional packages. Aggregate machine-readable results
are in `reports/scattering_pads_summary.json`.

## What this tests

Can fixed wavelet-envelope features improve the existing two-stream + spectral
TCN classifier? This is a new feature hypothesis, not a reproduction of the
MATLAB drinking-action model, and not a promise of clinical accuracy.

The motivation is [Anden & Mallat, *Deep Scattering Spectrum*, IEEE Transactions
on Signal Processing 62(16), 4114-4128, 2014](https://arxiv.org/abs/1304.6763).
Second-order scattering describes transient and amplitude-modulation structure.
That audio result motivates testing a sensor adaptation; it does **not** prove
that scattering separates PD and ET. Implementation uses
[Kymatio](https://www.kymat.io/).

## Fixed protocol

- PADS StretchHold only, strict diagnosis mapping, all 383 patients:
  79 normal, 276 PD, 28 ET. ET prevalence is 7.31%.
- Five outer patient folds, shuffle seed 0; every patient has exactly one
  out-of-fold test prediction. Both wrists stay with their patient.
- Within each outer development set, 20% is validation. All four arms use
  identical training, validation, and test identities.
- Reference: the current `final_model.evaluate` architecture and shared trainer:
  multitaper spectrum, ten descriptors, four asymmetry quantities plus availability,
  instantaneous-frequency/envelope trajectory, two-stream network and spectral
  residual TCN, three initialization seeds (0, 1, 2), 200 epochs, validation-best
  weights. This is a **retrained reference**, not a saved clinical model.
- Candidate preprocessing: existing 3-15 Hz bandpass, principal-axis waveform,
  40 Hz, standardized amplitude, central 384 samples. No padded or nonfinite
  recordings may enter this comparison. Average features over all recordings
  per patient, never average raw waveforms across recordings.
- Scattering: fixed J=5, Q=4, order <=2, 3-15 Hz carrier filters; 10 first-order
  features or 34 first+second-order features. Second order is divided by its
  first-order parent. Log compression precedes per-training-fold scaling.
- Logistic regression uses balanced class weights and C in {0.01, 0.1, 1};
  C is selected by three-fold macro-F1 CV **inside training only**, with scaling
  refitted in every inner fold. No augmentation or oversampling.
- Four arms: reference; first-order scattering; first+second-order scattering;
  equal probability average of reference and second-order classifier. The 50:50
  weight is fixed, not chosen after inspecting test outcomes.
- Each arm gets the existing validation-only logit-offset tuning. Exported
  probabilities are **before offsets**, predictions are **after offsets**.
- Primary comparison: fusion minus reference macro-F1. ET precision AND recall
  are reported, along with all other arms, not just the best result.
- Paired bootstrap uses patients and fixed out-of-fold predictions (2,000
  replicates). These intervals condition on the trained models; they do not
  include retraining uncertainty, prior exploratory selection, or site shift.

This differs from the README's capped, merged, repeated-holdout headline.
Numbers from the two protocols cannot be subtracted as a performance gain.
Neither patient grouping nor nested tuning makes this historically explored
cohort a new external test set. See [Cawley & Talbot, JMLR
2010](https://jmlr.org/papers/v11/cawley10a.html).

## Reproduce

Install the core packages in `requirements.txt` and `requirements-scattering.txt`.
Optional EMD, VMD, and ROCKET packages are not needed for this experiment.

```bash
python -m unittest discover -s tests -p test_scattering_benchmark.py -v
python -m experiments.verify_preprocessing
python -m experiments.scattering_benchmark --cohort pads --output artifacts/scattering_pads_seed0
```

Choose a new output path for each run: existing runs are never overwritten.
The output includes `summary.json`, `splits.json`, `selections.json`, and
`predictions.csv`, with package versions and feature fingerprints. Keep
patient-level exports under appropriate research-data access controls.

`--cohort merged` enables the same experiment on the current capped merged
population, provided `Data/` and `NewData/` are available. It also reports each
cohort separately. Do not claim local-patient improvement from PADS alone.

## Scope of the next decision

Do not replace the baseline just because one arm is higher on one metric.
Any promising candidate needs a locked follow-up across prespecified folds and
sites, followed by prospectively collected unseen patients.

The user's [2024 action-tremor paper](https://doi.org/10.1016/j.compbiomed.2024.108957)
studies DRINK and identifies rest-to-lift transitions as important. The current
postural central-window experiment does not test that hypothesis. A separate
action-phase experiment should retain those transitions and compare on matched
DRINK patients; changing model architecture cannot by itself reconcile different
activities, input representations, split units, and evaluation metrics.
