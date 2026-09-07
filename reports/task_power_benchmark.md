# Rest/posture power benchmark

This fixed experiment tests whether task information helps before introducing
another neural encoder. It follows the research proposal motivated by
[Varghese et al., npj Parkinson's Disease 2024](https://www.nature.com/articles/s41531-023-00625-7)
and [Liu et al., IEEE TBME 2023](https://doi.org/10.1109/TBME.2022.3193277).
Neither paper establishes an expected gain on this three-class cohort.

## Why a new comparison

The existing `task_contrast.py` uses `method_table`, which normalizes every
recording spectrum to unit sum. Its total-power log ratio therefore cannot
measure absolute tremor-power differences. This experiment measures band power
from the extracted gyroscope signals before any amplitude normalization.
It does not modify the old experiment or the production model.

## Fixed protocol

- Strict PADS labels, all 383 patients (79 N, 276 PD, 28 ET). Both rest and
  posture are required for every patient; missing tasks abort the comparison.
- Rest = pads_relaxed; posture = pads_stretchhold. Original 100 Hz recordings,
  all three gyro axes. No additional amplitude normalization or central crop.
- Welch PSD: four-second Hann windows, two-second overlap. Sum axis PSDs to
  preserve vector power without introducing rectification harmonics.
- Six features per recording: log integrated power in [3,5), [5,7), [7,10),
  [10,15] Hz, peak frequency, and normalized spectral entropy over 3–15 Hz.
  Bands are fixed engineering choices, not diagnostic thresholds.
- Average recording features within patient and task. The two-task model gets
  six mean levels and six posture-minus-rest differences. This representation
  preserves both task vectors; linear differences add no nonlinear capacity.
- Logistic regression C=0.1, balanced class weights, training-only scaling.
  No model, band, C, or random-seed search.
- Identical five patient folds and inner validation patients to the scattering
  and dictionary benchmarks. Validation-only decision offsets for every arm.
- Four arms: unchanged reference, posture logistic, two-task logistic, and
  fixed 50:50 probability fusion of reference and two-task logistic.
- Primary: two-task minus posture logistic macro-F1. Secondary comparisons
  against the reference. The primary isolates adding rest with the same model.
- Paired patient bootstrap: 2000 fixed-OOF replicates. Does not include
  retraining uncertainty, historical model selection, or external site shift.

This is exploratory reuse of PADS and does not establish clinical deployment
performance. Raw power assumes consistent extraction units and calibration
across tasks; cross-device transport requires separate validation. Missing
modalities, questionnaires, DRINK phases and a filter-bank neural encoder are
not tested here. The narrow comparison comes first to separate added data from
added model capacity.

## Reproduce

Dependencies match requirements-scattering.txt and its core dependencies.

```bash
python -m unittest discover -s tests -p test_task_power_benchmark.py -v
python -m experiments.task_power_benchmark --output artifacts/task_power_pads_seed0
```

Four tests check power scaling, rotation invariance, missing-task rejection,
invalid signals and isolation of fitting from test inputs/labels. Patient-level
outputs and checkpoints remain uncommitted.
