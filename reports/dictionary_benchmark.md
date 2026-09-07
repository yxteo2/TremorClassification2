# Patient-pooled WEASEL dictionary experiment

## Fixed protocol

Test **WEASEL 2.0's transform with a patient-pooled logistic head**, not its
off-the-shelf ridge classifier. Motivation: [Schaefer & Leser, Machine Learning
2023](https://doi.org/10.1007/s10994-023-06395-w); the related BOSS approach was
also evaluated in the [PADS study](https://www.nature.com/articles/s41531-023-00625-7).
Neither establishes a gain on this repository's N/PD/ET task.

- Strict PADS StretchHold, 383 patients: 79 N, 276 PD, 28 ET.
- Identical five patient folds and inner 20% validation split to the scattering
  experiment, seed 0. One held-out prediction per patient.
- Existing 3-15 Hz, principal-axis, 40 Hz, central 384-sample waveform. Both
  wrists processed separately; pool word counts, never raw waveforms.
- aeon 1.5.0 WEASELTransformerV2: seed 0, feature-count setting 4096, word
  lengths (7,8), raw and first-difference variants, default randomized windows,
  chi-squared selection. All dictionary fitting uses TRAINING records only.
- Mean counts within patient; L1 normalization and square root; balanced
  logistic regression, fixed C=0.1 and max_iter=3000. No parameter/seed sweep.
- Arms: retrained reference, dictionary, fixed 50:50 probability fusion.
  Each gets existing validation-only logit offsets.
- Primary comparison: dictionary minus reference macro-F1. Fusion is secondary.
  Report ET precision and recall together.
- Reference uses the unchanged two-stream/TCN architecture, three fixed neural
  seeds, 200 epochs, training-only scaling and validation-best checkpoints.
- Paired bootstrap of fixed patient OOF predictions (2000 replicates) excludes
  retraining uncertainty, prior exploratory selection, and site shift.

This is exploratory PADS-only evidence; these patients have already been used
in prior experiments. It is not an external clinical evaluation. Actual feature
counts can differ from the budget parameter and are recorded per fold.

## Completed five-fold results

All 383 patients received one outer-test prediction. No abstention was used.

| Model | Accuracy | Balanced accuracy | Macro-F1 | ET precision | ET recall |
|---|---:|---:|---:|---:|---:|
| Reference | 0.679 | 0.558 | 0.538 | 0.325 | 0.464 (13/28) |
| Dictionary | 0.517 | 0.591 | 0.490 | 0.372 | 0.571 (16/28) |
| Fixed 50:50 fusion | 0.692 | 0.584 | 0.562 | 0.359 | 0.500 (14/28) |

The primary dictionary comparison decreased macro-F1 by **0.049**; paired
95% bootstrap interval **[-0.104, +0.012]**. Dictionary alone is not a supported
replacement: its higher ET recall accompanies substantially worse PD recall
(0.442 versus 0.768).

The secondary fixed fusion increased macro-F1 by **0.024**; paired 95% interval
**[-0.010, +0.061]**. Its ET precision difference was +0.034
(interval [-0.014, +0.097]) and ET recall difference +0.036
(interval [0.000, +0.125]). These intervals do not establish a reliable gain.
The fusion is a candidate for a frozen follow-up evaluation, not evidence of
real-world clinical readiness. Its ET precision remains 0.359.

All five split manifests and all 383 reference labels, fold assignments,
decisions, and probabilities exactly reproduce the completed scattering
benchmark. Maximum absolute reference probability difference: 0.0. The feature
hashes also match. Actual dictionary dimensions were 7868, 7872, 7871, 7871,
and 7869 across the five folds; 4096 is the transform parameter, not a strict
total feature cap.

Validation: all 18 scattering/dictionary benchmark unit tests passed, including
patient pooling and isolation from changed test inputs/labels. Aggregate metrics,
confusion matrices, offsets, versions, and hashes are saved in
[dictionary_pads_summary.json](dictionary_pads_summary.json).

The baseline remains unchanged. Further model selection should be contained in
inner folds, followed by evaluation on patients/cohorts untouched by the prior
exploratory search. This study does not resolve the MATLAB/Python discrepancy
or establish performance on the original action-tremor task.

## Reproduce

Install core dependencies, requirements-scattering.txt (shared reference
builder), and requirements-dictionary.txt. The aeon version is pinned because
its internal transformer module is used.

```bash
python -m unittest discover -s tests -p test_dictionary_benchmark.py -v
python -m experiments.dictionary_benchmark --output artifacts/dictionary_pads_seed0
```

Choose a new output directory for every run. Outputs include split identities,
fold checkpoints, predictions, selections, versions, reference-feature hashes,
and aggregate results. Probabilities are before offsets; decisions are after.
Patient-level exports remain uncommitted research artifacts.

An earlier attempt was interrupted by a session reset. Its unfinished outputs
were lost, so it is not counted as a result. The same fixed model settings were
restored for the complete run.

This does not test action-phase FBMSNet, bicoherence, or questionnaire fusion,
and does not change the production baseline.
