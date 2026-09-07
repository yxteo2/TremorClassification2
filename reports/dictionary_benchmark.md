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
