# OUT-only transfer-learning pilot

## Predeclared protocol

One action only throughout: OUT. No REST, WING or PADS. Target: 151 2015
patients (61 N/75 PD/15 ET); source: 56 NewData (27 N/23 PD/6 ET). Patient
identities are cohort-qualified; no clinical cross-year linkage register is
available, so absence of overlapping participants between cohorts is assumed.

Small masked-reconstruction encoder, not a TS2Vec reproduction or a pretrained
foundation model: two 16-channel stride-two temporal convolutions, nine input
channels, source-only 30 epochs x 5 steps. Mask 20% of time positions and
reconstruct them with a temporary decoder. Each step samples one window per
source patient; labels never enter pretraining. Source per-channel RMS scaling
is fixed and reused for all neural arms. The decoder is discarded afterwards.

Signals: inherited quaternion conversion and NewData ten-second epoch selection;
3–15 Hz bandpass, resample 100 to 40 Hz, four-second nonoverlapping windows,
at most eight evenly selected windows per patient, incomplete tails discarded.
No padding, time warping or amplitude augmentation. Averaged temporal embeddings
are pooled into one patient representation. Longer/repeated records do not add
independent patients; the eight-window cap can discard information.

Four arms use the same supervised patients (2015 outer train plus all NewData):

- 18 spectral power/shape features from full OUT recordings, balanced logistic
  C=.1, training-only feature scaling.
- Small temporal encoder trained from scratch, linear three-class head.
- Source-pretrained frozen encoder, balanced logistic C=.1 on its 16-d patient
  embeddings, training-only embedding scaling.
- Source-pretrained encoder with only its last block and new linear head trained.

Five outer patient folds, fixed seed 0. Inside each fold, 20% of target training
patients provide validation only. Scratch and last-block models train 100 epochs
and select minimum validation weighted cross-entropy. Scratch LR=.001; last-block
LR=.0003; Adam weight decay=.001. Every classifier receives balanced class weights.
No test-driven tuning, offset selection, outlier removal or seed search.

Primary: frozen-pretrained minus scratch macro-F1. Last-block adaptation is
secondary. This comparison evaluates practical training recipes, not a perfectly
isolated initialization effect: frozen and scratch heads use different solvers,
and last-block adaptation differs in learning rate and trainable layers. A random
frozen-encoder control would be needed to isolate pretraining for the frozen arm.
Any positive result should be confirmed with that control and repeated seeds.

Source pretraining is reused across folds because it never accesses target
patients. Frozen embedding inference is independently applied to target patients;
it does not fit statistics on them. Spectral/embedding scalers and all supervised
heads use training patients only. Patient-level manifests and fold probabilities
are saved locally but not committed. The source checkpoint contains no patient IDs.

Motivation: representation learning in [TS2Vec, AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20881)
and [Yuan et al., npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01062-3).
Neither establishes transfer gains for this small source cohort or this task.
This pilot does not use those papers' pretrained weights or reproduce their losses.

## Reproduce

Dependencies: existing scientific Python and torch; shared spectral features from
PR #4. Retrieve complete Data/raw_quaternion and NewData before running.

```bash
python -m unittest discover -s tests -p test_out_transfer.py -v
python -m experiments.out_transfer --output artifacts/out_transfer
```

Four tests cover window shape/invalid duration, rejection of another action,
patient pooling and independent embedding computation. Bootstrap intervals use
2000 paired resamples of fixed held-out patient predictions; they exclude
retraining and historical model-selection uncertainty. All data have previously
been explored; this is not prospective validation. The previous .642 macro-F1
used OUT+REST and different patients/training splits and is not this control.
