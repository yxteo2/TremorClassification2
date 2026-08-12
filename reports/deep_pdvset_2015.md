# BiLSTM on 2015-only PD-vs-ET — chance, at every setting tried

Asked whether a deep model would fix the low ET precision. Tested directly on
2015 REST, lower_arm, STFT-512, patient-grouped CV, 3 seeds.

## Result

| config | bal-acc | **AUC** | precision | recall | F1 |
|---|---|---|---|---|---|
| baseline h128, no oversample | 0.501 | **0.517** | 0.072 | 0.104 | 0.085 |
| h32, no oversample | 0.463 | **0.475** | 0.150 | 0.375 | 0.213 |
| h32 + oversample | 0.513 | **0.504** | 0.180 | 0.750 | 0.286 |
| h16 + oversample | 0.487 | **0.517** | 0.171 | 0.792 | 0.280 |
| h16 + oversample + SpecAugment | 0.504 | **0.512** | 0.176 | 0.812 | 0.288 |
| h16 + oversample + focal γ=3 | 0.506 | **0.507** | 0.178 | 0.958 | 0.300 |
| **classical logreg, 10 descriptors** | **0.730** | **0.729** | **0.393** | 0.688 | **0.500** |

## Two separate findings

**1. A real defect, now fixed.** The first run had no minority oversampling, and
the network collapsed to the majority class — **2 of 3 seeds predicted zero ET
patients** (precision 0.000). Focal loss alone is not enough at 16 ET against
75 PD. `oversample_to` and `spec_augment` are now plumbed through
`train_bilstm` and `tfbench.deep.train_one`, train-split only. Oversampling
lifts precision 0.072 → 0.18 and recall 0.10 → 0.96.

**2. But the model has learned nothing.** **AUC is 0.475–0.517 in every
configuration.** AUC is threshold-free and prevalence-free, so at 0.5 the model
cannot rank a single ET patient above a PD patient. All the precision/recall
movement above is the operating point sliding along a *chance* ROC curve — the
γ=3 config's 0.958 recall is calling nearly everyone ET, not detecting them.

Logistic regression on 10 descriptors from the same recordings reaches **AUC
0.729**. The signal is present and linearly accessible; the BiLSTM cannot reach
it from raw spectrograms with 16 positive examples.

## Why this was predictable

* A BiLSTM has ~10⁵ parameters against **16 ET patients**.
* Two far smaller capacity increases already failed the same way at this n:
  30 multi-sensor features (paired −0.107, CI excluding zero) and 23 temporal
  features (−0.224). A deep network is that failure mode by orders of magnitude.
* The deep model already lost on the **easier** axis with more data — BiLSTM
  0.866 ± 0.010 vs engineered features 0.884 on N-vs-Tremor, where there are 90
  positives.

## Recommendation

Stop tuning the deep PD-vs-ET model. Capacity, oversampling, augmentation and
focal-γ have all been swept and none moved AUC off chance; the remaining knobs
act on the same bottleneck.

Deep learning is defensible in this project **on the N-vs-Tremor axis** (90
positives, BiLSTM 0.866), reported alongside the engineered-feature result that
still beats it. On PD-vs-ET the honest statement is that deep models sit at
chance, and that this is a sample-size result rather than an architecture one.

## Where the two "more power" routes now stand

| route | ET precision |
|---|---|
| baseline: 2015 only, logreg, 10 descriptors | **0.393** |
| more data — pool 2015 + NewData + PADS (16 → 50 ET) | 0.163 |
| more capacity — BiLSTM, best of 6 configs | 0.180 |

Both closed empirically. The constraint is **16 ET subjects from a single
consistent population**, and neither more subjects from *other* populations nor
more model parameters substitutes for it.
