# TCN and BiLSTM on 2015 + NewData + PADS, 3-class N/PD/ET

**Summary: a 1 k-parameter 1-D CNN beats both, and BiLSTM h=32 loses to logistic
regression. The pooled gain is carried almost entirely by PADS.**

## Setup

| cohort | n | N | PD | ET | source |
|---|---|---|---|---|---|
| 2015 | 151 | 61 | 75 | 15 | `Data`, OUT, lower_arm |
| NewData | 56 | 27 | 23 | 6 | 2025 Moveo, OUT, lower_arm |
| PADS | 383 | 79 | 276 | 28 | StretchHold, wrist |
| **pooled** | **355** | **148** | **158** | **49** | PADS capped at 60/class |

Capping PADS matters: uncapped it is 383 against 151 + 56 and the pooled fit
becomes a PADS fit. 60/class was the previously measured optimum.

Features are the **frequency-axis** normalised power spectrum (Welch-512,
3-15 Hz, 61 bins, axes averaged) -- not a raw spectrogram. A sequence model over
the time axis measured at chance (0.513) here while the same family over
frequency reached 0.913. Class weights ON throughout (ET is ~14 % of the pool).
5-fold `StratifiedGroupKFold`, patient-disjoint, probabilities averaged over
3 seeds.

## Pooled results

| model | params | prec N | prec PD | prec ET | macroP | rec N | rec PD | rec ET | macroF1 |
|---|---|---|---|---|---|---|---|---|---|
| logreg (baseline) | -- | 0.614 | 0.546 | 0.270 | 0.477 | 0.581 | 0.487 | 0.408 | 0.479 |
| MLPHead h=16 | 1 k | 0.653 | 0.570 | 0.302 | 0.508 | 0.649 | 0.595 | 0.265 | 0.505 |
| **Spectrum1DCNN** | **1 k** | **0.690** | **0.662** | **0.391** | **0.581** | 0.736 | 0.557 | 0.510 | **0.587** |
| SpectrumTCN ch=16 | 3 k | 0.614 | 0.575 | 0.314 | 0.501 | 0.784 | 0.291 | 0.551 | 0.492 |
| SpectrumBiLSTM h=32 | 9 k | 0.634 | 0.557 | 0.258 | 0.483 | 0.784 | 0.278 | 0.490 | 0.470 |
| SpectrumBiLSTM h=64 | 35 k | 0.649 | 0.574 | 0.290 | 0.505 | 0.736 | 0.342 | 0.551 | 0.500 |

Two things to note.

**Capacity is again non-monotone, with the optimum at the bottom.** 1 k > 3 k >
9 k, and 35 k recovers only partway. This is the third independent confirmation
of the same pattern in this repo.

**TCN and BiLSTM dissolve the middle class.** Their recalls run N 0.784 /
PD 0.278-0.291 / ET 0.490-0.551 -- PD is absorbed into N and ET. The CNN is the
only model with balanced recalls (0.736 / 0.557 / 0.510). Whatever the recurrent
and dilated-conv-over-frequency models are learning, it separates "no tremor"
from "tremor" and then splits on amplitude, rather than isolating PD.

## Per-cohort breakdown -- the important table

|  | 2015 (15 ET) | | NewData (6 ET) | | PADS (28 ET) | |
|---|---|---|---|---|---|---|
| model | precET | macroF1 | precET | macroF1 | precET | macroF1 |
| logreg | 0.216 | 0.477 | 0.091 | 0.366 | 0.423 | 0.514 |
| MLPHead h=16 | **0.318** | **0.560** | **0.143** | **0.409** | 0.357 | 0.484 |
| Spectrum1DCNN | 0.207 | 0.558 | 0.000 | 0.368 | **0.633** | **0.652** |
| SpectrumTCN ch=16 | 0.146 | 0.435 | 0.000 | 0.356 | 0.583 | 0.570 |
| SpectrumBiLSTM h=32 | 0.159 | 0.450 | 0.071 | 0.355 | 0.457 | 0.528 |
| SpectrumBiLSTM h=64 | 0.167 | 0.458 | 0.000 | 0.342 | 0.571 | 0.593 |

The CNN's pooled win is a PADS win. Its ET precision is 0.633 on PADS, **0.207
on 2015 -- below logreg's 0.216** -- and **0.000 on NewData**. Three of the four
deep models get zero NewData ET patients right.

On the 2015 cohort the best model is still the 1 k `MLPHead` (ET precision
0.318, macro F1 0.560), and the deep sequence models are all *worse* than
logistic regression there.

## Reading

* PADS is 42 % of the pooled set and has the cleanest ET separation. A model
  with enough capacity to specialise will specialise on it, and the pooled
  metric will reward that. Reporting only the pooled row would have been
  misleading, which is why the per-cohort split is the headline table.
* **Merging is still worth it for n**: 49 ET against 15 in 2015 alone. But the
  benefit shows up as a better pooled number, not as better performance on any
  one cohort's patients.
* NewData at 6 ET is not classifiable by anything tried. It should be pooled for
  training and not used as an evaluation cohort on the ET axis.
* The recurrent/dilated models (TCN, BiLSTM) are the wrong family for this
  input. On the *binary* PD-vs-ET DRINK axis the frequency BiLSTM reached 0.913;
  on 3-class pooled it is the worst model tested. The gap is the middle class.

## Verification -- and a reversal

The table above is one fold split. Repeating over 5 splits, and adding a
leave-one-cohort-out test, changes the conclusion.

### Stability over 5 fold-splits (pooled)

| model | precET | macroF1 |
|---|---|---|
| logreg | 0.244 +/- 0.029 | 0.472 +/- 0.014 |
| MLPHead h=16 | 0.308 +/- 0.051 | 0.508 +/- 0.022 |
| **Spectrum1DCNN** | **0.378 +/- 0.017** | **0.574 +/- 0.008** |
| SpectrumTCN ch=16 | 0.313 +/- 0.009 | 0.484 +/- 0.016 |
| SpectrumBiLSTM h=32 | 0.311 +/- 0.032 | 0.496 +/- 0.025 |

**Correction.** The single-split table showed BiLSTM h=32 at macro F1 0.470
against logreg's 0.479, and that was reported here as "BiLSTM loses to logistic
regression". Over 5 splits it does not: 0.496 +/- 0.025 against 0.472 +/- 0.014,
with ET precision 0.311 against 0.244. Both BiLSTM and TCN beat the linear
baseline, modestly. The claim is withdrawn.

The CNN's pooled win is real and tight -- 0.574 +/- 0.008, roughly seven
standard deviations above logreg.

### Leave-one-cohort-out: train on two cohorts, test on the third

| model | held-out | precET | recET | macroF1 |
|---|---|---|---|---|
| logreg | 2015 | 0.179 | 0.467 | 0.442 |
| logreg | NewData | 0.158 | 0.500 | 0.420 |
| logreg | PADS | 0.333 | 0.321 | 0.443 |
| MLPHead h=16 | 2015 | 0.242 | 0.533 | 0.517 |
| MLPHead h=16 | NewData | 0.111 | 0.167 | 0.342 |
| MLPHead h=16 | PADS | 0.286 | 0.143 | 0.442 |
| Spectrum1DCNN | 2015 | 0.214 | 0.400 | 0.540 |
| Spectrum1DCNN | NewData | **0.000** | **0.000** | 0.279 |
| Spectrum1DCNN | PADS | 0.467 | 0.250 | 0.456 |
| SpectrumTCN ch=16 | 2015 | 0.172 | 0.333 | 0.451 |
| SpectrumTCN ch=16 | NewData | 0.118 | 0.333 | 0.311 |
| SpectrumTCN ch=16 | PADS | 0.667 | 0.143 | 0.425 |
| SpectrumBiLSTM h=32 | 2015 | 0.186 | 0.533 | 0.470 |
| SpectrumBiLSTM h=32 | NewData | 0.053 | 0.167 | 0.265 |
| SpectrumBiLSTM h=32 | PADS | 0.267 | 0.286 | 0.363 |

Mean held-out macro F1:

| model | mean LOCO macroF1 | pooled macroF1 | gap |
|---|---|---|---|
| logreg | **0.435** | 0.472 | -0.037 |
| MLPHead h=16 | 0.434 | 0.508 | -0.074 |
| Spectrum1DCNN | 0.425 | 0.574 | **-0.149** |
| SpectrumTCN ch=16 | 0.396 | 0.484 | -0.088 |
| SpectrumBiLSTM h=32 | 0.366 | 0.496 | -0.130 |

**The CNN's +0.102 pooled advantage over logreg becomes -0.010 under LOCO.** The
ranking inverts: logreg and MLPHead lead, the CNN is third, and the two sequence
models are last. On an unseen NewData the CNN predicts *zero* ET patients
(precision 0.000, recall 0.000) while logreg reaches 0.420 -- its best cohort.

The size of the pooled-to-LOCO gap tracks model capacity almost monotonically
(-0.037 logreg, -0.074 MLP, -0.088 TCN, -0.130 BiLSTM, -0.149 CNN). That is the
signature of within-cohort specialisation, not of a better tremor
representation.

## Bottom line

* Pooled k-fold on merged cohorts **overstates deep models**, because PADS
  patients appear in both train and test. Report LOCO.
* Under LOCO no model beats logistic regression on average. The deep gain at
  n=355 is cohort fitting.
* For the 3-class problem the sequence models (TCN, BiLSTM) are the worst
  choice: best pooled-to-LOCO collapse *and* lowest held-out score.
* The merge remains worth doing for **n** (49 ET vs 15), but as training data,
  with LOCO as the evaluation.

Reproduce: `scratch/threedeep.py`, `scratch/threedeep_verify.py` (gitignored;
models in `models.architectures`, runners `kfold_proba` / `fit_predict_proba`).
