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

## Caveat

The numbers above come from one fold split (`random_state=0`). Stability across
5 splits and a leave-one-cohort-out generalisation test are in
`scratch/threedeep_verify.py`; this repo has retracted three findings that
looked this clean at one split.

Reproduce: `scratch/threedeep.py` (gitignored; models in
`tfbench.small_nets`, runner `kfold_proba`).
