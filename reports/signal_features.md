# Signal-processing features for PD/N/ET discrimination

Focus: does *richer data processing* — features describing how the tremor
**behaves** (regularity, peak sharpness, frequency stability), not just where
its spectral peak sits — improve separation? Code in `pdetn/signal_features.py`.

## Features added
From a 3–15 Hz bandpassed, hand-sensor angular-velocity magnitude:
peak Q-factor, spectral flatness/centroid/spread, half-power bandwidth, Hjorth
mobility & complexity, **sample entropy** (regularity), **frequency-stability
std**, amplitude-modulation depth.

## Finding 1 — they DO discriminate PD-vs-ET univariately
Mann-Whitney PD-vs-ET (rank-biserial effect, p), best per condition:

| condition | feature | effect | p |
|---|---|---|---|
| REST | Hjorth mobility | −0.37 | **0.002** |
| REST | spectral spread | −0.28 | 0.019 |
| OUT | **sample entropy** | −0.33 | **0.006** |
| OUT | amplitude-mod depth | +0.24 | 0.041 |
| OUT | peak Q-factor | −0.23 | 0.056 |

Physiology confirmed and quantified: **PD tremor is more *regular*** (lower
sample entropy) and lower-mobility than ET — a discriminative axis beyond
dominant frequency.

## Finding 2 — but they do NOT improve the classifier
Two-stage (logreg, tuned ET threshold), leave-one-patient-out, subject CIs:

| feature set | #feat | macro-F1 | ET-F1 |
|---|---|---|---|
| **base spectral (best)** | 48 | **0.582** | **0.324** [0.15, 0.50] |
| base + advanced (all) | 78 | 0.495 | 0.175 |
| base + advanced, SelectKBest k=12 | — | 0.500 | 0.222 |
| base spectral, SelectKBest k=20 | — | 0.577 | 0.320 |
| base spectral, SelectKBest k=10 | — | 0.466 | 0.200 |

Adding the processed features **overfits** the 16-ET cohort (curse of
dimensionality); feature selection does not rescue it. Base condition-aware
spectral features, no selection, remain best.

## Conclusion
Better signal processing extracts **real, interpretable discriminative
structure** (useful as biomarkers — PD is more regular than ET) but does **not
raise the classification ceiling**. Consistent with every other axis tried
(architecture, fusion, TFD method, wavelet depth): on 16 ET subjects the
**cohort size is the binding constraint, not the data representation**. The
advanced features remain available via `build_patient_table(..., advanced=True)`
and are valuable for the interpretability/biomarker narrative, but the default
classifier uses the base spectral set.

Reproduce: the "signal-processing features" section of `pdetn/experiments.ipynb`.
