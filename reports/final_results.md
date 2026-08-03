# Canonical results — both diagnostic axes

Local cohort, subject-grouped 5-fold CV, patient-level. Balanced-class logistic
regression unless stated. All numbers reproduced in this session.

---

## AXIS 1 — N vs Tremor  (n = 155 patients: 61 Normal, 94 Tremor)

Features: biomarker + signal features per sensor (hand/lower/upper) + spatial,
conditions OUT+WING.

| metric | value |
|---|---|
| **Accuracy** | **0.884**  95% CI [0.832, 0.929] |
| Balanced accuracy | 0.884 |
| **AUC** | **0.936** |
| majority-class baseline | 0.606 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Normal | 0.831 | 0.885 | 0.857 | 61 |
| **Tremor** | **0.922** | 0.883 | **0.902** (95% CI 0.857–0.942) | 94 |
| macro avg | 0.876 | 0.884 | 0.880 | 155 |

Sensitivity (Tremor) 0.883 · Specificity (Normal) 0.885
Confusion: `[[TN 54, FP 7], [FN 11, TP 83]]`

**Deep BiLSTM alternative** (4 seeds, patient-level aggregation):
accuracy 0.866 ± 0.010, balanced 0.867 ± 0.008, AUC 0.935 ± 0.005 — i.e. the
engineered-feature model is **not** beaten by the deep model.

---

## AXIS 2 — PD vs ET  (n = 90 patients: 75 PD, 15 ET)

**Best configuration — `lower_arm` sensor only**, STFT spectral profile +
biomarker/signal features, condition OUT:

| metric | value |
|---|---|
| **AUC** | **0.873** |
| Balanced accuracy | 0.673 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| PD | 0.887 | 0.947 | 0.916 | 75 |
| **ET** | **0.600** | **0.400** | **0.480** | 15 |

ET detected 6/15 at only 4 false positives.

**Sensor comparison** (same axis, same patients) — the forearm sensor alone beats
using all three, mirroring the Axis-1/sensor-selection finding:

| config | AUC | ET precision | ET recall | ET F1 |
|---|---|---|---|---|
| **lower_arm only** | **0.873** | **0.600** | **0.400** | **0.480** |
| all 9 channels + spatial | 0.800 | 0.500 | 0.267 | 0.348 |

**Reporting note:** raw accuracy is meaningless on this axis — "always PD" scores
0.833. Report **AUC and balanced accuracy**, plus the per-class table.

### PADS augmentation on this axis — tested, does not help
Adding PADS's 41 ET patients (training ET 15 → 56; tested on local patients only,
with per-dataset standardisation):

| training set | AUC | ET precision | ET recall | ET F1 | ET found | false-ET |
|---|---|---|---|---|---|---|
| **LOCAL only** | **0.873** | **0.600** | 0.400 | **0.480** | 6/15 | **4** |
| LOCAL + PADS | 0.686 | 0.389 | 0.467 | 0.424 | 7/15 | 11 |

Quadrupling the ET training examples finds one extra ET patient but nearly
triples false positives and costs 0.19 AUC.

**Adding only the PADS ET patients (minority-class augmentation) is worse still**
— all variants tested, always evaluated on local patients:

| training set | AUC | ET prec | ET recall | ET F1 | ET found | false-ET |
|---|---|---|---|---|---|---|
| **LOCAL only (15 ET)** | **0.873** | **0.600** | 0.400 | **0.480** | 6/15 | **4** |
| + PADS ET only, aligned | 0.723 | 0.179 | **1.000** | 0.303 | 15/15 | **69** |
| + PADS ET only, not aligned | 0.735 | 0.364 | 0.267 | 0.308 | 4/15 | 7 |
| + PADS ET + equal PADS PD | 0.597 | 0.208 | 0.733 | 0.324 | 11/15 | 42 |

The aligned ET-only row is a cautionary case: **ET recall reaches 1.000**, which
looks like a breakthrough, but precision is 0.179 — it labels 69 of the 75 PD
patients as ET, i.e. it answers "ET" to nearly everything. Adding 41 PADS ET
makes training nearly class-balanced (75 PD vs 56 ET), shifting the boundary hard
toward ET; because PADS ET patients do not resemble local ET patients, the
shifted boundary does not transfer. AUC falls 0.873 → 0.723, so the *ranking*
degrades as well — this is not fixable by moving the threshold.

**Conclusion (all combinations tested): PADS cannot be used as training data for
this cohort — as a full set, PD+ET balanced, or ET-only, aligned or not. It is a
validation cohort.**

---

## Feature-set progression (the AI-contribution result)

### N vs Tremor
| feature set | accuracy | AUC |
|---|---|---|
| max + mean frequency only (2–6 feats) | 0.703 – 0.781 | 0.800 – 0.871 |
| **full engineered features** | **0.884** | **0.936** |
| deep BiLSTM (4-seed) | 0.866 ± 0.010 | 0.935 |

### PD vs ET
| feature set | AUC | ET precision | ET recall | ET F1 |
|---|---|---|---|---|
| max + mean frequency only | 0.582 (chance) | — | — | — |
| summary features (dom freq, band powers, entropy, regularity) | 0.589 | 0.273 | 0.375 | 0.316 |
| **STFT spectral profile + spatial** | **0.800** | **0.500** | 0.267 | 0.348 |

**Core finding:** on the hard PD-vs-ET axis, classical spectral summaries are at
chance (AUC 0.58–0.59); AUC rises to **0.800 only when the model is given the
full spectral shape**. The discriminative information exists but is discarded by
hand-crafted summary statistics — the motivating result for learned features and
the natural question for explainable AI.

**Honest limits.** AUC 0.800 reflects good *ranking*, not good detection: at the
operating point ET recall is 0.267 (4/15 patients). With 15 ET patients this is
the realistic ceiling; the ET F1 CI [0.10, 0.59] is correspondingly wide.

## Targets vs achieved

| target | achieved | verdict |
|---|---|---|
| N-vs-Tremor 70% (classical) | 0.703 with **2 features** | met |
| N-vs-Tremor 90% | 0.884 [0.832, 0.929]; deep 0.866 ± 0.010 | not demonstrated; CI includes 0.90 |
| PD-vs-ET 70% (classical) | chance (AUC 0.58) | not achievable classically |
| PD-vs-ET 90% | AUC 0.800, balanced 0.607 | not achievable on this cohort |

A single deep run reached 0.903 on Axis 1, but 4-seed verification gave
0.866 ± 0.010 with 0/4 seeds ≥ 0.90 — that run was a favourable draw, not a
reproducible result.

## Resolving the PD/ET frequency overlap — what carries the signal

Peak/mean frequency cannot separate PD from ET (identical 6.64 Hz medians). Where
does the AUC 0.87 come from? Feature-block ablation (lower_arm, OUT, PD-vs-ET):

| feature block | AUC |
|---|---|
| biomarker summaries (13) | 0.594 |
| regularity / advanced (10) | 0.605 |
| biomarker + advanced (23) | 0.612 |
| **STFT spectral profile (234)** | **0.882** |
| all combined (257) | 0.873 |

**The signal is entirely in the spectral profile**; summary statistics dilute it.

### Which frequencies differ (per-bin Mann-Whitney, relative power)

| Hz | effect | p |
|---|---|---|
| **6.64** | −0.49 | **0.003** |
| **6.25** | −0.43 | **0.009** |
| 7.03 | −0.32 | 0.050 |
| 3.91 | +0.30 | 0.072 |
| 4.30 | +0.29 | 0.079 |

(negative = ET has more relative power; positive = PD does)

PD and ET **peak at the same 6.64 Hz**, but ET concentrates power there while PD
retains an additional **~4 Hz component** — the classical parkinsonian band
sitting alongside the shared peak. This is why peak-based measures are blind to
the difference, and why wide bands (`rel_b_3_5`, `rel_b_5_7`) miss it: they
average the discriminative 6.25–6.64 Hz region together with neutral neighbours.

### A narrow-band ratio captures it locally but does NOT replicate

Single feature `log(P[3.7–4.6 Hz] / P[6.0–6.9 Hz])`:

| | LOCAL | PADS (independent) |
|---|---|---|
| PD median | −0.382 | −0.402 |
| ET median | −0.968 | −0.784 |
| effect | +0.47 | +0.13 |
| p | **0.0046** | 0.165 (ns) |
| AUC | 0.724 | ~0.57 |

The direction is consistent but the effect is weak and non-significant
externally. The local AUC 0.724 is optimistically biased — the bands were chosen
on that same data. **Do not build a claim on this single-feature biomarker.**

### Conclusion
The PD/ET overlap **is** resolvable, but only with the **full spectral profile**
(AUC 0.882), not with hand-picked summary features or a two-band ratio. The
discriminative structure is distributed across the spectrum rather than
localisable to one interpretable statistic — the strongest argument in this work
for learned features over classical summaries, and a well-posed question for
explainable AI: *which parts of the spectrum does the model combine, and how?*
