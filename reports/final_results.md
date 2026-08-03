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

Features: STFT spectral profile + spatial, condition OUT.

| metric | value |
|---|---|
| Accuracy | 0.833  95% CI [0.756, 0.911] — **equals the 0.833 majority baseline** |
| Balanced accuracy | 0.607 |
| **AUC** | **0.800** |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| PD | 0.866 | 0.947 | 0.904 | 75 |
| **ET** | **0.500** | **0.267** | **0.348** (95% CI 0.095–0.588) | 15 |
| macro avg | 0.683 | 0.607 | 0.626 | 90 |

Sensitivity (ET) 0.267 · Specificity (PD) 0.947
Confusion: `[[TN 71, FP 4], [FN 11, TP 4]]`

**Reporting note:** raw accuracy is meaningless on this axis — "always PD" scores
0.833. Report **AUC and balanced accuracy**, and the per-class table.

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
