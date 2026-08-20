# catch22: a temporal family that matches the spectral one, and does not beat it

## Why this was tried

A literature scan for ways to improve PD-vs-ET turned up one directly usable
lead. **Häring et al., *Movement Disorders* 2025** ("Phenotypical Differentiation
of Tremor Using Time Series Feature Extraction and Machine Learning") applied
massive time-series feature extraction to hand accelerometry from **414
patients** — a cohort almost exactly the size of this project's merged 404 — and
reported for PD vs ET:

| | accuracy | sensitivity | specificity |
|---|---|---|---|
| feature-based ML | **81.8 %** | 86.4 % | 76.6 % |
| Tremor Stability Index | 70.4 % | 70.8 % | 70.2 % |

Two things made it worth acting on. First, **their TSI baseline and ours agree** —
this repo implements TSI in `stability.py` and measures AUC 0.757 on PADS, so the
comparison is like-for-like and their headline is the interesting number. Second,
their reading is mechanistic and testable:

> *"different discrete but stable signal states in PD indicate several central
> oscillators, while signal characteristics in ET point towards a singular
> pacemaker"*

Every feature family in this project is derived from the **power spectrum** or
from the instantaneous-frequency trajectory. **None reads the waveform's temporal
structure directly**, which is exactly what that claim is about.

They also report **linear SVMs reaching 86.1 % and beating random forests**,
which matches this project's own finding that simple models win at this n.

## What was built

`catch22` — the canonical 22-feature distillation of the hctsa library — rather
than tsfresh. tsfresh emits hundreds of features and needs a selection stage
fitted inside every fold; catch22 is a **fixed** set chosen once, offline, on 93
unrelated datasets, so there is nothing to leak and it is comparable in size to
the 10 spectral descriptors already in use. That matters here: thirteen feature
unions have underperformed their best member because dimensionality binds harder
than information at 49 ET patients.

Six of the 22 encode the mechanism above, and were fixed **a priori** from the
paper rather than selected on this data:

    SB_TransitionMatrix_3ac_sumdiagcov   transitions between discretised states
    SB_BinaryStats_mean_longstretch1     how long the signal stays in one state
    SB_BinaryStats_diff_longstretch0     the same for the differenced signal
    SB_MotifThree_quantile_hh            entropy of 3-symbol motifs
    DN_HistogramMode_5 / _10             multimodality of the amplitude
                                         distribution

**Rotation invariance.** catch22 takes one series; the sensor gives three axes.
The magnitude ‖ω(t)‖ is rotation-invariant but wrong — for a linear oscillation
it has fundamental **2f**. Verified on a 6 Hz synthetic: the magnitude peaks at
**11.91 Hz**, the principal-axis projection at **6.05 Hz**. So the pipeline
band-passes to 3–15 Hz, projects onto the leading eigenvector of the time-domain
covariance, fixes the arbitrary sign by forcing non-negative skewness, and
z-scores. Invariance checks pass at 5×10⁻¹³ for rotation and 7×10⁻¹³ for scale.

## Result — PADS PD vs ET, 28 ET, permutation-tested

| family | dim | AUC | precET | p (perm) |
|---|---|---|---|---|
| descriptors (spectral) | 10 | 0.794 | **0.464** | 0.005 * |
| stability / TSI | 6 | 0.758 | 0.464 | 0.005 * |
| spectrum | 16 | 0.798 | 0.393 | 0.005 * |
| catch22, full | 22 | 0.761 | 0.393 | 0.005 * |
| **catch22 state subset** | **6** | **0.805** | 0.429 | 0.005 * |
| descriptors + catch22 | 32 | 0.749 | 0.357 | 0.005 * |

The 6-feature mechanism subset gives the **highest AUC measured on PADS**, above
descriptors and above the full 22 it is drawn from. The single best-separating
catch22 feature is `SB_MotifThree_quantile_hh` (Cohen's d **1.235**), a state
feature — and it tops the list on MERGED too (d 0.610).

## Paired, over 20 repeats — where it stops

| PADS arm | dim | AUC | precET | sd(AUC) |
|---|---|---|---|---|
| descriptors | 10 | 0.794 | **0.448** | 0.023 |
| catch22 state | 6 | 0.798 | 0.414 | **0.012** |
| **rank-avg hybrid** | – | **0.808** | 0.421 | 0.013 |
| concat (control) | 16 | 0.769 | 0.380 | 0.014 |

paired against descriptors:

| arm | AUC | precET |
|---|---|---|
| catch22 state | +0.003 [−0.009, +0.015] | **−0.034 [−0.055, −0.013]** * |
| rank-avg hybrid | **+0.014 [+0.008, +0.019]** * | **−0.028 [−0.045, −0.009]** * |
| concat | **−0.026** * | **−0.068** * |

**The combination rule worked, and it bought the wrong metric.** The hybrid
gives a significant AUC gain — two members comparable in strength (0.794 vs
0.798) that differ in kind (spectral vs temporal), combined at the score level,
exactly the configuration `score_vs_feature_fusion.md` predicts should pay. It
also **significantly worsens ET precision**, which is the metric this project
optimises.

The divergence has a mechanism. AUC integrates over the whole ranking; precision
at 9.2 % prevalence is read from the very top of it. Rank-averaging improves the
middle of the ranking while diluting descriptors' most confident ET calls, which
is where precision lives. The same AUC-up/precision-down pattern appeared in
`score_vs_feature_fusion.md`.

On MERGED the hybrid's AUC gain disappears entirely (+0.001, ns) and precET is
still −0.024 *.

## What this is worth

**Not an improvement.** Descriptors remain the best PD-vs-ET model on PADS by ET
precision, and nothing here changes the headline.

**But two things are genuinely new:**

1. **An independent, mechanism-derived replication.** Six temporal features
   chosen from someone else's hypothesis, never tuned on this data, match ten
   tuned spectral descriptors on AUC (0.798 vs 0.794) using **60 % of the
   dimensions and half the variance** (sd 0.012 vs 0.023). The "discrete stable
   states" account of PD tremor holds up on a cohort its authors never saw.
2. **The lower variance is the practically useful part.** At 28 ET, sd(AUC) 0.012
   against 0.023 means the state family is markedly more stable across folds.
   That is worth having in a paper even without a mean gain.

**In-house it changes nothing.** No arm reaches significance (best p = 0.075),
consistent with `permutation_null.md`: at 21 ET the null spans [0.298, 0.655] and
nothing measured clears it.

**Concatenation diluted twice more** (PADS −0.026 AUC *, MERGED −0.040 *),
bringing the count of feature unions that underperform their best member to
fifteen.

## Context this scan also settled

* **The PADS published baseline for the hard axis is 72.42 % balanced accuracy**
  for PD vs DD (Varghese et al., *npj Parkinson's Disease* 2024) against 91.16 %
  for PD vs HC. This project's PADS PD-vs-ET AUC 0.794 sits in the same regime.
  The difficulty is not an artifact of this pipeline.
* **A 2026 arXiv preprint reports 87.04 % for PD vs DD** on PADS using
  self-supervised dual-channel cross-attention. Its preprocessing applies the
  **Differential Hopping Windowing Technique with class-dependent overlap** — 70 %
  for HC, 0 % for PD, 65 % for DD. Window overlap chosen by class label is a
  preprocessing step that uses the target, and unless the splits are strictly
  patient-level the overlapping windows put near-duplicate segments on both sides
  of the split. Treat that number with caution; it is not a like-for-like
  comparator. (This repo separately measured that window-level evaluation is
  *not* inflationary here — `window_vs_patient_level.md` — so the concern is the
  class-conditional overlap, not the windowing.)
* **Häring's 81.8 % remains the credible target** for PD vs ET at this cohort
  size, and the gap to it is real rather than a measurement difference.

## Reproducing

`pip install pycatch22` (recorded in `requirements-extra.txt`), then
`python -m experiments.catch22_family` and
`python -m experiments.catch22_hybrid`.

## Sources

* Häring et al., *Movement Disorders* 2025 — https://movementdisorders.onlinelibrary.wiley.com/doi/10.1002/mds.70032
* Varghese et al., *npj Parkinson's Disease* 2024 (PADS) — https://www.nature.com/articles/s41531-023-00625-7
* PADS dataset — https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
* Self-supervised cross-attention on PADS — https://arxiv.org/abs/2604.18372
