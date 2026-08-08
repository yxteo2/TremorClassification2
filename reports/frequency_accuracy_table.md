# Accuracy from peak + mean frequency alone — all three datasets

Two features only (`max_freq`, `mean_freq`) from a Welch PSD, 3–15 Hz,
patient-level LOSO, class-balanced logistic regression. Wrist-equivalent sensor
throughout (2015/NewData `lower_arm`, PADS wrist). NewData uses the corrected
10 s task segmentation.

| dataset | axis | n neg/pos | accuracy | majority | **bal-acc** | AUC | F1 [95% CI] |
|---|---|---|---|---|---|---|---|
| **2015** (OUT) | N vs Tremor | 61/90 | 0.715 | 0.596 | **0.724** | 0.808 | 0.739 [0.66–0.81] |
| **2015** (REST) | N vs Tremor | 61/91 | 0.704 | 0.599 | **0.707** | 0.798 | 0.737 [0.66–0.81] |
| **PADS** (StretchHold) | N vs Tremor | 79/304 | 0.629 | 0.794 | **0.612** | 0.664 | 0.733 [0.69–0.77] |
| **2015** (OUT) | PD vs ET | 75/15 | 0.489 | 0.833 | **0.453** | 0.465 | 0.207 [0.07–0.35] |
| **2015** (REST) | PD vs ET | 75/16 | 0.571 | 0.824 | **0.568** | 0.628 | 0.316 [0.15–0.47] |
| **PADS** (StretchHold) | PD vs ET | 276/28 | 0.724 | 0.908 | **0.703** | 0.775 | 0.311 [0.20–0.41] |
| **NewData** (OUT) | either axis | — | — | — | — | — | **not measurable** |
| 2015 + **NewData** ET pooled | PD vs ET | 75/21 | 0.500 | 0.781 | **0.474** | 0.486 | 0.273 [0.13–0.41] |

## Why NewData has no accuracy of its own

All 6 subjects are **ET**. With one class there is nothing to discriminate, so
no accuracy, AUC or F1 exists — reporting a number here would be meaningless.
The only way NewData contributes is by adding its ET subjects to a cohort that
has PD, which is the last row: **ET 15 → 21, and it does not help**
(bal-acc 0.453 → 0.474, AUC 0.465 → 0.486, both at chance).

**Read `accuracy` against `majority`, never alone.** PADS PD-vs-ET accuracy of
0.724 looks strong but is *below* its 0.908 majority baseline — the model is
worse than always guessing PD on raw accuracy while being genuinely informative
on balanced accuracy (0.703) and AUC (0.775). This is why balanced accuracy is
the column to use.

## What the table says

1. **N vs Tremor — your data is the best of the three.** 2015 OUT reaches
   bal-acc 0.724 / AUC 0.808 from two frequency numbers; PADS manages only
   0.612 / 0.664. Your acquisition is not the weaker one.
2. **PD vs ET — your data is at chance except at REST.** OUT 0.453 and
   AUC 0.465 mean frequency carries no PD/ET information in your postural task.
   REST reaches 0.568 / AUC 0.628 — weak, but the only place it appears.
3. **PADS separates PD from ET where you cannot** (0.703 / 0.775) on the same
   two features and an equivalent postural task. The contrast exists in that
   cohort and not in yours.
4. **Adding NewData ET does not fix it.** Even segmented, pooling to 21 ET
   leaves PD-vs-ET at chance.

## Pipeline status

| fix | in the default path? |
|---|---|
| PADS strict labels (`strict=True`) | **yes** |
| power `\|S\|` → `\|S\|²` (multitaper/cwt/hht/sst) | **yes** |
| AR innovation-variance gain | **yes** |
| descriptors integrate with bin width | **yes** |
| paired bootstrap on balanced accuracy | **yes** |
| Bonferroni reported separately from the CI | **yes** |
| NewData 10 s task segmentation | **yes — `segment=True` is now the default** |

**Stale and being re-run:** `reports/tfbench_stage1_results.md` and the cached
`artifacts/tfbench_tables.npz` were computed **before** the power fixes, so the
stage-1 method ranking used amplitude-weighted descriptors. The cache has been
deleted and the ranking is re-running; until it lands, treat the
`hht_imf2plus` result as provisional.
