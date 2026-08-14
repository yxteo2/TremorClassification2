# Window-level vs patient-level metrics: my inflation hypothesis was wrong

**Question this answers.** A published bilateral-attention paper reports
HC-vs-PD accuracy 0.932 / AUROC 0.963 and "PD vs DD" accuracy 0.909, while
everything measured in this repo sits at 0.72-0.83 on N-vs-Tremor and at chance
on PD-vs-ET. The paper's Table 2 caption reads *"window-level, with band-pass,
5 patient-disjoint folds"* -- their unit of evaluation is a 2-second window, not
a patient. I hypothesised that this metric definition alone explains a large
part of the gap, because windows from the same patient are correlated and a
per-window average is dominated by whoever contributed the most windows.

**It does not.** Measured directly, on the same features and the same classifier,
switching from patient-level to window-level makes N-vs-Tremor *worse*, not
better.

## Protocol

The paper's windowing: 2 s windows at 50 % overlap (200 samples, hop 100 at
100 Hz), lower-arm angular velocity, 3-15 Hz. Features are the repo's 10
descriptors. Classifier is `StandardScaler + LogisticRegression(class_weight=
"balanced")`, 5-fold `GroupKFold` grouped by **patient** in both arms -- so the
folds are patient-disjoint either way and the only thing that changes is what a
row is. Patient-level rows use `stft512` descriptors averaged per patient.

## Result

|                                | n    | acc   | bal-acc | AUC   | F1    | sens  |
|--------------------------------|------|-------|---------|-------|-------|-------|
| 2015 OUT N-vs-Tremor -- window  | 3891 | 0.770 | 0.774   | 0.861 | 0.785 | 0.746 |
| 2015 OUT N-vs-Tremor -- patient | 151  | 0.808 | **0.815** | 0.896 | 0.828 | 0.778 |
| 2015 OUT PD-vs-ET -- window     | 2186 | 0.540 | **0.528** | 0.583 | 0.286 | 0.510 |
| 2015 OUT PD-vs-ET -- patient    | 90   | 0.544 | 0.433   | 0.484 | 0.163 | 0.267 |
| NewData OUT N-vs-Tremor -- window  | 900 | 0.672 | 0.673 | 0.738 | 0.687 | 0.665 |
| NewData OUT N-vs-Tremor -- patient | 56  | 0.714 | **0.714** | 0.802 | 0.724 | 0.724 |
| NewData OUT PD-vs-ET -- window     | 486 | 0.582 | **0.464** | 0.385 | 0.210 | 0.250 |
| NewData OUT PD-vs-ET -- patient    | 29  | 0.483 | 0.366   | 0.275 | 0.118 | 0.167 |

Reproduce: `scratch/windowlevel.py` (gitignored; the script is short and the
protocol above is complete).

## Reading it

* **N-vs-Tremor: aggregating windows to patients HELPS** (+0.041 bal-acc on
  2015, +0.041 on NewData; AUC +0.035 / +0.064). Averaging the spectrum over a
  whole recording is a denoising step, and it buys more than the correlated-row
  effect costs. So "they report window-level, therefore their number is
  inflated" is **not supported on this data** and should not be used as an
  argument in the paper.
* **PD-vs-ET: window-level is higher** (0.528 vs 0.433; 0.464 vs 0.366), but
  both arms are at or below chance. This is not evidence that windowing helps --
  it is two noisy estimates of the same nothing, at 90 and 29 patients.
* The direction being *opposite* on the two axes is itself the tell: if the
  correlated-window effect were the dominant term it would inflate both.

## What is left of the comparison

The metric definition is not the explanation. Three things still are, and they
are the honest ones to put in the paper:

1. **Cohort size.** The published cohort is ~469 patients on both wrists at
   100 Hz with HC/PD/DD groups -- that description matches PADS. Against 151
   (2015) and 56 (NewData) patients here, that is a different regime, and the
   measured learning curve on PADS descriptors is still rising at our n.
2. **"PD vs DD" is not PD-vs-ET.** DD (differential diagnoses) includes atypical
   parkinsonism -- PSP, MSA, vascular parkinsonism -- alongside ET. In PADS
   itself, 20 records labelled parkinsonian are Atypical Parkinsonism (see
   `pads_label_bug.md`). A PD-vs-{everything-else-that-shakes} boundary is a
   different and plausibly easier problem than PD-vs-ET, and their 0.909 should
   not be quoted as a PD-vs-ET number.
3. **Band-pass before windowing.** They band-pass first; the descriptors here
   integrate the 3-15 Hz band without pre-filtering. This is a small effect but
   it is a real protocol difference.

## Correction to an earlier claim

I set this experiment up expecting window-level inflation and said so before
measuring it. The measurement does not support that, and the claim is
withdrawn. Nothing in the repo's results depended on it -- every number reported
elsewhere is already patient-level, which this shows is the *conservative*
choice on the axis that works.
