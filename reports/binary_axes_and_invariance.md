# Binary axes (the real targets) + the invariance sweet spot

## 1. The two diagnostic axes, measured honestly

Local data, lower_arm + OUT, TF+spatial features, subject-grouped 5-fold,
balanced-class logistic regression.

### N vs tremor — strong, and 90% is reachable
| metric | value |
|---|---|
| accuracy | 0.854 |
| balanced accuracy | 0.862 |
| **AUC** | **0.937** |
| majority-class baseline | 0.596 |

Clearly beats the baseline; AUC 0.937 indicates real headroom. **A 90% target on
this axis is legitimate.**

### PD vs ET — accuracy is a trap here
| metric | value |
|---|---|
| accuracy | 0.833 |
| **majority-class baseline ("always PD")** | **0.833** |
| balanced accuracy | 0.607 |
| **AUC** | **0.800** |
| ET-F1 | 0.348 |

**The accuracy exactly equals the do-nothing baseline** (the cohort is 125 PD vs
29 ET). Reporting "PD-vs-ET accuracy = 83%" — or optimising toward "90%
accuracy" — would be an artifact of class imbalance and is indefensible in
review. The genuine signal is **AUC 0.800**: the model ranks ET above PD well
above chance, it simply cannot commit to ET at a default threshold.

**Reporting rule for this axis: use balanced accuracy and AUC, never raw
accuracy.** A realistic target is AUC ≈ 0.85, not 90% accuracy.

## 2. Invariance sweet spot (cross-dataset feature design)

Incrementally adding feature groups to the rotation+scale-invariant spectral
shape, tracking the dataset-identity probe (lower = more invariant) against
ET-F1 on the local cohort and with PADS augmentation:

| cumulative feature set | dim | identity AUC | LOCAL ET-F1 | AUGMENTED ET-F1 |
|---|---|---|---|---|
| shape only | 40 | 0.488 | 0.276 | 0.323 |
| **+ relative band powers** | 45 | **0.533** | 0.271 | **0.343** |
| + frequency features | 49 | 0.543 | 0.245 | 0.278 |
| + shape/sharpness | 53 | 0.564 | 0.286 | 0.263 |
| + regularity | 58 | **0.977** | 0.233 | 0.359 |
| + absolute power | 60 | 0.987 | 0.261 | 0.457 |

**Sweet spot: spectral shape + relative band powers** (45-dim, identity AUC 0.533
≈ chance). It is the richest genuinely dataset-invariant set, and PADS
augmentation helps there (ET-F1 0.271 → 0.343).

**The regularity group is what destroys invariance** (0.564 → 0.977) — those
features (sample entropy, Hjorth, AM depth, frequency stability) are sensitive to
noise floor and smoothing, which differ between devices (PADS peaks are ~3.3×
larger at equal RMS).

**Caution on the apparent best (AUG ET-F1 0.457 with absolute power):** it comes
with identity AUC 0.987, i.e. the classifier can tell the datasets apart and can
therefore specialise per-dataset. That gain is not trustworthy generalisation.
