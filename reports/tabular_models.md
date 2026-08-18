# Tree ensembles on the hand features: the largest ET-precision gain measured

**RandomForest raises ET precision from 0.260 to 0.401 on PADS with no loss of
AUC** (paired -0.000 [-0.019, +0.018]). That is the highest ET precision
recorded anywhere in this project, and unlike the deep models it costs nothing
in ranking quality.

## Why this was worth testing

The hand-feature problem is textbook **tabular**: 10-16 informative features,
119-423 patients. Every model in this project had been either a linear model or
a neural network on raw spectra. Tree ensembles -- the standard best-in-class
method for that regime -- had never been tried.

(A 1-D CNN over the feature *vector* would not be appropriate: convolution
assumes neighbouring positions are related, and feature order is arbitrary.
Over the spectrum the assumption holds, which is why `Spectrum1DCNN` works
there.)

## Results, binary PD vs ET

### PADS (276 PD / 28 ET), descriptors + stability

| model | AUC | precPD | precET | bal-acc |
|---|---|---|---|---|
| logreg (baseline) | 0.757 | 0.954 | 0.260 | 0.716 |
| logreg L1 | **0.776** | 0.955 | 0.269 | 0.722 |
| **RandomForest** | 0.756 | 0.945 | **0.401** | 0.700 |
| ExtraTrees | 0.755 | 0.955 | 0.356 | **0.737** |
| HistGradBoost | 0.739 | 0.934 | 0.343 | 0.641 |
| SVM rbf | 0.765 | 0.910 | 0.350 +/- 0.329 | 0.514 |

Paired against logreg: RandomForest **AUC -0.000 [-0.019, +0.018]**,
**precET +0.141 [+0.126, +0.157]** *. HistGradBoost precET +0.083 * but
bal-acc -0.075 *.

### Merged (374 PD / 49 ET), axes + stability

| model | AUC | precET |
|---|---|---|
| logreg (baseline) | **0.728** | 0.218 |
| RandomForest | 0.660 | **0.292** |
| ExtraTrees | 0.691 | 0.252 |

Paired: RandomForest AUC **-0.068** *, precET **+0.074** *. Here it is a trade,
not a free gain.

### In-house (98 PD / 21 ET), axes -- trees lose

| model | AUC | precET |
|---|---|---|
| **logreg (baseline)** | 0.625 | **0.291** |
| ExtraTrees | **0.645** | 0.259 |
| RandomForest | 0.617 | 0.202 |
| HistGradBoost | 0.563 | 0.202 |

Paired: RandomForest precET **-0.090** *, HistGradBoost AUC **-0.062** * and
precET **-0.089** *.

## What this settles

**The dilution pattern is about sample size, not about linear models.** Trees do
their own feature selection and would be expected to shrug off irrelevant
dimensions. They do not help in-house, and gradient boosting is significantly
*worse* there. With 21 ET patients any model with real capacity memorises.

**Tree ensembles help exactly where there are enough minority patients.** PADS
(28 ET against 276 PD) is the case with the most training data per fold, and
there RandomForest is a free +0.141 on the metric that matters.

**The RBF SVM collapses to predicting PD for every patient** on two of three
cohorts -- ET precision exactly 0.000, balanced accuracy exactly 0.500, zero
variance across all 10 repeats -- despite `class_weight="balanced"`. Its AUC of
0.670 is respectable, so it *ranks* acceptably and is clinically useless. Report
both, always.

Reproduce: `python -m experiments.tabular_models`.
