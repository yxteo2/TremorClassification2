# SMOTE and resampling: the variant matters more than the idea

Every model in this project handled imbalance through **loss reweighting**
(`class_weight="balanced"`). Data-level resampling had never been tried.

**Plain SMOTE hurts. The boundary-focused variants help, but only where there
are enough minority patients.**

## Method note that decides whether these numbers mean anything

Resampling is fitted **inside each training fold**, via `imblearn.Pipeline`, so
it applies at `fit` time only and never touches the data being predicted.
Fitting SMOTE before the split creates synthetic training points interpolated
from *test* patients -- the most common way SMOTE results are overstated in
published work, and worth 0.1-0.2 AUC of pure leakage.

## Results, binary PD vs ET, logistic regression

| sampler | in-house precET (21 ET) | merged precET (49 ET) | PADS precET (28 ET) |
|---|---|---|---|
| none (class_weight) | **0.291** | 0.218 | 0.268 |
| RandomOver | 0.293 | 0.222 | 0.262 |
| SMOTE | 0.273 | 0.206 | 0.235 |
| BorderlineSMOTE | 0.288 | 0.241 | 0.301 |
| ADASYN | 0.273 | 0.204 | 0.230 |
| **SVMSMOTE** | 0.291 | **0.291** | **0.336** |
| SMOTETomek | 0.274 | 0.206 | 0.238 |
| SMOTEENN | 0.244 | 0.171 | 0.219 |

SVMSMOTE paired against the class-weight baseline:

| cohort | AUC | precET | bal-acc |
|---|---|---|---|
| in-house | +0.007 [-0.004, +0.018] | -0.001 [-0.023, +0.023] | **-0.039** * |
| **merged** | -0.011 * | **+0.073 [+0.063, +0.083]** * | **+0.015 [+0.005, +0.025]** * |
| **PADS** | -0.006 [-0.019, +0.003] | **+0.068 [+0.051, +0.083]** * | -0.018 [-0.040, +0.000] |

Plain SMOTE paired: in-house precET **-0.018 [-0.026, -0.009]** *, and worse on
both other cohorts.

## Why the variant matters

Plain SMOTE interpolates between **randomly chosen** minority neighbours. With a
sparse minority manifold -- 14 to 39 ET patients per training fold in a 4-16
dimensional space -- that manufactures ET patients in regions no real patient
occupies, and the classifier learns a boundary around fiction.

`SVMSMOTE` and `BorderlineSMOTE` synthesise **only near the decision boundary**,
which is far more conservative, and they are the two that survive. `SMOTEENN`,
the most aggressive (synthesise then clean), is the worst everywhere.

## Where it does and does not work

**In-house (21 ET): nothing helps.** SVMSMOTE is flat on ET precision and
significantly worse on balanced accuracy; plain SMOTE, ADASYN and SMOTETomek are
all significantly worse. At ~14 ET patients per training fold, interpolation has
too few real points to work from.

**Merged and PADS: SVMSMOTE is a real gain** of +0.07 ET precision, at a small
or no AUC cost.

This is the same threshold seen with tree ensembles (`tabular_models.md`), which
also help on PADS and merged and lose in-house. Two independent methods, one
boundary: **methods that add capacity or synthesise data need more minority
patients than the in-house cohort has.**

Reproduce: `python -m experiments.resampling`.

## RandomForest + SVMSMOTE do not combine

Both improve ET precision independently on PADS (+0.141 and +0.102). Stacking
them is worse than either alone.

### PADS

| model | AUC | precET |
|---|---|---|
| logreg (baseline) | 0.757 | 0.260 |
| **RandomForest** | 0.756 | **0.401** (+0.141 *) |
| logreg + SVMSMOTE | **0.764** | 0.362 (+0.102 *) |
| RF + SVMSMOTE | 0.749 | 0.374 (+0.114 *) |
| RF + BorderlineSMOTE | 0.750 | 0.355 (+0.095 *) |

### Merged

| model | AUC | precET | bal-acc |
|---|---|---|---|
| logreg (baseline) | **0.728** | 0.218 | 0.653 |
| RandomForest | 0.660 | 0.292 | 0.651 |
| **logreg + SVMSMOTE** | 0.717 | 0.291 | **0.668** |
| RF + SVMSMOTE | 0.652 | 0.276 | 0.638 |

**Why they do not stack:** both do the same job -- moving the decision boundary
toward the minority class. RandomForest does it with class-weighted splits,
SVMSMOTE by planting synthetic minority points near the boundary. Applying both
over-corrects, and the combination lands between the two rather than above them.

## Final PD-vs-ET recommendation per cohort

| cohort | model | AUC | precPD | precET |
|---|---|---|---|---|
| in-house (21 ET) | logreg on 4 axis features | 0.625 | 0.879 | 0.291 |
| merged (49 ET) | logreg + SVMSMOTE | 0.717 | 0.927 | 0.291 |
| PADS (28 ET) | RandomForest | 0.756 | 0.945 | **0.401** |

On the merged cohort `logreg + SVMSMOTE` is preferred over RandomForest despite
identical ET precision (0.291 against 0.292): it holds AUC at 0.717 against
0.660 and significantly improves balanced accuracy. Same precision, better
ranking.
