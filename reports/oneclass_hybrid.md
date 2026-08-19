# One-class modelling of PD, rank-averaged with logistic regression

**Question this answers.** Every discriminative model here is asked to learn
what ET looks like from 21 (in-house) or 28 (PADS) patients. One-class
classification inverts the problem: model the *abundant* class — PD, 98 and 276
patients — and score ET as deviation from it. The minority class then never has
to be learned, only detected, so the 28-patient boundary that has stopped
trees, SMOTE variants, attention and every pretrained backbone should not bind
in the same way.

Run: `python -m experiments.oneclass_paired`. 20 repeats, patient-level
stratified folds, PD-vs-ET only (tremor patients, no controls). Threshold is set
at the prevalence quantile, so precision equals recall per class and the
operating point is not tuned on test.

## Result

**In-house (2015 + NewData), n=119, 98 PD / 21 ET**

| method | precPD | precET | macroP | AUC | PPV@0.5 |
|---|---|---|---|---|---|
| logreg | 0.851 | 0.302 | 0.576 | 0.613 | 0.669 |
| one-class Mahalanobis | 0.856 | 0.329 | 0.592 | 0.620 | 0.695 |
| **rank-averaged hybrid** | **0.856** | 0.325 | 0.591 | **0.634** | 0.693 |

paired vs logreg, 20 repeats:

| | one-class alone | rank-averaged hybrid |
|---|---|---|
| precPD | +0.006 [−0.000, +0.012] | **+0.006 [+0.002, +0.009]** * |
| precET | +0.026 [+0.000, +0.055] | **+0.023 [+0.005, +0.042]** * |
| macroP | +0.016 [+0.000, +0.033] | **+0.014 [+0.004, +0.026]** * |
| AUC | +0.007 [−0.008, +0.021] | **+0.022 [+0.014, +0.029]** * |
| ETsens | +0.026 [+0.000, +0.055] | **+0.029 [+0.012, +0.045]** * |

**PADS, n=304, 276 PD / 28 ET**

| method | precPD | precET | macroP | AUC | PPV@0.5 |
|---|---|---|---|---|---|
| logreg | 0.944 | 0.446 | 0.695 | **0.765** | 0.888 |
| one-class Mahalanobis | 0.940 | 0.411 | 0.675 | 0.717 | 0.873 |
| **rank-averaged hybrid** | **0.946** | **0.456** | **0.701** | 0.762 | **0.892** |

paired vs logreg: one-class alone is significantly **worse** on every column
(precET −0.036 [−0.052, −0.021], AUC −0.048 [−0.058, −0.038]). The hybrid is
+0.002 precPD * and +0.020 ETsens *, with precET +0.010 and macroP +0.006 both
spanning zero.

## Reading it

* **The hybrid is the only arm that wins on both cohorts.** In-house it is
  significant on five of six columns; on PADS it is level-or-better everywhere
  while one-class alone loses everywhere.
* **One-class alone is a small-data method, and behaves like one.** It beats
  logistic regression where the minority class is thinnest relative to what the
  discriminative model needs (21 ET in-house), and loses where there is enough
  signal to fit a boundary directly (PADS, where logreg already reaches AUC
  0.765). The crossover is the interesting part: it is the same ~28-patient
  boundary seen elsewhere in this project, read from the other side.
* **In-house, one-class alone has a lower CI bound of exactly +0.000 on precET.**
  On its own that is not a result. The hybrid's +0.023 [+0.005, +0.042] is.

## Why rank-averaging rather than probability-averaging

The one-class score is a Mahalanobis distance, not a probability. Averaging it
with `predict_proba` output puts an unbounded quantity on the same scale as one
bounded to [0, 1], and the distance's scale varies with fold. Converting both to
within-fold ranks first removes the scale question entirely:

```python
s[te] = 0.5 * (rankdata(lp) / len(lp) + rankdata(oc) / len(oc))
```

This also means the hybrid is monotone-invariant in each input, so it cannot be
helped or hurt by either model's calibration.

## Where this sits against the feature-union rule

The repo's standing rule is **prefer replacing a feature family over appending
one** — eight feature unions have underperformed their best member. This is the
second union that does not, and both share a property: the things combined are
not two views of the same information.

| union | outcome |
|---|---|
| axes + stability | wins — spatial shape and temporal shape are independent |
| logreg + one-class | wins — a boundary and a density are different objects |
| the other eight | dilute |

The distinction is not feature-count. `descriptors + stability` (0.754 vs 0.807)
adds ten more numbers describing the same spectrum. The two that work combine
estimators whose errors are not the same errors. That is a usable rule going
forward: **combine at the score level when the models differ in kind, at the
feature level almost never.**

## Limits

* Both gains are small in absolute terms. In-house precET moves 0.302 → 0.325 at
  a prevalence of 0.176 — a lift of 1.84× over prevalence, up from 1.71×.
  This does not change the ET
  ceiling documented in `precision_ceiling.md`.
* The covariance estimate warns "matrix not full rank" on in-house folds
  (98 PD patients, more features than is comfortable for a robust covariance).
  `MinCovDet` regularises through its support fraction, but the estimate is at
  the edge of what the sample supports, and that is part of why the one-class
  arm is noisier than the hybrid.
* PADS precET 0.456 and in-house 0.325 are **not comparable to each other** —
  prevalence is 0.092 and 0.176 respectively.
