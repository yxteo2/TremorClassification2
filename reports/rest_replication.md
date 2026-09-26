# The 2015 REST signal does not replicate in-house, and reverses in PADS

Run: `python -m experiments.rest_replication` (minutes; logistic regression only)

## Question

`01_tremor_characteristics.ipynb` found PD vs ET separable at REST on 2015 (16
ET), PD slower than ET. For a paper, one cohort at 16 ET is a lead. Is it a
property of in-house REST, or of 2015?

Six frequency characteristics, **lower-arm sensor** (the pipeline's, fixed before
looking; the hand sensor scored best in the notebook and was not chosen).
Cross-validated tests use 5-fold x 20 repeats against a 500-permutation null
built from the same statistic; frozen tests fit on all of 2015 REST and apply
the model unchanged, with an exact one-sided Mann-Whitney p.

**Not blind:** NewData REST and PADS Relaxed univariate medians had already been
printed in the audit (`inhouse_pd_vs_et.md`). The multivariate transfer (C, D)
had not.

## Result

| test | n | ET | AUC | p |
|---|---|---|---|---|
| A  2015 REST, CV | 91 | 16 | 0.659 | 0.038 (null [0.294, 0.668]) |
| B  in-house pooled, z-scored within cohort, CV | 122 | 22 | 0.573 | 0.162 (null [0.334, 0.644]) |
| C  2015 model -> NewData REST | 31 | 6 | 0.533 | 0.41 |
| D  2015 model -> PADS Relaxed | 304 | 28 | **0.280** | **< 0.001, reversed** |
| E  max_freq "ET higher" on 2015 | 91 | 16 | 0.636 | 0.045 |
| E  same rule on NewData REST | 31 | 6 | 0.573 | 0.30 |
| E  same rule on PADS Relaxed | 304 | 28 | **0.300** | **< 0.001, reversed** |

## Reading

* **The in-house REST signal is a 2015 result, and a marginal one.** A reaches
  p = 0.038 on the pre-chosen sensor, and three sensors were looked at in the
  notebook, so it does not survive a multiplicity correction. Pooling with
  NewData takes it inside the null (B). NewData leans the same way (C 0.533,
  E 0.573) but 6 ET cannot confirm or refute an effect this size.
* **The reversal in PADS is robust.** A rule learned on 2015 -- by a fitted model
  or by one threshold on one feature -- ranks PADS ET *below* PADS PD at
  AUC 0.28-0.30, p < 0.001, n = 304.

So the defensible statement for the paper is not "in-house REST separates PD
from ET" but:

> **The frequency relationship between PD and ET tremor is cohort-dependent.**
> In PADS, ET tremor is slower than PD at rest and in posture; in the in-house
> cohorts it is equal at posture and, in one cohort, faster at rest. A
> frequency rule learned in one population is reversed in the other.

That is consistent with every transfer failure on record (PADS -> in-house 0.472
to 0.578) and explains them without appeal to the model. Clinically plausible
drivers (ET frequency falls with age and severity; PD patients without visible
tremor contribute a physiological ~6-8 Hz peak) cannot be tested without age,
disease duration and severity, which the in-house cohorts do not record.

## Predictions (recorded in the docstring)

* C above 0.5 and not significant -- **held**.
* D significantly below 0.5 -- **held**.
* B stays above its null -- **failed** (p = 0.162): NewData's 6 ET diluted
  rather than added.
