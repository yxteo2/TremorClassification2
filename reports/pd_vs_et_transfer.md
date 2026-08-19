# PD vs ET does not transfer between cohorts, in either direction

**Question this answers.** Every PD-vs-ET result in this project is within-cohort
cross-validation. On PADS that works — five of six families beat a permutation
null, descriptors AUC 0.794, p = 0.005 (`permutation_null.md`). A clinic does not
ask that question. It asks whether a model fitted at one site works at another.

Run: `python -m experiments.loco_pd_et`. Fit on one cohort, predict the other,
never refit. Because the model is fixed and only the test draw varies, a
class-stratified patient bootstrap on the test set is the correct interval here —
unlike the cross-validation setting, where holding the fitted model fixed made the
same bootstrap anti-conservative.

## PADS → in-house (the useful direction)

PADS has 28 ET, the largest ET group available. If a PADS-fitted model worked
in-house, the in-house ET shortage would stop being the binding constraint.

| family | AUC | 95 % CI | precPD | precET | |
|---|---|---|---|---|---|
| descriptors | 0.519 | [0.389, 0.652] | 0.816 | 0.143 | no transfer |
| spectrum | 0.567 | [0.448, 0.679] | 0.837 | 0.238 | no transfer |
| stability | 0.526 | [0.391, 0.662] | 0.837 | 0.238 | no transfer |
| axes | 0.634 | [0.497, 0.758] | 0.857 | **0.333** | no transfer |
| harmonics | 0.494 | [0.353, 0.635] | 0.827 | 0.190 | no transfer |
| ampmod | 0.564 | [0.437, 0.683] | 0.816 | 0.143 | no transfer |
| ALL concat | **0.651** | [0.533, 0.765] | 0.847 | 0.286 | CI excludes 0.5 |

**Not one individual family transfers.** `descriptors`, the strongest model
within PADS at AUC 0.794, lands at 0.519 in-house — chance. Test prevalence is
0.176, so precET 0.143 is *below* prevalence and precET 0.333 (axes) is a 1.9×
lift.

## in-house → PADS (the control)

| family | AUC | 95 % CI | precPD | precET | |
|---|---|---|---|---|---|
| descriptors | 0.427 | [0.295, 0.563] | 0.913 | 0.143 | no transfer |
| spectrum | 0.504 | [0.360, 0.648] | 0.920 | 0.214 | no transfer |
| stability | 0.569 | [0.459, 0.679] | 0.909 | 0.107 | no transfer |
| axes | 0.451 | [0.338, 0.565] | 0.906 | 0.071 | no transfer |
| harmonics | 0.337 | [0.250, 0.429] | 0.902 | 0.036 | CI below 0.5 |
| ampmod | 0.639 | [0.523, 0.743] | 0.913 | 0.143 | CI excludes 0.5 |
| ALL concat | 0.558 | [0.445, 0.668] | 0.913 | 0.143 | no transfer |

As expected from a cohort where nothing clears its own permutation null, nothing
transfers out of it either.

## Reading the three marginal results

Three cells have intervals excluding 0.5: `ALL concat` PADS→in-house (lower bound
0.533), `ampmod` in-house→PADS (0.523), and `harmonics` in-house→PADS inverted
(upper bound 0.429). **There are 14 tests here.** At that multiplicity, lower
bounds of 0.523 and 0.533 are what one expects from noise, and none of the three
should be reported as a transfer result on its own.

The `ALL concat` cell is the one worth a second look, because it is *directionally*
interesting: concatenating all six families transfers (0.651) when no single
family does. That is the opposite of the within-cohort behaviour, where
concatenation is the worst option and dilutes badly (precET 0.316 vs 0.450 on
PADS — `score_vs_feature_fusion.md`). A plausible reading is that concatenation
lets the model average over site-specific quirks it cannot do anything about
individually. It is a hypothesis, not a finding, and would need its own test.

## What this settles

* **The PADS result cannot be presented as a PD-vs-ET method that works.** It is a
  result about PADS. AUC 0.794 within PADS becomes 0.519 on in-house patients.
* **This explains the earlier merged-training finding.** Adding PADS to in-house
  training significantly *degrades* in-house PD precision (−0.082,
  `own_data_reality_check.md`). A training set whose PD-vs-ET boundary does not
  hold in-house should be expected to hurt, and it does.
* **`axes` remains the least bad candidate.** It is the only family with no
  significant PADS-vs-in-house AUC difference (−0.096 [−0.280, +0.098]), the best
  in-house within-cohort family (p = 0.085), and the best single-family transfer
  (0.634, precET 0.333). None of that is significant. It is where to look next,
  not something to claim.

## Limits

* One training cohort per direction, so the intervals are conditional on that
  training set and do not include the variance of having recruited a different
  training cohort.
* Logistic regression only. A model that transfers where logreg does not is
  possible but unlikely given that no deep model has beaten logreg on this axis
  within-cohort either.
* NewData contributes 6 ET to the in-house side; the in-house arm is effectively
  2015 plus a small perturbation.
