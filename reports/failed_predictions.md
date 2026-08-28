# Register of predictions made before the run, and how they turned out

This project writes the prediction into the experiment's docstring *before*
launching it, so that a null cannot be reinterpreted afterwards as the expected
outcome. This file is the register. It exists because the count was previously
asserted in the README as a bare number that nothing on disk could check.

The pattern it documents is the useful part: **mechanism stories about why a
method ought to work have a poor record here, and predictions derived from
measurements of this dataset have a better one.**

## Failed

| # | prediction | where | what happened |
|---|---|---|---|
| 1 | robust estimation should help | `tf_window_length.md` | failed |
| 2 | demodulation cost explains the gap | `tf_window_length.md` | failed |
| 3 | median-centring should help | `tf_window_length.md` | failed |
| 4 | N-vs-Tremor blurring explains the gap | `tf_window_length.md` | failed |
| 5 | measured sub-component gains should compose in two stages | `tf_window_length.md` | failed |
| 6 | the hardest majority patients are mislabelled noise; dropping them sharpens the boundary | `prune_training.md` | **inverted** — significantly worse than both keeping them and dropping random ones. They were boundary-defining. |
| 7 | subjects measurably harmful in training can be identified and removed | `influence_prune.md` | failed — no better than random, and trending worse, with an unstable ranking |
| 8 | geometric pooling should buy ET precision by vetoing over-confident members | `pooling_rules.md` | failed — reproduces arithmetic pooling to three decimals on macroP |
| 9 | the pooling null is explained by the six members being near-copies | `ensemble_diversity.md` | **refuted by measurement** — they disagree on 20.5 % of patient pairs |
| 10 | a dedicated ET detector beats ET's column in the 3-class softmax | `one_vs_rest.md` | **inverted** — precET −0.162 [−0.249, −0.073], and worse as a pure ranker (AUC 0.750 vs 0.770) |
| 11 | routing contested patients to a second model should help | `contested_gating.md` | failed — +0.001 against the fusion control that ignores the gate |
| 12 | slow patients are contested because their signal is entangled with drift at the 3 Hz band edge; extending the band down should help them specifically | `low_band_edge.md` | failed — slow-tercile contested rate rose slightly (+0.005 at 2 Hz, +0.019 at 1.5 Hz) and precision was null on every column |

Prediction 5 was the most sobering of the early batch because it was the
disciplined kind — built from measured sub-component gains rather than a story —
and it still failed. **Sub-component gains on this dataset are not evidence about
the composite task.**

Prediction 12 is worth separating from the rest. It failed, but the experiment
was built so that its failure *eliminated* one of two named accounts rather than
just returning nothing — the rival account (cycle count) is untouched by a band
change, so the null moved it from one-of-two to the live explanation. A
prediction designed so that either outcome is informative is worth more than one
that only pays out when it holds.

Predictions 6 and 10 share a shape worth naming: both identified something that
*looked* like a handicap for the minority class (hard patients dragging the
boundary; ET's logit diluted by two majority columns) and both turned out to be
**load-bearing**. A structure that looks wasteful at 404 patients with 49 ET is
more often doing work than not.

## Held

| # | prediction | where | what happened |
|---|---|---|---|
| A | thirteen feature families would not beat the best single one | `tcn_fusion.md` | held |
| B | balanced bagging would be null, because the members already disagree on 20.5 % of patients so data diversity is not the binding constraint | `balanced_bagging.md` | held — macroP +0.001 [−0.015, +0.019], and +0.007 against the matched seed control |

Prediction B is the contrast that makes the register worth keeping. It was
recorded in writing in two reports before the run finished, and unlike the failed
ones it was derived from a **measurement of this dataset** (the ensemble's actual
internal disagreement) rather than from an argument about why a method ought to
work.

## How to use this file

Append a row when an experiment's docstring commits to a direction before the
run. Do not edit a row after the fact. If a prediction is later shown to have
been right for the wrong reason, add a row rather than amending one — that is
what happened to #9, which was a wrong explanation of a correct null.
