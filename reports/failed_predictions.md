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
| 13 | the frequency-contested gradient cannot be class confusion, because confusion produces opposing signs | `contested_profile.md` | **refuted by re-reading my own numbers** — that holds only when the class means straddle the range. They are monotonically ordered (8.16 / 7.51 / 7.04 Hz), so confusion gives the same sign for N and PD and none for ET, and the measured effect sizes (−0.385 / −0.241 / −0.051) fall monotonically with each class's own mean frequency. |
| 14 | cohort ID should cut NewData's contested rate more than 2015's, absorbing a domain shift | `cohort_id_input.md` | failed — contested rate moved −0.002 / −0.009 / −0.007 across cohorts, no differential effect; the precN gain it did produce is unexplained by this mechanism |
| 15 | model performance should rise monotonically as the spectral estimator gets smoother | `estimator_smoothing.md` | failed — the relationship is an inverted U peaking at the *current* nw 2.5. Sharper loses (ar16 macroP −0.031 *) and smoother loses (nw6 −0.016); there is no free gain either way. |
| 16 | trimming the PADS arm-raising onset should raise PADS-to-in-house PD-vs-ET transfer, because the onset is a PADS-only class-ordered signature the in-house cohorts lack | `pads_onset_trim.md` | failed — transfer AUC 0.578 → 0.563, −0.015 [−0.042, +0.009]; every arm below the 0.655 floor. The onset is real (mechanism held) but is second-order on a cohort gap far larger than it. |
| 17 | the old IF trajectory's significant gain lived in its transient end points reading the class-ordered PADS onset (point 0 sits on the onset) | `descriptor_trajectory_fix.md` | failed — the pre-fix point-0 magnitude is ~1.0 Hz for every class on PADS (N 1.01 / PD 0.89 / ET 1.01) and correlates with the onset ratio at +0.032. Class-agnostic noise. |
| 18 | training-time logit adjustment would lean NEGATIVE on precET, because prior_objective measured precET −0.236 when this project's imbalance correction was aimed at balanced accuracy and logit adjustment is Fisher-consistent for that objective | `logit_adjustment.md` | **wrong in direction, right in magnitude** — τ = 0.5 came out +0.034 precET / +0.012 macroP at 40 splits, positive but null. The flaw: post-hoc offset tuning and training-time adjustment share an objective but not a failure mode; the offset search moves a threshold on a fixed representation, adjustment shapes the representation. The sub-prediction τ = 0.5 > τ = 1.0 held. |
| 19 | MiniRocket, being unlearned like catch22, should beat the learned TCN on the same waveform | `rocket_waveform.md` | **refuted** — macroP 0.555/0.558 against the TCN's 0.626, and significantly worse than the reported model (−0.088 *, −0.085 *). It broke the standing rule it was built on: "unlearned" is the wrong abstraction; the repaired rule is "few features, selected for classification". |

Prediction 5 was the most sobering of the early batch because it was the
disciplined kind — built from measured sub-component gains rather than a story —
and it still failed. **Sub-component gains on this dataset are not evidence about
the composite task.**

Prediction 13 is the second wrong *explanation* in this register, after #9, and
both were caught the same way: by checking whether a competing account predicted
the numbers **including their magnitudes**, not just their signs. #9 was caught
by measuring ensemble disagreement; #13 by noticing that the three effect sizes
were ordered exactly as the rival account required. A sign test is weak evidence
when a magnitude ordering is available.

Prediction 12 is worth separating from the rest. It failed, but the experiment
was built so that its failure *eliminated* one of two named accounts rather than
just returning nothing — the rival account (cycle count) is untouched by a band
change, so the null moved it from one-of-two to the live explanation. A
prediction designed so that either outcome is informative is worth more than one
that only pays out when it holds.

Prediction 18 adds a distinct failure mode to the two above: **sharing an
objective does not make two methods share a failure mode.** Post-hoc offset
tuning and training-time logit adjustment both target balanced error, and the
first had failed badly here, so the second was predicted to fail too. It did not
— it came out positive, just not significantly. The methods differ in what they
can overfit: the offset search sees ~11 validation ET patients, adjustment sees
none.

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
| C | fixing the multitaper frequency axis would change little or nothing, because the distortion was uniform across every recording and the model was fitted and evaluated on it consistently | `axis_fix_audit.md` | held — macroP −0.008 [−0.029, +0.010], nothing significant on any column |
| D | peak-aligned averaging should beat the random-shift control on precET if the misalignment mechanism is real (deliberately NOT a prediction of improvement over the plain mean) | `peak_aligned_average.md` | held — +0.119 [+0.018, +0.226] * — while adoption was null (+0.007 precET vs the plain mean). A narrow prediction that held and still produced no gain. |
| E | trimming the PADS onset collapses the N > PD > ET first-1.5 s excess and the length-matched trim-end control does not | `pads_onset_trim.md` | held — 1.39/1.33/1.06 → 1.10/1.04/0.96; control 1.40/1.33/1.05 |
| F | the mixed-cohort headline effect of trimming the onset is small with uncertain sign | `pads_onset_trim.md` | held — macroP −0.006 [−0.031, +0.022], null on every column |
| G | the 0.25 s guard on the IF trajectory would be null, because the corrupted points are the same two positions for every patient and their magnitude does not depend on class | `descriptor_trajectory_fix.md` | held — macroP −0.004 [−0.027, +0.021], precET −0.004 [−0.068, +0.071] |
| H | the contiguous Q-factor fix would be small with uncertain sign, because the mislabelled feature nevertheless carried class-correlated information | `descriptor_trajectory_fix.md` | held — macroP +0.012 [−0.003, +0.034], precET +0.036 [−0.012, +0.097]; trended positive rather than the leaned-toward negative, which was explicitly not the claim |
| I | MiniRocket's failure is dimensionality, so reducing 9 996 features to ~22 should help substantially, with the optimum nearer 22–64 than 9 996 | `rocket_waveform.md` | held — PCA 22 gives macroP +0.046 [+0.013, +0.076] * and precET +0.086 [+0.006, +0.155] *, optimum exactly at 22. It does not rescue the method (0.605 vs the reported 0.643), so dimensionality is necessary but not sufficient. |

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
