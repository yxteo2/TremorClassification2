# Do more sensors help at REST? No — and that is useful

Every REST result so far used `lower_arm` alone, chosen so the features would
also be computable on PADS (a single wrist unit). That looked like a compromise
forced by cross-dataset comparability. It is not.

2015 REST, PD-vs-ET, patient-level LOSO, paired bootstrap against `lower_arm`:

| sensors | n_feat | C | bal-acc | AUC | ET-F1 [95% CI] | vs lower_arm |
|---|---|---|---|---|---|---|
| **lower_arm (current)** | 10 | 1.0 | **0.730** | 0.729 | **0.500 [0.31, 0.67]** | — |
| hand | 10 | 1.0 | 0.650 | 0.708 | 0.393 | −0.080 [−0.24, +0.08] |
| upper_arm | 10 | 1.0 | 0.686 | **0.791** | 0.444 | −0.045 [−0.20, +0.11] |
| hand + lower | 20 | 1.0 | 0.623 | 0.702 | 0.372 | **−0.107 [−0.21, −0.01]** |
| hand + lower | 20 | 0.1 | 0.653 | 0.708 | 0.400 | −0.079 [−0.20, +0.03] |
| lower + upper | 20 | 1.0 | 0.693 | 0.772 | 0.455 | −0.038 [−0.19, +0.11] |
| lower + upper | 20 | 0.1 | 0.697 | 0.759 | 0.449 | −0.033 [−0.17, +0.10] |
| ALL THREE | 30 | 1.0 | 0.661 | 0.722 | 0.419 | −0.070 [−0.20, +0.06] |
| ALL THREE | 30 | 0.1 | 0.679 | 0.735 | 0.435 | −0.050 [−0.19, +0.10] |

(stft512; the welch sweep gives the same ordering.)

**Nothing beats `lower_arm` alone.** Every combination is worse, and
`hand + lower` at C=1.0 is *significantly* worse — its paired CI excludes zero.
Regularisation (C=0.1) recovers some of the multi-sensor loss but never reaches
the single sensor.

## Two things this settles

1. **The single-sensor restriction costs nothing.** It was adopted for PADS
   comparability; it turns out to be the optimum for the 2015 cohort at REST
   too. The best local model and the cross-dataset-comparable model are the
   same model, so no result is being sacrificed for comparability.
2. **This extends `reports/sensor_selection.md` to REST.** That report found
   `lower_arm` best at OUT; the same holds at REST, now with paired CIs. Adding
   channels dilutes rather than adds — 30 features against 16 ET subjects is
   straightforwardly too many.

## One lead worth following

**`upper_arm` has the highest AUC of any single sensor (0.791 vs lower_arm's
0.729) while scoring lower on balanced accuracy (0.686 vs 0.730).** AUC is
threshold-free, so this means upper_arm *ranks* patients better but the 0.5
decision threshold is badly placed for it. A tuned threshold — selected inside
the CV, never on the test fold — could plausibly make upper_arm the better
sensor at REST.

That is worth testing, with the caveat that threshold tuning on 16 ET subjects
is exactly the kind of small-n operation that has produced retracted results
earlier in this project. It needs the in-CV protocol already implemented in
`pdetn.model.TwoStageClassifier`, not a post-hoc sweep.
