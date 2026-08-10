# In-CV threshold tuning at REST — the lead fails, and tuning itself is harmful

Follow-up to `reports/sensor_combination_rest.md`, which noted that `upper_arm`
has the highest AUC of any single sensor (0.791) but the lowest balanced
accuracy of the good ones (0.686) — suggesting the 0.5 threshold was simply
misplaced for it.

Threshold selected by **inner 5-fold CV on the training split only**, never on
the held-out patient. 2015 REST, stft512, PD-vs-ET, paired bootstrap against
`lower_arm` @ 0.5.

| sensor | threshold | bal-acc | AUC | ET-F1 [95% CI] | vs lower_arm@0.5 |
|---|---|---|---|---|---|
| **lower_arm** | **0.5** | **0.730** | 0.729 | **0.500 [0.31, 0.67]** | — |
| lower_arm | in-CV | 0.624 | 0.729 | 0.367 | **−0.107 [−0.21, −0.01]** |
| upper_arm | 0.5 | 0.686 | **0.791** | 0.444 | −0.045 [−0.20, +0.11] |
| upper_arm | in-CV | 0.722 | 0.791 | 0.441 | −0.009 [−0.13, +0.12] |
| hand | 0.5 | 0.650 | 0.708 | 0.393 | −0.080 [−0.24, +0.09] |
| hand | in-CV | 0.704 | 0.708 | 0.431 | −0.026 [−0.17, +0.12] |

## The lead was half right

Tuning **does** help the sensors whose threshold was misplaced — `upper_arm`
0.686 → **0.722**, `hand` 0.650 → **0.704**. So the AUC/bal-acc gap really was a
threshold problem.

But **nothing overtakes `lower_arm` at the default 0.5** (0.730). Even tuned,
upper_arm lands at 0.722 with a paired CI straddling zero. The headline result
is unchanged.

## The finding that generalises: tuning HURTS the best configuration

In-CV threshold tuning makes `lower_arm` **significantly worse** — 0.730 →
0.624, paired CI **[−0.21, −0.01], excluding zero**. This is not noise.

The mechanism is small-n: with 16 ET subjects the inner-CV threshold estimate is
high-variance, and when 0.5 is already near-optimal a noisy estimate can only
move away from it. Tuning helps only where the default is genuinely badly
placed; where it is not, tuning is a pure variance cost.

**This matters beyond this table.** `pdetn.model.TwoStageClassifier` is used
throughout the project with `tune_et_threshold=True`, and
`pdetn.deep_eval`/`deep_crossdataset` tune an ET threshold too. On this cohort
size that default may be costing accuracy rather than buying it. Any
configuration relying on it should be re-run with tuning **off** as a control
before its number is reported.

`pdetn/deep_crossdataset.py::DeepTwoStage` already defaults `tune_et=False` with
a comment recording that tuning destabilised the deep PD-vs-ET model on tiny
validation ET sets — the same effect, observed independently. This makes it two
places, so it should be treated as a property of the cohort, not a quirk.

## Standing best result, unchanged

**2015 REST, lower_arm, stft512 descriptors, threshold 0.5 —
bal-acc 0.730, AUC 0.729, ET-F1 0.500 [0.30, 0.67]** on 75 PD vs 16 ET.
