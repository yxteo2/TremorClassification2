# The headline merged result survives patient-level uncertainty

**Question this answers.** Every paired interval in this project bootstraps over
**splits** on a fixed set of 404 patients. That answers "is A better than B on
these patients". A paper claims something else — that A beats B on patients not
yet seen — and for that the sampling unit is the patient. Given that a
patient-level bootstrap had just produced three false verdicts elsewhere
(`permutation_null.md`), the headline comparison needed checking rather than
assuming.

Run: `python -m experiments.patient_level_ci`. Both arms run on the same 20
splits, keeping each split's **per-patient test predictions**. Then 404 patients
are drawn with replacement; for every split both arms are re-scored using only
the drawn patients falling in that split's test fold, carrying multiplicity;
the results are averaged over splits and differenced. Both arms see the identical
patient draw and identical folds, so the comparison stays paired and only the
patient sample varies.

## Result

| arm | precN | precPD | precET | macroP |
|---|---|---|---|---|
| welch + desc + asym (baseline) | 0.639 | 0.636 | 0.583 | 0.619 |
| multitaper + trajectory | 0.639 | 0.655 | **0.685** | **0.660** |

| | diff | split-level 95 % | patient-level 95 % | width ratio |
|---|---|---|---|---|
| precN | +0.001 | [−0.029, +0.029] | [−0.019, +0.020] | 0.7 |
| precPD | +0.019 | [−0.005, +0.044] | [−0.006, +0.045] | 1.1 |
| **precET** | **+0.102** | [+0.031, +0.175] * | **[+0.005, +0.163] *** | 1.1 |
| **macroP** | **+0.041** | [+0.014, +0.067] * | **[+0.005, +0.066] *** | 1.1 |

**Both significant results hold.** macro precision +0.041 [+0.005, +0.066] and ET
precision +0.102 [+0.005, +0.163] with the patient as the sampling unit.

## Why the two intervals nearly agree

The patient-level interval is only ~1.1× wider. That is not a coincidence of this
dataset — it follows from the design. Each split tests 20 % of patients, so over
20 splits every patient is tested about four times under four different fold
compositions. The split-level bootstrap is therefore already integrating over a
great deal of patient-composition variability; it is not the naive
"same-test-set-every-time" quantity it might look like.

This is worth recording as a positive methodological result: **for the merged
20-split protocol, the split-level bootstrap this repo has always used is a good
approximation to the patient-level one.** The convention was sound. The lesson
from `permutation_null.md` — that a patient bootstrap can badly mislead — applies
to the *cross-validation single-model* setting, not to this paired one, and the
two findings do not conflict.

The width ratio of 0.7 on precN is not meaningful: the difference there is +0.001,
so both intervals are describing the same nothing.

## What neither interval covers

Both bootstraps hold the **fitted models** fixed and resample only what they are
scored on. Neither captures the variance of having trained on a different sample
of patients. That term is real and is why `permutation_null.md` — which refits on
every replicate — is the right instrument for single-model claims like "family X
separates PD from ET".

For a *paired* comparison the omission matters much less, because both arms are
fitted on the identical training patients in every split and the training-sample
effect largely cancels in the difference. It does not cancel exactly, so the
intervals above should be read as slightly optimistic, not exact.

## Standing

The merged two-stream result — multitaper spectrum + IF trajectory, soft-voted
with a residual TCN — is the one headline number in this project that has now
been checked at the patient level and held. It remains a **mixed-cohort** result:
it says "better at sites we trained on", and `pd_vs_et_transfer.md` shows the
PD-vs-ET axis does not transfer across sites in either direction.
