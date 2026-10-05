# Patient-level uncertainty: which 2015 gains would hold for new patients?

Run: `python -m experiments.patient_bootstrap_2015` (~1 h CPU, saved predictions).

## Why

An audit of this session's experiment code (2026-10-05) found that every paired
CI resamples the 40 per-repeat differences. All repeats score the same 151
rows, so those CIs measure re-partitioning and seed noise -- not which patients
were sampled, the dominant uncertainty with 15 ET. "Confirmed on fresh
partitions" (seeds 100-139) is the same patients again: it shows a gain is not
specific to one partition, not that it generalises.

Here patients are resampled (stratified by class, 2000 draws); each repeat's
out-of-fold predictions are scored on the draw and averaged over repeats. The
two pre-fix PD 12 rows (one person, see below) are resampled as one patient.

## Result

| claim | point | **patient-level 95 %** | verdict |
|---|---|---|---|
| PADS transfer (`ft` − scratch): precET | +0.092 | **[+0.012, +0.211]** | **holds** |
| PADS transfer: PD-vs-ET AUC | +0.053 | [−0.022, +0.138] | n.s. |
| PADS transfer: precPD | −0.037 | [−0.074, +0.002] | n.s. |
| coupling in the network (`ft_seg` − `ft`): AUC | +0.015 | [−0.008, +0.041] | n.s. |
| REST w 0.25 (`fuse_w25` − `ft`): precN / AUC | +0.012 / +0.021 | [−0.009, +0.034] / [−0.006, +0.052] | n.s. |
| same, fresh partitions: precPD / AUC | +0.018 / +0.024 | [−0.002, +0.037] / [−0.004, +0.053] | n.s. |
| REST w 0.5 (`fuse` − `ft`): precPD / precET | +0.036 / −0.062 | [−0.008, +0.079] / [−0.161, +0.039] | n.s. |
| **WING** (`+ WING` − base): precPD | +0.032 | **[+0.007, +0.058]** | **holds** |
| WING, fresh partitions: precPD | +0.027 | **[+0.001, +0.053]** | **holds** |
| **NewData pooled** (`nd_pool` − `ft`): AUC | +0.044 | **[+0.007, +0.087]** | **holds** |
| NewData pooled, fresh partitions: AUC | +0.044 | **[+0.010, +0.083]** | **holds** |

Not bootstrapped here: the coupling second stage and the stacked rankers (their
scores are not stored in one file per arm), and the TCN-epoch-20 checkpoint rule
(`checkpoint_rule_2015.py`), whose epoch was read off test curves covering every
patient -- its gain (AUC +0.017 \*, top-5 ET +0.090 \* on fresh partitions) is
**biased upward** and unconfirmed until the epoch is chosen inside the training
folds.

## Reading

* **Three claims survive patient resampling**: PADS transfer's ET precision
  gain, WING's PD precision gain, and NewData's PD-vs-ET ranking gain.
* **Gains of +0.01-0.03 are below what 150 patients / 15 ET can resolve.**
  REST fusion, coupling, and transfer's AUC gain are consistent across
  partitions but not distinguishable from patient-sampling luck.
* Earlier "\*" in this session's reports mean "robust to re-partitioning",
  not "generalises".

## The PD 12 duplicate (fixed in `common/quaternion_data.py`)

`PD 12_OUT N LOAD.txt` was loaded as a second subject next to `PD 12_OUT`: one
person as two patients, landing in different CV folds (train and test) in 34 of
40 repeats. Also `PD 1_out` / `PD 23_out` (lower case) failed case-sensitive
cross-task joins. Subject ids are now canonical (`<class> <n>_<TASK>`); 2015 OUT
is **150 patients (61 N / 74 PD / 15 ET)**. Every saved 2015 run (this session's
and `transfer_2015`'s) used the 151-row table; the effect is 1 of 151 rows and
has not been re-run.
