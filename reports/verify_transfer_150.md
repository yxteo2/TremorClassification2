# PADS transfer on the corrected 150-patient table: consistent, but not established for new patients

Run: `REPS=0-40 python -m experiments.verify_transfer_150` then `... report`
(arms `scratch` and `ft` only; `transfer_2015.fit` unchanged; written by a
verification agent, report completed by the coordinator after the agent hit an
API limit -- the run itself finished).

The 2015 subject-id fix (`common/quaternion_data.py`) merged a duplicated PD 12
and corrected two lower-case ids: **150 patients, 61 N / 74 PD / 15 ET**. Every
earlier 2015 run used the 151-row table.

| arm (40 repeats) | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| scratch | 0.705 | 0.755 | 0.242 | 0.567 | 0.554 | 0.560 |
| **ft** | 0.727 | 0.719 | **0.321** | 0.589 | 0.580 | 0.618 |

| ft − scratch | repeat level | **patient level (2000 draws)** |
|---|---|---|
| precET | +0.079 [+0.031, +0.122] \* | **[−0.028, +0.202]** |
| PD-vs-ET AUC | +0.058 [+0.043, +0.073] \* | [−0.023, +0.148] |
| precPD | −0.036 [−0.051, −0.022] \* | [−0.078, +0.007] |
| macroP | +0.022 [+0.003, +0.039] \* | [−0.027, +0.074] |

On the 151-row table: precET 0.249 -> 0.350 (+0.092), patient level
[+0.012, +0.211] -- just clear of 0.

## Verdict

* The transfer gain **reproduces in direction and size** on the corrected table
  (precET +0.079, AUC +0.058, robust to re-partitioning, wins 75 % / 88 % of
  repeats).
* It is **no longer distinguishable from patient-sampling luck**: removing one
  duplicated patient moved the patient-level interval from just above 0 to just
  across it. With 15 ET, no 2015-only claim about ET precision can currently be
  shown to generalise; the honest statement is "consistent +0.08 ET precision,
  95 % patient-level interval [−0.03, +0.20]".
* `ft` remains the recommended model (largest and most consistent effect
  measured), but the paper should report the patient-level interval and the
  cohort-size limitation.
