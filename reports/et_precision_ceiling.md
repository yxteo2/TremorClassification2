# Why 90 % ET precision is not reachable

**Maximum ET precision at ANY threshold, from pooled out-of-fold probabilities
over 10 repeats:**

| cohort | ET | prevalence | AUC | avg precision | **max ET precision** |
|---|---|---|---|---|---|
| PADS | 28 | 0.092 | 0.761 | 0.346 | **0.615** |
| merged | 49 | 0.116 | 0.721 | 0.289 | **0.538** |
| in-house | 21 | 0.176 | 0.637 | 0.270 | **0.344** |

Targets of 0.90, 0.80 and 0.70 are **not reachable on any cohort at any
operating point** -- not merely unmet at the default threshold, but above the
maximum the model produces when flagging only its most confident patient.

## The arithmetic

ET is 9.2 % of the PADS tremor cohort (28 of 304). For 90 % precision you need
9 true ET for every 1 false positive. Flagging 10 ET patients allows **exactly
1 error among 276 PD** -- a false-positive rate of 0.0036 at a true-positive
rate of 0.36. That corner of the ROC requires **AUC around 0.98**. The best
model here reaches 0.76.

This is not an incremental shortfall that better tuning closes.

## What tightening the threshold does buy (PADS)

| target ET precision | achievable recall | ET found | PD false positives |
|---|---|---|---|
| 0.60 | 0.286 | 8 of 28 | 5 |
| 0.50 | 0.357 | 10 of 28 | 10 |
| 0.40 | 0.536 | 15 of 28 | 22 |

## 90 % IS met on two of the three clinical questions

| question | precision | status |
|---|---|---|
| N vs Tremor | 0.910 (2015) / 0.924 (PADS) | **met** |
| PD, within PD-vs-ET | 0.945 (PADS) / 0.929 (merged) | **met** |
| ET, within PD-vs-ET | 0.401, max 0.615 | not reachable |

ET precision is uniquely hard because **ET is the rare class**. Precision on a
rare class demands near-perfect separation; PD precision is high partly because
PD is the majority (276 of 304 on PADS) and must be read against that
prevalence.

## Three routes, in order of honesty

**1. Reconsider the target.** For a tool that flags patients for specialist
review, high ET *recall* at tolerable precision is the right objective --
catching 15 of 28 ET at 0.40 precision, with a neurologist confirming, is
clinically usable. High precision is what a *confirmatory* device needs, and a
wrist sensor alone is unlikely to be one.

**2. More ET patients.** Every method boundary measured in this project sits at
exactly this constraint: tree ensembles, SVMSMOTE and deep models all help on
the larger cohorts and fail in-house at 21 ET.

**3. Multimodal input.** The published 91 % on PADS
(`pads_literature.md`) uses IMU **plus a 30-item non-motor symptom
questionnaire**. Non-motor symptoms separate PD from ET on grounds no wrist
recording can reach. This is the one lever the state of the art uses that this
pipeline does not.

Reproduce: `scratch/pr_curve.py`.
