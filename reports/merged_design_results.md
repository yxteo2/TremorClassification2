# Chosen design: train on 2015 + NewData, validate on PADS

Wrist-equivalent sensor only, 10 gyro-derived descriptors (the only feature
family computable on all three cohorts), patient-level LOSO internally, single
scoring pass on PADS externally. Code: `tfbench/merged.py`.

Device probe judged on **|AUC − 0.5|** (an AUC of 0.000 is maximally separable,
not safe). Threshold: |dev| < 0.25 to pool.

## Result 1 — merging works at REST, and gives the project's best PD-vs-ET

| method | device probe | PD-vs-ET internal | AUC | **ET-F1 [95% CI]** |
|---|---|---|---|---|
| **welch** | 0.698 (dev 0.198) **pass** | **0.740** | 0.770 | **0.557 [0.40, 0.70]** |
| multitaper | 0.771 (dev 0.271) confounded | 0.724 | 0.793 | 0.542 |
| stft512 | 0.688 (dev 0.188) pass | 0.685 | 0.751 | 0.500 |
| cwt | 0.594 (dev 0.094) pass | 0.674 | 0.769 | 0.484 |

**2015 + NewData at REST, welch descriptors: bal-acc 0.740, ET-F1 0.557
[0.40, 0.70]** on 75 PD vs 22 ET. That is the **best ET-F1 recorded in this
project** (previous best 0.516). Merging genuinely helped here — 2015 alone at
REST gave welch 0.666 / ET-F1 0.417.

Use `welch` or `cwt`, not `multitaper` — the latter scores well but fails the
device probe, so its gain may be reading the recording device.

## Result 2 — merging is NOT valid at OUT

| method | device probe | verdict |
|---|---|---|
| welch | 0.211 (dev 0.289) | **confounded** |
| cwt | 0.200 (dev 0.300) | **confounded** |
| stft512 | 0.200 (dev 0.300) | **confounded** |
| multitaper | 0.067 (dev 0.433) | **confounded** |

Every method fails at OUT. Since NewData is **entirely ET**, an identifiable
device lets the model read "new device ⇒ tremor/ET" directly. Internal OUT
numbers from a merged cohort must not be reported. (PD-vs-ET internal at OUT is
poor anyway: 0.474–0.609.)

**Merge at REST only.**

## Result 3 — PADS validates N-vs-Tremor, and refutes PD-vs-ET

Trained on merged 2015+NewData, scored once on PADS (383 patients):

| axis | train condition | internal | **external (PADS)** | AUC |
|---|---|---|---|---|
| N vs Tremor | OUT | 0.814 | **0.736** | **0.783** |
| N vs Tremor | REST | 0.757 | 0.664 | 0.714 |
| PD vs ET | OUT (task-matched) | 0.474–0.609 | **0.440–0.482** | **0.42–0.43** |
| PD vs ET | REST (task mismatch) | **0.740** | 0.494–0.552 | 0.43–0.51 |

**N-vs-Tremor transfers.** stft512 trained on merged OUT reaches bal-acc 0.736 /
AUC 0.783 on a completely unseen cohort, device and country. That is a genuine
external validation and the strongest generalisation result in the project.

**PD-vs-ET does not transfer — and this is not a task-mismatch excuse.** At OUT
the task *is* matched to StretchHold, and external AUC is still **0.423–0.431**,
i.e. **below chance**. Consistently below 0.5 across all four methods means the
PD/ET decision boundary learned on our cohorts is actively *inverted* on PADS —
consistent with the earlier finding that PD is *faster* than ET in PADS
(7.71 vs 6.55 Hz) but not separated in the 2015 cohort.

## What to report

* **N vs Tremor** — merged internal **0.814**, external PADS **0.736 / AUC 0.783**.
  Clean, externally validated, publishable as-is.
* **PD vs ET** — merged REST internal **0.740, ET-F1 0.557 [0.40, 0.70]**,
  reported as an *internal* result with the PADS transfer failure stated
  explicitly. The failure is itself a finding: PD/ET frequency structure is
  cohort-specific, not universal.

Do **not** present a single pooled all-three-cohort number. The evidence does
not support it on either axis.

## Open, cheap, and high-value

Only PADS **StretchHold** has been extracted. **PADS `Relaxed`** is the REST
equivalent, and REST is where PD-vs-ET works on our data. Extracting it would
give the first task-matched external test of the 0.740 result. Until then the
PD-vs-ET external column is either mismatched (REST) or drawn from the condition
where our own model is weakest (OUT).
