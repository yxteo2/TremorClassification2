# 2015-first classification exploration

## Data and scope

No PADS data enters any experiment. The 2015 repository contains OUT, REST and
WING, but not DRINK. Available unique patients per task (N/PD/ET):

| Task | Patients | N | PD | ET | Recordings |
|---|---:|---:|---:|---:|---:|
| OUT | 151 | 61 | 75 | 15 | 274 |
| REST | 152 | 61 | 75 | 16 | 275 |
| WING | 137 | 61 | 63 | 13 | 250 |

Requiring OUT and REST leaves 148 patients (61/72/15); seven of the 155 unique
patients across the three tasks are excluded for missing one of those tasks.
WING is audited but not included in the matched comparison. Recording durations
are 9.93–41.18 seconds after quaternion conversion.

NewData contributes 53 patients with both tasks (25/22/6), for a pooled 201
(86/94/21). Its loader returns 56 OUT and 58 REST patients; eight of their union
lack one task. These counts describe usable loader outputs, not a complete
clinical enrollment inventory: the inherited NewData loader can skip recordings
with missing sensors, short recordings or conversion errors.

## Methods

Use the existing quaternion-to-angular-velocity conversion at 100 Hz. Preserve
three sensors separately. For each sensor compute six fixed features: log power
in 3–5, 5–7, 7–10 and 10–15 Hz bands, peak frequency and spectral entropy. The
Welch implementation is shared with task_power_benchmark. Average repeated
recording features within patient/task, never split recordings into independent
patients. 2015 action suffixes are removed by the multi-action loader.

Three representations: middle-sensor OUT (6 features), all-sensor OUT (18),
all-sensor OUT+REST (36). Inner CV selects the representation and either balanced
logistic regression (C=.01,.1,1) or balanced RBF SVM (C=.1,1,10, gamma=scale).
Scaling is fitted inside each fold. Five outer patient folds and three inner
folds use seed 0. Decisions use the fitted classifier directly; no test tuning
or offset tuning. Primary comparator is fixed C=.1 middle-sensor OUT logistic.
This is a new simple control, not the historical deep network or MATLAB model.

Two additional analyses: pooled nested CV, reported by cohort; and train on
2015 alone, select only within 2015, then evaluate NewData. A paired augmentation
analysis keeps the 2015 test folds fixed and adds all usable NewData only to
training; its inner validation also contains only 2015 patients.

NewData retains its existing label-blind ten-second tremor epoch selection,
whereas 2015 uses full recordings. This preprocessing difference, hardware,
cohort composition and selection of complete cases limit transportability.
Cohort-qualified patient IDs avoid accidental numeric-ID collisions; absence
of clinical participant overlap between years is assumed, not verified from a
cross-cohort identity register.

## Evidence behind the choices

[Teo et al., Computers in Biology and Medicine 2024](https://pubmed.ncbi.nlm.nih.gov/39098236/)
reports that resting-to-lifting transitions during drinking were important to
its classifier. Those DRINK transitions are not available in the 2015 folder,
so this experiment cannot reproduce that paper's input or accuracy.

[Di Biase et al., Brain 2017](https://pubmed.ncbi.nlm.nih.gov/28459950/)
provides evidence for temporal tremor characteristics, including independent
validation (AUC .855). This motivates a future temporal-feature comparison;
the current screen measures spectral power and shape only, not the published
Tremor Stability Index. No accuracy from either paper is transferred to this
cohort as an expected result.

## Reproduce

Requires the task-power benchmark from PR #4 and existing scientific Python
and h5py dependencies. Finish retrieving Data/raw_quaternion and NewData first.

```bash
python -m unittest discover -s tests -p test_inhouse_2015_explore.py -v
python -m experiments.inhouse_2015_explore --include-newdata --output artifacts/inhouse_2015_pooled_complete
```

Five unit tests cover repeated-trial pooling, missing tasks, label conflicts,
empty inputs and sensor-column selection. No patient-level export is committed.
All results are exploratory on previously investigated data. Paired bootstrap
intervals use 2000 fixed-prediction patient resamples and exclude retraining,
historical selection and site uncertainty. A run initially reached NewData
before its download completed and stopped; the complete rerun reproduced the
2015 predictions. No settings were chosen from that interrupted run's scores.

## Results and practical decision

The following rows score exactly the same 148 held-out 2015 patients:

| Training and model | Accuracy | Macro-F1 | ET precision | ET recall |
|---|---:|---:|---:|---:|
| 2015, fixed middle-OUT logistic | .608 | .523 | .161 | .333 (5/15) |
| 2015, nested representation/model selection | .635 | .563 | .241 | .467 (7/15) |
| 2015 + NewData training, nested selection targeting 2015 | .703 | .629 | .276 | .533 (8/15) |

2015-only nested selection minus fixed control: macro-F1 +.040,
paired 95% interval [-.032,+.113]. All five folds selected logistic regression;
three selected both tasks and two selected all-sensor OUT.

**Adding NewData to training on the same 2015 test folds:** macro-F1 +.066,
paired 95% interval **[+.007,+.132]** versus 2015-only nested selection. All
five folds selected all-sensor OUT+REST; four selected logistic C=.1 and one
RBF SVM C=1. PD recall rose from .500 to .625 and PD precision from .783 to .849.
This is the most promising exploratory result here. Its interval conditions on
fixed predictions and does not account for this follow-up being added after
initial exploration. ET precision remains only .276 (8 true positives among
29 ET predictions), so this is not a clinically sufficient diagnosis model.

Separate pooled nested CV on 201 patients: macro-F1 .559, ET precision .241,
ET recall .333. Its fixed-control macro-F1 is .537; difference interval
[-.044,+.091]. Within the pooled experiment, 2015 macro-F1 is .588 and NewData
.481. These are different outer splits/populations and cannot quantify the
causal benefit of adding NewData; the matched augmentation comparison above
addresses that question directly.

Train on all 2015 and test the 53 NewData patients without fitting to their
labels: macro-F1 .571, accuracy .604, ET precision .429 and recall .500 (3/6).
This is a historical cohort-transfer stress test, not a fresh prospective
external validation; six ET patients give very weak precision on the estimate.

Recommended next candidate: keep all three sensors and separate OUT/REST
features, use regularized logistic as the simple implementation candidate, and
retain inner-fold selection when comparing against SVM. The .629 figure belongs
to the nested selection procedure, not to one fixed final logistic model.
A temporal-feature ablation (frequency variability and envelope modulation)
should be evaluated within the same nested protocol before adding network size.
Do not add WING via complete-case filtering without reporting its exclusions.

Aggregate confusion matrices, selections and intervals are stored in
[inhouse_2015_summary.json](inhouse_2015_summary.json). The 2015 nested predictions
reproduced exactly across the interrupted, standalone and pooled runs. The
production model and historical reports remain unchanged.
