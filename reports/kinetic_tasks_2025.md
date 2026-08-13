# The kinetic tasks — PD-vs-ET works where rest and posture fail

The 2025 cohort records **7 tasks**, but only REST and OUT were ever loaded. The
other five were unusable while the cohort was ET-only: no class to discriminate
against. With 31 HC and 34 PD they became testable, and two of them change the
picture.

2025 cohort, lower_arm, stft512, 10 descriptors, patient-level LOSO:

| task | N/PD/ET | axis | bal-acc | AUC | **ET precision** | F1 |
|---|---|---|---|---|---|---|
| REST | 27/25/6 | PD-vs-ET | 0.487 | 0.273 | 0.182 | 0.235 |
| OUT | 27/23/6 | PD-vs-ET | 0.388 | 0.196 | 0.100 | 0.125 |
| **DRINK** | 29/23/6 | **PD-vs-ET** | **0.790** | **0.812** | **0.667** | **0.667** |
| **FINGER_NOSE** | 29/24/6 | **PD-vs-ET** | 0.708 | **0.826** | 0.400 | 0.500 |
| POUR | 27/27/6 | PD-vs-ET | 0.500 | 0.525 | 0.182 | 0.235 |
| TAP | 25/26/6 | PD-vs-ET | 0.417 | 0.474 | 0.133 | 0.190 |
| PRON_SUP | 29/27/6 | PD-vs-ET | 0.565 | 0.438 | 0.231 | 0.316 |

**ET precision 0.667 on DRINK.** Every previous best was 0.393 (2015 REST,
binary) or 0.219 (3-class). **AUC 0.826 on FINGER_NOSE** is the highest anywhere
in this project — the 2015 best is 0.729 and PADS's is 0.775.

## Why this is more than a lucky draw

Two things argue against noise, though see the caveat below.

1. **The pattern matches the clinical definition.** ET is a **kinetic/action**
   tremor; PD is a **rest** tremor. Drinking and finger-to-nose are the standard
   bedside manoeuvres for eliciting kinetic tremor. Those are precisely the two
   tasks that work, while rest, posture, tapping and pronation-supination sit at
   or below chance. Nothing in the pipeline knows which task is which.
2. **It is two tasks, not one.** DRINK (AUC 0.812) and FINGER_NOSE (AUC 0.826)
   agree, and they are separate recordings on separate action codes.

This is also the first result in the project that goes the *right* way against
prior expectation rather than dissolving under checking.

## The caveat, stated as strongly as the result

**There are 6 ET subjects.** At n=6 these numbers are extremely unstable — an
AUC of 0.826 can turn on one or two patients. This project has produced several
findings at small n that did not survive (`reports/handedness_does_not_survive.md`,
`reports/quaternion_session_verdict.md`), and the discipline that caught them
applies here too:

* no paired CI has been run against a baseline;
* no multiplicity correction across the 7 tasks (2 of 7 "hits" at these
  thresholds is not far from what chance would give);
* not replicated in any other cohort — 2015 and PADS have no kinetic tasks, so
  **there is nothing to validate against**.

Treat this as **the strongest available lead, not an established result**.

## What would confirm it, in order of value

1. **More ET subjects on the kinetic tasks.** This is now a much more focused
   ask than "more ET data": 6 → ~20 ET recorded on DRINK and FINGER_NOSE would
   settle it. Unlike every other lever tried, the cost is recruitment on two
   specific tasks rather than a new modality or method.
2. **Paired CIs and BH correction across the 7 tasks**, once n supports them.
3. **PD medication state.** `Clinical Study Subjects - Non-Identifiable Data.csv`
   (not yet uploaded) has H&Y stage and time since last dose. PD kinetic tremor
   is dose-sensitive, so that file bears directly on this result.

## Consequence for the project's direction

Every previous attempt to raise ET precision failed: pooling three cohorts
(0.393 → 0.163), BiLSTM (0.180), temporal features (significantly worse),
multi-sensor (significantly worse), quaternion geometry (retracted). All of them
varied the *method* on rest/postural recordings.

This varied the **task**, and produced the largest jump seen. If it holds, the
lesson is that the binding constraint was never only the ET count — it was
measuring ET during conditions that do not elicit ET tremor.
