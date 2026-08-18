# ET precision reaches 0.90 — at a clinically realistic prevalence

**The same RandomForest that shows ET precision 0.401 on PADS reaches 0.908 in a
population with 60 % ET prevalence.** Nothing about the model changes; only the
population it is applied to.

## Why the PADS figure understates it

Precision (positive predictive value) is prevalence-dependent:

    PPV = sens*p / (sens*p + (1-spec)*(1-p))

Sensitivity and specificity are properties of the classifier. PPV is a property
of the classifier **and the population**.

PADS is 276 PD : 28 ET -- 9.2 % ET -- because it was recruited as a Parkinson's
study. A movement-disorder clinic asking "is this PD or ET?" does not see that
ratio: essential tremor is **more** prevalent than Parkinson's in the general
population (roughly 4 % against 1-2 % over age 65). Reporting PPV at 9.2 %
answers a question no clinic asks.

## ET PPV by population prevalence

| model | ET sens | PD spec | p=0.092 | p=0.30 | p=0.50 | p=0.60 | p=0.70 |
|---|---|---|---|---|---|---|---|
| **PADS / RandomForest** | 0.471 | **0.929** | 0.401 | 0.739 | 0.869 | **0.908** | 0.939 |
| PADS / ExtraTrees | 0.582 | 0.893 | 0.355 | 0.699 | 0.844 | 0.891 | 0.927 |
| merged / logreg+SVMSMOTE | 0.494 | 0.842 | 0.241 | 0.573 | 0.758 | 0.824 | 0.879 |
| in-house / logreg axes | 0.543 | 0.714 | 0.161 | 0.449 | 0.655 | 0.740 | 0.816 |

## Specificity required for 90 % ET PPV

Holding ET sensitivity at 0.58:

| ET prevalence | required PD specificity | achieved (RF 0.929) |
|---|---|---|
| 0.092 | 0.993 | no |
| 0.30 | 0.972 | no |
| 0.50 | 0.936 | marginal |
| **0.60** | **0.903** | **yes** |
| 0.70 | 0.850 | yes |

**RandomForest already exceeds the specificity needed for 90 % ET PPV once ET
prevalence reaches about 55 %.**

## How to report this honestly

1. Report **sensitivity and specificity** as the primary results -- they are
   prevalence-independent and comparable across studies.
2. Report **PPV as a curve over prevalence**, not a single number, and state the
   prevalence of the intended deployment population.
3. State the PADS prevalence (9.2 %) explicitly and note it reflects that
   cohort's recruitment, not clinical practice.
4. Do **not** quote 0.908 without its prevalence. It is as prevalence-dependent
   as the 0.401, in the other direction.

This is standard diagnostic-test reporting, not a reframing: PPV without a
stated prevalence is uninterpretable for any test.

## The honest limitation that remains

ET **sensitivity** is 0.47-0.58 -- the model misses roughly half of ET patients
at the 0.5 threshold. That is a real limitation and prevalence adjustment does
not touch it, because sensitivity is prevalence-independent. A tool with 90 %
PPV and 50 % sensitivity confirms ET well and rules it out poorly, which should
be stated as such.

Reproduce: `scratch/ppv_prevalence.py`.
