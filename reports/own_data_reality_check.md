# What the model actually achieves on 2015 + NewData patients

**The merged headline (ET precision 0.685) does not hold on in-house patients.
There it is 0.193.** Adding PADS to training does not help ET (+0.003) and
significantly hurts PD precision (-0.082).

## Setup

Test sets drawn from **2015 + NewData only**, each containing exactly **10 ET**
patients, at the cohorts' natural prevalence (42 N / 47 PD / 10 ET, ET
prevalence 0.101). 20 repeated draws. PADS enters **training only** -- never
validation, never test -- so every arm is scored on the same in-house patients.

The in-house cohorts hold 21 ET between them, so fixing 10 in test leaves 11 for
train+val. That is the cost of a trustworthy test estimate, and the `own+pads`
arms exist precisely to separate "in-house data is hard" from "11 ET is not
enough to learn from".

## Result

| training data | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **2015 + NewData only** | 0.652 | **0.769** | 0.193 | **0.538** | 0.471 |
| + PADS capped at 90/class | 0.685 | 0.687 | 0.196 | 0.523 | 0.486 |
| + PADS uncapped | 0.687 | 0.653 | 0.190 | 0.510 | 0.488 |

Paired against own-data-only, same 20 test sets:

| | precPD | precET | macroP |
|---|---|---|---|
| + PADS capped | **-0.082 [-0.142, -0.025]** * | +0.003 [-0.108, +0.096] | -0.015 [-0.055, +0.019] |
| + PADS uncapped | **-0.116 [-0.185, -0.050]** * | -0.003 [-0.085, +0.086] | -0.028 [-0.065, +0.011] |

## The correction this forces

`merge_design.md` reported that dropping PADS collapses ET precision from 0.519
to 0.065, and that was used to justify keeping PADS in every merged model. **That
was measured on a merged test set containing PADS patients.** Scored on in-house
patients only, PADS in training contributes nothing to ET.

The merged ET precision of 0.685 was therefore substantially **PADS predicting
PADS**. It is a valid number for the merged cohort and a misleading one for the
in-house cohort, and the two should never be quoted interchangeably.

Adding PADS also **significantly degrades PD precision** on in-house patients,
and more of it degrades it further (-0.082 capped, -0.116 uncapped) -- a dose
response, which is what a domain-mismatch effect looks like. PADS is 72 % PD
recorded on different hardware at a different site; the model learns PADS's PD
and applies it to patients it does not fit.

## What actually holds on in-house data

* **PD precision 0.769** -- the strongest per-class figure, and it is *hurt* by
  adding PADS. This is the most defensible in-house claim available.
* **N precision 0.652**, improved slightly by PADS (+0.033, not significant).
* **ET precision 0.193** at prevalence 0.101 -- a lift of ~1.9x over chance.
  Real but weak, and not improvable with PADS.

## Consequences

1. **Report in-house and merged numbers separately, always.** A merged number
   describes the merged cohort, not this clinic's patients.
2. **PADS is not a substitute for in-house ET patients.** 28 PADS ET subjects do
   not transfer; the 21 in-house ET subjects remain the binding constraint.
3. **If PADS is kept in training, expect a PD-precision cost** on in-house
   patients. Capping at 90/class halves that cost relative to uncapped, which is
   consistent with the capping result but for a different reason than assumed.

Reproduce: `python -m experiments.own_data_10et`.
