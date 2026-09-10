# The time axis: order carries nothing, and what is left cannot be pinned down

## What was tested

Every spectral model in this project consumes `P.mean(0)` — the time-frequency
surface averaged over its frames **before any model sees it**. Nothing here had
modelled the time axis: MIL pooled over *recordings* (significantly worse), HPSS
is a two-way sustained/transient split, and short windows are a different
time-frequency trade-off rather than a temporal model.

`ConvTimeTransformer` (18.6 k params, inside the 9–35 k band this cohort peaks
in): 1-D conv over frequency per frame → transformer with sinusoidal positions
over 46 frames → mean-pool → classify. Recording-level throughout, probabilities
aggregated to the patient, splits patient-disjoint.

## Result — 20 splits, standalone

| arm | precN | precPD | precET | macroP | macroF1 | recET | nETpred |
|---|---|---|---|---|---|---|---|
| A: CNN on `P.mean(0)` | 0.646 | 0.628 | 0.548 | **0.608** | 0.576 | 0.455 | 8.80 |
| B: conv+time-transformer | 0.658 | 0.619 | 0.548 | **0.608** | 0.569 | 0.430 | 8.45 |
| C: B, frames **shuffled** | 0.672 | 0.631 | **0.600** | **0.634** | 0.586 | 0.465 | 8.25 |
| D: B, **no position encoding** | 0.647 | 0.638 | 0.576 | 0.620 | **0.588** | 0.465 | 8.40 |

**Paired vs A:**

| arm | precET | macroP | macroP win |
|---|---|---|---|
| B | −0.000 [−0.044, +0.046] | +0.001 [−0.025, +0.028] | 0.35 |
| **C** | **+0.052 [+0.013, +0.093]** \* | **+0.027 [+0.005, +0.052]** \* | 0.65 |
| D | +0.028 [−0.035, +0.086] | +0.013 [−0.013, +0.038] | **0.70** |

## The one solid finding: temporal order carries nothing

**A and B are identical to three decimals** — macroP 0.608 both, precET 0.548
both. Giving the model the real frame order reproduces the time-average exactly,
and *destroying* that order (C) improves on it.

This is a clean, quotable negative: at n = 404, the **order** of tremor frames
carries no usable class information. It closes a family rather than leaving it
open, which is what the experiment was designed to do.

## The mechanism cannot be resolved, and that is the honest verdict

Two accounts of why C beats A:

    removing order       forcing permutation-invariance lets the model use the
                         frame DISTRIBUTION, which the mean destroys
    shuffle as noise     a per-split random permutation is input augmentation,
                         and regularisation is doing the work

Arm D — `pos_enc=False`, verified permutation-invariant to 1.2e-07 — was built
to separate them. It lands **between** A and C:

    D vs C   macroP −0.014 [−0.042, +0.015]   null on every column
    spread   0.014, against the ~0.040 that 20 splits resolves

**The two candidate mechanisms are 0.014 apart against a resolution of 0.040.
This experiment does not have the power to choose between them**, and no amount
of re-reading the table changes that.

The recorded prediction, *"D matches C to within noise"*, is **satisfied in the
letter and empty in substance** — "within noise" holds because the noise is
larger than the difference under test. Same shape as the easy-drop prediction
earlier in this project: a prediction can hold and tell you nothing once the
comparison lacks resolution.

## The win rates disagree with the means, and that matters

| arm vs A | precET mean | precET win | macroP mean | macroP win |
|---|---|---|---|---|
| C | **+0.052** \* | **0.45** | +0.027 \* | 0.65 |
| D | +0.028 | **0.70** | +0.013 | **0.70** |

C has the larger mean with a **sub-0.5 precET win rate** — the exact pattern
invariant 5 exists to catch, and the one that has burned this project before. D
has a smaller mean with a consistent 0.70 win rate across precPD, precET, macroP
and macroF1.

**On the project's own rules, D is the more trustworthy of the two despite
scoring lower.** C's significant precET is likely carried by a few favourable
folds.

## What this is not

**None of these beats the reported model.** The best arm here is macroP 0.634
against the reported **0.652**; baseline A is a *standalone* `Spectrum1DCNN`, not
the six-member two-stream ensemble. So the correct summary is "a bag-of-frames
input beats a stripped-down CNN", not "beats the reported model".

Whether it survives fusion into the real ensemble is a separate test, and the
prior is unfavourable: **16 feature unions in this project have underperformed
their best member**, and dimensionality binds hard at n = 404.

## Standing

* **Temporal order is closed.** A = B to three decimals. Do not re-open without
  a cohort where recordings are long enough for order to mean something.
* **The frame distribution may carry a little**, worth macroP +0.013 to +0.027
  over a stripped-down CNN — but the effect is inside the fragility band, the
  two mechanisms cannot be separated at this n, and the larger of the two
  estimates has a sub-0.5 win rate.
* **Prefer D over C if either is pursued**, on win-rate consistency and because
  it is deterministic rather than depending on a per-split random permutation.
* **Do not put this in a paper as a performance result** unless it clears the
  reported model in fusion at 40 splits.
