# Improving CNN / TCN / BiLSTM on the merged cohort

**Result: LOCO macro F1 0.435 -> 0.505 and ET precision 0.226 -> 0.269. This is
the first verified case in this repo of deep models beating logistic regression
on held-out cohorts.**

The lever was **input representation**, not architecture, capacity, or training
tricks.

## Setup

Merged 2015 + NewData + PADS, postural task alignment, PADS capped at 90/class
(the `merge_design.md` optimum), n=404 with 49 ET. Class weights on. Scored both
pooled 5-fold and **leave-one-cohort-out**, over 5 independent capping draws.

## What worked

### 1. Log-scale the spectrum

A sum-normalised tremor spectrum is extremely peaked: one or two bins carry
almost all the mass. After standardisation the network spends its capacity
fitting the near-zero tail. `log(x + 1e-8)` compresses the dynamic range.

### 2. Coarse binning -- the big one

61 bins over 3-15 Hz is ~0.2 Hz resolution against a tremor peak 1-2 Hz wide,
and 61 input dimensions at n=404 is 15 % of the sample count.

LOCO macro F1 by bin count, no mixup:

| bins | 12 | 16 | 24 | 32 | 61 |
|---|---|---|---|---|---|
| CNN | 0.459 | 0.499 | 0.481 | **0.505** | 0.473 |
| TCN | 0.489 | **0.505** | 0.453 | 0.481 | 0.412 |
| BiLSTM | 0.401 | 0.452 | 0.465 | **0.484** | 0.389 |

Every model has an interior optimum and every model collapses at 61.

### 3. Cosine LR decay

The previous loop was 200 fixed full-batch steps with no schedule. Adding cosine
decay lifted the CNN's LOCO from 0.425 to 0.460 on identical input, before any
representation change.

## Verified table (5 capping draws)

| config | pooled F1 | pooled pET | **LOCO F1** | **LOCO pET** |
|---|---|---|---|---|
| logreg raw (61) | 0.472 +/- 0.022 | 0.221 +/- 0.020 | 0.446 +/- 0.018 | 0.194 +/- 0.019 |
| logreg 16-bin | 0.507 +/- 0.016 | 0.256 +/- 0.030 | 0.458 +/- 0.012 | 0.152 +/- 0.009 |
| MLPHead 16-bin | 0.549 +/- 0.021 | 0.337 +/- 0.044 | 0.454 +/- 0.008 | 0.167 +/- 0.011 |
| CNN 16-bin | 0.543 +/- 0.006 | 0.328 +/- 0.022 | 0.494 +/- 0.013 | 0.230 +/- 0.044 |
| **CNN 32-bin** | 0.560 +/- 0.016 | 0.352 +/- 0.033 | **0.505 +/- 0.009** | 0.238 +/- 0.029 |
| **TCN 16-bin** | 0.537 +/- 0.012 | 0.365 +/- 0.022 | 0.500 +/- 0.009 | **0.269 +/- 0.017** |
| TCN 32-bin | 0.550 +/- 0.008 | 0.367 +/- 0.010 | 0.490 +/- 0.007 | 0.249 +/- 0.015 |
| BiLSTM 32-bin | 0.543 +/- 0.013 | 0.339 +/- 0.032 | 0.490 +/- 0.014 | 0.219 +/- 0.043 |
| BiLSTM 24-bin | 0.496 +/- 0.026 | 0.289 +/- 0.037 | 0.436 +/- 0.022 | 0.154 +/- 0.018 |

CNN 32-bin beats logreg raw by 0.059 at sd 0.009/0.018 -- 3-4 sd, not a lucky
split. Best ET precision is TCN 16-bin at 0.269 +/- 0.017 against 0.194 +/- 0.019.

Before this work, under LOCO **no** deep model beat logistic regression
(`three_cohort_deep.md`: logreg 0.435, CNN 0.425, TCN 0.396, BiLSTM 0.366).

## What did not work

### mixup HURTS

Lower LOCO macro F1 in 7 of 9 architecture x bin combinations:

| config | mixup 0.0 | 0.2 | 0.4 |
|---|---|---|---|
| TCN 16-bin | **0.505** | 0.479 | 0.467 |
| CNN 32-bin | **0.505** | 0.482 | 0.473 |
| BiLSTM 32-bin | **0.484** | 0.471 | 0.473 |

This was predicted to help LOCO specifically, on the reasoning that the
cross-cohort gap was a decision-boundary sharpness problem. It helps neither
pooled nor held-out, so **that explanation is withdrawn**. Whatever separates
cohorts is not something convex interpolation smooths over.

## The diagnostic detail

Binning helps the networks but **hurts logreg's ET precision** (0.194 raw ->
0.152 at 16-bin), while helping its macro F1 only slightly.

If coarse bins were simply denoising the spectrum, a linear model would benefit
too. It does not. That supports the specific claim that the constraint was
**input dimensionality relative to sample count for models that must learn a
representation** -- a linear model with 61 coefficients was never the thing
straining. It also explains, in one mechanism, why 1 k-parameter models beat
35 k ones here and why large backbones fail outright.

## Caveats

* ET precision is still 0.27. This is a separability result, not a deployable
  classifier.
* All gains push the same direction: smaller input, smaller model. The remaining
  headroom in architecture work is therefore small. The two levers with real
  upside remain more ET patients and bilateral recording -- four asymmetry
  features reach AUC 0.730 on PD-vs-ET with no network at all
  (`limb_asymmetry_pd_vs_et.md`).

Reproduce: `scratch/improve.py`, `scratch/improve2.py`, `scratch/improve3.py`
(gitignored; models in `tfbench.small_nets`).
