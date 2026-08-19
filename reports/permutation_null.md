# No in-house PD-vs-ET result is distinguishable from chance

**What this settles.** Two things at once: the "below chance" readings this
project has reported are not real, and neither are most of the above-chance
in-house ones. At 21 in-house ET patients the permutation null for PD-vs-ET AUC
spans roughly **[0.30, 0.66]**, and every in-house family measured falls inside
it.

Run: `python -m experiments.cv_null`. Identical pipeline, labels permuted 200
times, so the null is what the *whole procedure* — fold splitting, class
weighting, fitting, out-of-fold scoring — produces when the labels carry no
information.

## The null is centred, but enormous

| cohort | ET | folds | null mean | null 95 % |
|---|---|---|---|---|
| PADS | 28 | 5 | 0.493 | [0.346, 0.624] |
| in-house pooled | 21 | 3 | 0.491 | [0.298, 0.655] |
| 2015 only | 15 | 3 | 0.492 | [0.323, 0.684] |
| NewData only | 6 | 3 | 0.490 | [0.195, 0.819] |

The null is **not** shifted below 0.5, so there is no systematic negative bias —
the mechanism I proposed (held-out minority patients excluded from their own
class centroid) does not produce one. What the small minority class produces is
*variance*, and a great deal of it.

## Observed AUC and its permutation p-value

| family | PADS | p | in-house | p | 2015 | p |
|---|---|---|---|---|---|---|
| descriptors | **0.794** | **0.005** | 0.430 | 0.413 | 0.492 | 0.995 |
| spectrum | **0.791** | **0.005** | 0.568 | 0.378 | 0.625 | 0.204 |
| stability | **0.757** | **0.005** | 0.547 | 0.488 | 0.665 | 0.114 |
| axes | 0.565 | 0.279 | 0.629 | 0.085 | 0.677 | 0.104 |
| harmonics | **0.726** | **0.005** | 0.421 | 0.438 | 0.410 | 0.542 |
| ampmod | **0.700** | 0.010 | 0.464 | 0.572 | 0.500 | 0.995 |

* **On PADS, five of six families are significant.** Those results stand.
* **On in-house patients, not one family reaches significance.** The best is
  `axes` at p = 0.085, and `stability` on 2015 at p = 0.114.
* **`axes` is not significant on PADS either** (p = 0.279), which is consistent
  with it being the family that does *not* differ between cohorts.

## Two claims are withdrawn

**1. The "anti-predictive" verdicts.** `experiments/family_inversion.py`
bootstrapped over patients and reported descriptors 0.339 [0.225, 0.461],
harmonics 0.323 [0.225, 0.430] and asymmetry 0.307 [0.183, 0.449] in-house, all
with intervals excluding 0.5, and labelled them ANTI-predictive. Under
permutation those same families give p = 0.413, 0.438 — ordinary chance results.

The two tests disagree because they hold different things fixed. **The patient
bootstrap resamples patients while keeping the out-of-fold predictions fixed**,
so it estimates "what if a different sample of patients had been scored by this
already-fitted model". It cannot see the variance of the fitting procedure
itself, and at 21 ET that variance is the dominant term. The permutation test
refits everything on every replicate and therefore includes it. **Where the two
disagree at this sample size, the permutation test is the one to believe.**

This also explains the NewData AUC of exactly 0.000 (all 6 ET below all 23 PD):
its null spans [0.195, 0.819], so 0.000 is extreme but the procedure's spread at
6 ET is so wide that only a perfect inversion registers at all. `ampmod` there
reaches p = 0.005, the smallest value 200 permutations can produce, but that is
1 significant result in 24 tests with a Bonferroni threshold near 0.002, on a
block containing a constant feature. It is noise.

**2. "On 2015 every frequency feature is below chance (0.29–0.32)."** This was
recorded in the skill file as a finding. 2015 descriptors measure 0.492 with
p = 0.995 — the single most ordinary result in the table. There is no
below-chance effect on 2015 to explain.

## What this does *not* say

It does **not** say the in-house families carry no signal. `axes` at p = 0.085
and `stability` on 2015 at p = 0.114 are exactly what a real but modest effect
looks like when there is not enough data to resolve it. The correct reading is
**underpowered, not refuted**.

The useful number is the detection floor. With a null upper bound of 0.655 at 21
ET, an in-house PD-vs-ET model must reach **AUC ≈ 0.66 before it can be
distinguished from chance at all** — and the best measured is 0.629. That is a
cleaner statement of the project's binding constraint than any precision figure,
because it does not depend on prevalence or on a threshold.

## Consequence for anything in-house

Every in-house PD-vs-ET comparison in this repo has been made between models
whose individual AUCs sit inside this null. Paired tests between them remain
valid — a paired difference can be real even when neither arm separates from
chance, because the pairing removes the shared fold noise. But **no in-house
single-model claim of the form "family X separates PD from ET" is supported**,
including the axis-feature result the README reports.

Reported per-class precision is subject to the same limit and should be read
alongside it.
