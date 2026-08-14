# How to combine 2015 + NewData + PADS

Judged throughout by **leave-one-cohort-out** or a **fixed held-out cohort**,
never by pooled k-fold. Pooled CV was shown in `three_cohort_deep.md` to reward
cohort fitting: every deep model's pooled advantage vanished held out, and the
pooled-to-LOCO gap grew with capacity.

## 1. There is no domain shift left to correct

Cohort-identity probe, reported as |accuracy - majority| so that 0 means the
cohorts are indistinguishable:

| alignment | |probe - maj| (postural) |
|---|---|
| **none** | **0.023 - 0.273** |
| per-cohort z-score | 0.285 - 0.480 |
| per-cohort rank | 0.306 - 0.471 |
| CORAL covariance | 0.241 - 0.407 |

`spectrum_table` normalises each spectrum to sum 1 and averages the three axes.
That is scale-invariance plus the rotation-invariant trace, and it already
removes the cohort signature -- 0.023 at cap 60, 0.003 at REST cap 30.

**Every alignment method makes cohort identity MORE detectable.** Per-cohort
centring subtracts each cohort's own class mixture, and PADS is 72 % PD against
a balanced 2015, so "aligning" injects a class-dependent shift that becomes the
signature. This also resolves the earlier puzzle in which per-cohort z-scoring
appeared to remove the domain shift yet hurt accuracy: it never removed it.

**Do not apply distribution alignment to these features.**

## 2. Merge on the postural task, not REST

Best LOCO macro F1 / ET precision, logreg, 5 capping draws:

| task alignment | best macroF1 | best precET |
|---|---|---|
| **postural** (2015 OUT / NewData OUT / PADS StretchHold) | **0.451** | **0.282** |
| rest (2015 REST / NewData REST / PADS Relaxed) | 0.399 | 0.209 |

The margin holds across nearly every configuration. A previous decision to merge
on REST came from 2015-only evidence; for cross-cohort merging, postural wins.

One dissociation worth noting: at REST the alignment methods *help*
(none 0.352 -> rank 0.393) while at postural they hurt. Rest tremor is weak --
ET barely tremors at rest -- so cohort differences dominate the signal there and
normalising pays. It is not enough to make REST competitive.

## 3. Capping: a retracted claim

The LOCO sweep showed ET precision falling monotonically with more PADS
(0.282 -> 0.225 -> 0.194 -> 0.178 -> 0.156, sd 0.008-0.019), and this was
initially read as "adding PADS destroys the minority class".

**That was a prevalence artifact and is withdrawn.** Capping PADS also changes
the PADS *test* fold: at cap 30 its ET prevalence is 0.318, uncapped 0.073.
Precision tracks prevalence mechanically.

Re-run with the test cohort **fixed** so only the training set varies:

### test = 2015 (fixed, n=151, 15 ET, ET prevalence 0.099)

| PADS cap | n_train | precN | precPD | precET | lift | macroF1 |
|---|---|---|---|---|---|---|
| 0 (no PADS) | 56 | 0.517 | 0.667 | 0.150 | 1.51 | 0.421 |
| 30 | 144 | 0.603 | 0.591 | 0.210 | 2.12 | 0.461 |
| 60 | 204 | 0.596 | 0.581 | 0.209 | 2.10 | 0.461 |
| **90** | 253 | 0.655 | 0.624 | 0.217 | 2.19 | **0.498** |
| 150 | 313 | 0.627 | 0.570 | **0.227** | 2.28 | 0.475 |
| none | 439 | 0.653 | 0.549 | 0.226 | 2.27 | 0.473 |

### test = NewData (fixed, n=56, 6 ET, ET prevalence 0.107)

| PADS cap | n_train | precET | lift | macroF1 |
|---|---|---|---|---|
| **0 (no PADS)** | 151 | **0.167** | 1.56 | 0.386 |
| 30 | 239 | 0.139 | 1.30 | 0.384 |
| 60 | 299 | 0.130 | 1.22 | 0.396 |
| 90 | 348 | 0.128 | 1.20 | 0.392 |
| 150 | 408 | 0.121 | 1.13 | 0.373 |
| none | 534 | 0.118 | 1.10 | 0.359 |

The trend is **target-dependent, not monotone**: PADS helps 2015 substantially
and hurts NewData mildly. At 6 ET subjects a 0.02 change on NewData is under one
patient, so that arm carries little weight.

Recommended cap for a 2015 target: **90-150/class**, not the 30 suggested by the
confounded sweep, and not the 60 used in earlier work.

## 4. Does merging beat a single cohort at all?

Postural, PADS capped at 30/class, no alignment, 5 draws:

| test | trained on | precET | macroF1 |
|---|---|---|---|
| 2015 | NewData | 0.150 | 0.421 |
| 2015 | PADS | 0.187 | 0.445 |
| 2015 | **NewData + PADS** | **0.210** | **0.461** |
| NewData | **2015** | **0.167** | **0.386** |
| NewData | PADS | 0.094 | 0.338 |
| NewData | 2015 + PADS | 0.139 | 0.384 |
| PADS | **2015** | 0.479 | **0.476** |
| PADS | NewData | **0.516** | 0.435 |
| PADS | 2015 + NewData | 0.496 | 0.453 |

Merging beats the *average* single source (mean +0.016) but **never beats the
best single source** (merged mean 0.433 vs best-single 0.436). Since the best
single source cannot be identified in advance without labels on the target,
merging buys **robustness, not peak accuracy**. That is a legitimate reason to
merge, but it is not the reason previously assumed.

Notable in its own right: **training on 2015 alone and testing on PADS gives ET
precision 0.479 and macro F1 0.476**, with no PADS in training -- a clean
external-validation result. (Read against a 0.318 ET prevalence in the capped
PADS test set: lift 1.51.)

## Recommended configuration

* postural task alignment (OUT / OUT / StretchHold)
* **no** distribution alignment
* PADS capped at 90-150/class when the target is 2015; no PADS when the target
  is NewData
* logistic regression on `spectrum_table` features, class weights balanced
* report per-class precision **with the test set's class prevalence**, since
  precision is not comparable across differently-composed test sets

Reproduce: `scratch/mergesweep.py`, `scratch/mergevalue.py`,
`scratch/capclean.py` (gitignored).
