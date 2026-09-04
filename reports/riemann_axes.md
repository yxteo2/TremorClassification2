# Riemannian tangent space: the orientation is real, and the model cannot use it

## The gap

`method_table` computes a power spectrum **per gyroscope axis and then averages
the axes**. The direction the limb oscillates in is discarded before any model
sees it. `spectral_representation.md` closed the two obvious repairs — principal
eigenvalue and polarisation spectrum — but both are *rotation-invariant
scalars*: they keep the oscillation's strength and throw the orientation away.

The untested object is the full covariance matrix. Its log-Euclidean tangent
vector (Barachant et al., *IEEE TBME* 59(4) 2012 — the canonical small-n method
in EEG brain–computer interfaces) is 6 numbers for a 3×3, and is **scale-free
once the matrix is trace-normalised**, so it survives the sum-normalisation that
deleted λ₁'s gain. It also satisfies the repaired rule in `closed_families.md`:
*few features, selected for classification*.

Physiology: PD rest tremor is classically pronation–supination, a rotation about
the forearm's long axis; ET postural tremor is predominantly flexion–extension.

## The information is unambiguously there

`_axis_orientation_diagnostic.py`, no model, before any fits:

    anisotropy lambda_1/trace      0.65-0.81   (1/3 would be isotropic)
    axis reliability, 2015         15.8 deg within patient vs 40.2 between
    PD-vs-ET AUC, 6 features       2015 0.702 (null p95 0.578) *
                                   PADS 0.713 (null p95 0.604) *

Strongest pre-run signal measured in this project.

## The model result — 20 splits, paired

| arm | precN | precPD | precET | macroP | macroF1 | recET | nETpred |
|---|---|---|---|---|---|---|---|
| reported | 0.642 | 0.649 | 0.648 | 0.646 | 0.590 | 0.475 | 8.95 |
| + tangent (6) | 0.642 | **0.665** | 0.642 | 0.649 | **0.594** | 0.430 | 7.75 |
| **+ tangent SHUFFLED** | 0.645 | 0.641 | **0.663** | **0.650** | 0.592 | 0.435 | 7.40 |
| tangent *replacing* descriptors | **0.658** | 0.639 | 0.616 | 0.638 | 0.590 | 0.455 | 9.10 |

**Adoption — paired vs the reported model:**

| arm | precET | macroP |
|---|---|---|
| + tangent (6) | −0.006 [−0.078, +0.062] | +0.003 [−0.021, +0.025] |
| + tangent SHUFFLED | +0.015 [−0.048, +0.075] | +0.003 [−0.017, +0.023] |
| tangent replacing descriptors | −0.032 [−0.080, +0.011] | −0.009 [−0.027, +0.010] |

**The shuffled control reproduces the real feature.** Six columns of
within-cohort-permuted noise score +0.003 macroP and +0.015 precET, against the
real feature's +0.003 and −0.006. Whatever the fusion moves is **dimensionality,
not orientation**.

## The one attribution signal that survives

Paired against the shuffle rather than the baseline, real orientation is worth

    precPD  +0.024 [+0.008, +0.041] *      win rate 0.70

So the model *does* read the orientation, and it reads it for PD precision. But
against the plain baseline that same arm is +0.016 [−0.017, +0.042], null.
**Attribution positive, adoption null** — the two questions the matched control
exists to separate.

## Predictions, scored

1. *"tangent alone loses to the reported model"* — **held.** macroP −0.009,
   precET −0.032.
2. *"any gain is larger on precPD and precET than on precN"* — **failed.**
   precN −0.000, precPD +0.016, precET −0.006. precPD held up; precET came in
   *below* precN, which is the half the prediction most needed.
3. *"macroP leaning positive, small, uncertain"* — technically satisfied at
   +0.003, and **rendered meaningless by the control**, which gives the identical
   +0.003. A prediction a shuffled feature also satisfies was not a prediction
   about this feature.

## A correction to this experiment's own labelling

The arm named `tangent alone` does **not** measure the feature in isolation. It
swaps the 10 descriptors for the 6 tangent numbers while keeping the spectrum
and trajectory streams. Read correctly it says something worth having:
**substituting six covariance numbers for ten hand-computed spectral descriptors
costs macroP −0.009 and precN +0.016** — the descriptor block is very nearly
replaceable by a rotation summary a quarter its size.

A side result with a significant interval: adding six *random* columns cuts ET
predictions by **−1.55 per split [−3.35, −0.15] \***. Dimensionality alone makes
the model more conservative about the minority class, which is the mechanism to
suspect whenever an appended feature block moves precET without moving recET.

## Why it fails: redundant on one axis, unusable on the other

`_tangent_complementarity_diagnostic.py` — logistic regression, patient level,
within cohort, permutation nulls — was written **before** these numbers existed
and enumerated what each outcome would mean. It answers differently on the two
contrasts, which is sharper than either branch it anticipated:

| cohort | contrast | descriptors | tangent | union | verdict |
|---|---|---|---|---|---|
| PADS | PD vs ET | **0.795** | 0.713 | 0.797 | **redundant** (+0.002) |
| 2015 | PD vs ET | 0.474 (p = 0.75) | **0.702** \* | 0.677 | tangent carries what descriptors lack; union *dilutes* |
| PADS | N vs tremor | 0.791 | 0.579 | **0.818** | complementary (+0.027) |
| 2015 | N vs tremor | 0.890 | 0.761 | **0.894** | complementary (+0.004) |
| NewData | N vs tremor | 0.852 | 0.709 | **0.935** | complementary (+0.083), n = 56 |

* **On PD-vs-ET, where it would have mattered, the tangent is redundant.** On
  the only cohort with power (PADS, n = 304) the union beats the descriptors by
  +0.002. The two feature sets are reading the same thing. **So the deep null on
  precET needs no appeal to rule #5** — there was nothing new to compose.
* **On N-vs-tremor the tangent is genuinely complementary** (+0.027 on PADS at
  n = 383) and the 3-class model still moved precN by −0.000. **That half is
  rule #5, fourth instance.**
* **2015 PD-vs-ET is the interesting anomaly.** The descriptors sit at chance
  (AUC 0.474, p = 0.75) while the tangent reaches 0.702 — and the union *loses*
  to the tangent alone, ten uninformative columns diluting six informative ones.
  Stated with its caveat: this is 5-fold CV at n = 90, not the repo's protocol,
  so it should **not** be quoted against the documented in-house PD-vs-ET floor
  of 0.655 (`permutation_null.md`) without a matched re-run.

## Standing

* **Do not adopt the tangent vector.** Null on adoption, and a within-cohort
  shuffle of the same six columns performs as well.
* **The orientation is real and the model cannot use it.** AUC 0.702 / 0.713
  model-free; precPD +0.024 \* against the shuffle; nothing against the baseline.
  This is the fourth and cleanest instance of descriptor-level gains failing to
  compose — cleanest because the control, not just the interval, carries it.
* **Report a shuffled control on every appended feature block.** Without it this
  would have been written up as "+0.003 macroP, +0.016 precPD, promising".
* **Cohort-level Euclidean Alignment is now moot** (`euclidean_alignment.md`
  made it conditional on this result). Harmonising a representation the model
  demonstrably ignores has nothing to act on.
* **`spectral_representation.md`'s conclusion is extended, not overturned.** It
  found rotation-*invariant* summaries null because normalisation removes them;
  this finds the rotation-*covariant* full matrix null for a different reason —
  redundancy with the descriptors on the contrast that matters. Both routes into
  inter-axis structure are now closed, by different mechanisms.
