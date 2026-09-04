# Euclidean Alignment: right method, wrong unit — and the third instance of one failure shape

## The method and why it looked promising

**Euclidean Alignment** (He & Wu, *IEEE TBME* 67(2) 2020) is the standard
unsupervised domain adaptation in EEG brain–computer interfaces. Per subject,
take the arithmetic mean of their trials' spatial covariances and whiten by its
inverse square root:

    R_bar = mean_i (X_i X_i^T / n)        X_i' = R_bar^(-1/2) X_i

Every subject's mean spatial covariance then equals the identity, removing
per-subject sensor gain, mounting and impedance differences. It is cheap,
label-free, needs nothing from the target subject, and has won BCI competitions.

It is also the closest published method to the one item `SKILL.md` lists as
genuinely open: **feature-level cohort harmonisation, fitted on train only**.
That is why it was worth checking properly rather than importing.

## The argument that made this a diagnostic instead of a run

EA works in BCI because **each subject supplies trials of every class**. The
subject's mean covariance is then a pure subject effect, and whitening by it
leaves the between-class differences intact.

**Here each patient has exactly one label.** A patient's mean covariance is
their class signature as much as their subject signature, so whitening by it
should remove precisely what `_axis_orientation_diagnostic.py` measured to be
worth PD-vs-ET AUC 0.702 / 0.713.

**Cohort-level** alignment is the version that survives the argument: whitening
by the cohort mean removes what differs between 2015, NewData and PADS while
leaving each patient's deviation from their own cohort untouched.

Both were measured at **zero model cost** on the 6-feature tangent vector, with
the same permutation null.

## Result — PD-vs-ET AUC, patient level, within cohort

| cohort | alignment | n | AUC | null p95 | p |
|---|---|---|---|---|---|
| 2015 | none | 90 | **0.702** | 0.656 | 0.000 \* |
| 2015 | **patient-level EA** | 90 | **0.558** | 0.653 | 0.300 |
| 2015 | cohort-level EA | 90 | **0.679** | 0.604 | 0.000 \* |
| NewData | none | 29 | 0.246 | 0.626 | 0.900 |
| NewData | patient-level EA | 29 | 0.551 | 0.706 | 0.350 |
| NewData | cohort-level EA | 29 | 0.341 | 0.774 | 0.600 |
| PADS | none | 304 | **0.713** | 0.619 | 0.000 \* |
| PADS | **patient-level EA** | 304 | **0.574** | 0.562 | 0.050 |
| PADS | cohort-level EA | 304 | **0.708** | 0.630 | 0.000 \* |

NewData is uninformative on this contrast at n = 29 — it is inside its null in
every arm including the unaligned one, so it neither supports nor contradicts.

## The prediction held, on both halves

Recorded before the run: **patient-level EA collapses the AUC to chance;
cohort-level EA does not.**

* **Patient-level: collapsed.** 2015 falls 0.702 → 0.558 and lands *inside* its
  permutation null (p = 0.300). PADS falls 0.713 → 0.574, marginal at p = 0.050
  against a null p95 of 0.562. On both cohorts the surviving AUC is
  indistinguishable from, or barely above, chance.
* **Cohort-level: preserved.** 2015 holds 0.679 and PADS 0.708, both p < 0.001.
  Whitening by the cohort mean costs essentially nothing.

## The failure shape, now seen three times

Patient-level EA fails for the same structural reason PCEN did:

> **Dividing a unit by its own average destroys what varies across the units
> being classified.**

| method | what it divides by | what that erases here |
|---|---|---|
| PCEN | each band's own time-average | *which band* has energy |
| patient-level EA | each patient's own mean covariance | *which direction* the patient's tremor points |
| per-cohort priors (`cohort_strategies.md`) | each cohort's own class mix | the class prior the pooled model needs |

The generalisable rule: **an adaptive normaliser is safe only when the unit it
normalises over contains every class.** In BCI a subject does; here a patient
does not, a band does not, and a cohort only partly does. Check that before
importing any per-unit normalisation, and the check is cheap.

## Standing

* **Do not apply Euclidean Alignment per patient.** It removes the class
  signature by construction, and the diagnostic confirms it at zero fit cost.
* **Cohort-level alignment is not refuted** — it preserves the orientation
  information, which is the precondition for it being useful, not evidence that
  it helps. Whether harmonising the three cohorts this way moves the model is
  untested and remains the open item. It is worth a real arm **only if
  `riemann_axes.md` shows the covariance composes to the model at all**;
  harmonising a representation the model ignores would be pointless.
* **The diagnostic cost minutes and closed a method.** Second time this session
  after PCEN, and this one closed it *before* the fits rather than explaining
  them afterwards.
