# Logit adjustment: positive against my prediction, and null against the data

## What was tested

The reported model corrects its 167 / 188 / 49 imbalance twice — inverse-frequency
**class weights** inside the loss, and validation-tuned **logit offsets**
afterwards. The second is the project's largest measured gain. The first had
never been compared against its modern replacement.

Menon et al. (ICLR 2021) show weighting and additive **logit adjustment** are
different corrections for the same imbalance: the loss becomes
`CE(z + tau·log(prior), y)` and inference uses `z` unadjusted. Weighting rescales
each example's gradient, so at 49 ET a handful of patients dominate the update;
adjustment shifts the boundary inside the loss and leaves gradient magnitudes
alone. It is the canonical long-tail baseline, with class-balanced reweighting
treated as its weaker predecessor.

`common.protocol.train(..., logit_adj=tau)` implements it; `logit_adj=None`
is verified bit-identical to the existing path, so no prior result moves.

**`tune_offsets` runs in every arm**, so a post-hoc correction absorbs anything
adjustment did to the threshold alone. A gain that survives it would have to be
a *representation* gain — which is the only reason this was worth a run.

## Result

| arm | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **baseline (class weights)** | 0.648 | 0.654 | 0.654 | 0.652 | 0.593 | 0.066 |
| logit adj τ = 0.5 | 0.652 | 0.653 | **0.688** | **0.664** | **0.602** | **0.061** |
| logit adj τ = 1.0 | 0.649 | **0.663** | 0.660 | 0.657 | 0.601 | 0.066 |

paired vs the class-weighted baseline, **40 splits**:

| arm | precET | macroP |
|---|---|---|
| τ = 0.5 | +0.034 [−0.004, +0.071] | +0.012 [−0.003, +0.027] |
| τ = 1.0 | +0.006 [−0.048, +0.054] | +0.005 [−0.015, +0.023] |

**Null.** Every interval spans zero.

## The number that matters is how it moved on doubling

| τ = 0.5 | 20 splits | 40 splits |
|---|---|---|
| macroP | +0.023 [−0.002, +0.050] | **+0.012** [−0.003, +0.027] |
| precET | +0.053 [−0.013, +0.120] | **+0.034** [−0.004, +0.071] |
| macroP win rate | 0.70 | **0.53** |
| precET win rate | 0.55 | **0.42** |

**The effect halved on doubling the splits, and the win rates collapsed to
chance.** This is the fourth time in this project a difference of ~0.02–0.03 has
shrunk or flipped when the split count doubled, and it is exactly why
`headline_audit.md` exists.

The precET win rate is the sharpest tell: **0.42 — τ = 0.5 loses on ET precision
more often than it wins**, despite a positive mean of +0.034. A positive mean
built from a minority of large wins is not a method that helps; it is a method
whose gains are a few favourable folds.

Extrapolating: resolving +0.012 macroP at this variance would need roughly 160
splits. An effect that halves each time it is measured more carefully is not
worth that.

## The prediction was wrong in direction, and the reasoning is worth keeping

Recorded before the run: *"small, and more likely negative than positive on
precET, with τ = 0.5 less harmful than τ = 1.0"*, on the grounds that
`prior_objective.md` measured **precET −0.236 [−0.340, −0.142]** * when this
project's imbalance correction was aimed at balanced accuracy, and logit
adjustment is Fisher-consistent for exactly that objective.

**"Small" was right; "negative" was wrong; "τ = 0.5 over τ = 1.0" was right.**

The flaw is worth naming because it is a general one: I treated *post-hoc offset
tuning* and *training-time adjustment* as the same intervention because they
target the same objective. They are not. The offset search moves a threshold on
a fixed representation and can overfit ~11 validation ET patients; adjustment
shapes the representation while it is being learned and never sees the
validation split. Sharing an objective does not make two methods share a failure
mode.

## Standing

* **Do not adopt logit adjustment.** Null at 40 splits, and the effect halved on
  doubling.
* **τ = 0.5 beats τ = 1.0** consistently, so if this is ever revisited — a larger
  ET cohort, say — mild adjustment is the setting to try, not full.
* **`logit_adj` stays in `train`**, defaulting to `None` and verified bit-exact.
  It costs nothing and the next person should not have to re-implement it.
* **A positive mean with a sub-0.5 win rate is not a result.** Report the win
  rate beside every paired mean; it distinguishes "helps generally" from "helps
  on a few folds".
* Registered in `failed_predictions.md` as wrong in direction, right in
  magnitude.
