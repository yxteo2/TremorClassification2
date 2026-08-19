# The class priors are tuned for the "wrong" metric, and fixing that makes it worse

**What prompted this.** Validation-tuned class priors are the second-largest
measured contributor in this project — "ET precision 0.475 → 0.612, the single
largest gain". They are fitted by `common.protocol.tune_offsets`, whose docstring
reads *"Per-class logit offsets maximising VALIDATION macro F1"*. The project's
standing instruction is to optimise **per-class precision**, especially ET
precision. F1 spends half its weight on recall, so on the face of it the biggest
lever in the pipeline is aimed at the wrong target.

It is aimed at the wrong target, and it should stay that way.

Run: `python -m experiments.prior_objective`. The offsets are applied to the
network's output logits *after* training, so one training run per split serves
every objective. The arms therefore differ **only** in which offset the identical
probabilities receive — the cleanest pairing available anywhere in this repo.
Merged cohort, multitaper + trajectory (the reported best model), 20 splits.

## Result

| objective | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **macro F1 (current)** | 0.639 | 0.655 | **0.685** | **0.660** | **0.593** | **0.068** |
| macro precision | 0.707 | 0.528 | 0.598 | 0.611 | 0.505 | 0.149 |
| macro P, guarded | 0.689 | 0.622 | 0.648 | 0.653 | 0.570 | 0.070 |
| 0.5·(macroP + macroF1) | 0.669 | 0.636 | 0.683 | 0.663 | 0.572 | 0.073 |
| balanced accuracy | 0.639 | 0.648 | 0.449 | 0.579 | 0.548 | 0.090 |
| macro P, guarded, 21×21 grid | 0.685 | 0.628 | 0.630 | 0.648 | 0.575 | 0.066 |
| macro F1, 21×21 grid | 0.636 | 0.662 | 0.633 | 0.644 | 0.582 | 0.068 |

paired against the current objective:

| arm | precET | macroP |
|---|---|---|
| macro precision | −0.087 [−0.235, +0.041] | −0.049 [−0.127, +0.016] |
| macro P, guarded | −0.036 [−0.109, +0.016] | −0.007 [−0.037, +0.015] |
| 0.5·(macroP + macroF1) | −0.002 [−0.077, +0.063] | +0.003 [−0.028, +0.026] |
| balanced accuracy | **−0.236 [−0.340, −0.142]** * | **−0.081 [−0.124, −0.043]** * |
| macro F1, 21×21 grid | **−0.052 [−0.113, −0.001]** * | −0.016 [−0.038, +0.002] |

**Nothing beats the current objective.** The best alternative,
0.5·(macroP + macroF1), is +0.003 macroP — indistinguishable.

## Why optimising precision directly fails

Look at the standard deviation column. Tuning for macro precision more than
**doubles** the split-to-split spread of macro precision itself, 0.068 → 0.149,
while lowering its mean. That is the signature of overfitting the objective:
the validation split holds roughly 11 ET patients, and precision computed on 11
patients is a very noisy quantity to maximise over a 2-D offset grid. The offset
that looks best on those 11 does not generalise to the test fold.

F1's recall term acts as a **regulariser** on the offset search. It penalises the
degenerate direction — pushing the ET threshold up until only a handful of very
confident patients are predicted ET — that pure precision actively rewards.

The `guarded` arm tests exactly that explanation by forbidding the degenerate
corner directly (each class must be predicted at least half as often as its
validation prevalence). It recovers most of the loss (macroP −0.049 → −0.007) and
brings the spread back to normal (0.149 → 0.070), which confirms the mechanism.
It still does not *beat* macro F1, because the remaining gap is ordinary
small-validation-set noise rather than a wrong corner.

## The grid is not too coarse either

The existing grid is 9×9 over [−1, 1], so 0.25 in logit units. Refining it to
21×21 costs nothing here, since no retraining is involved — and it makes things
slightly **worse** (macro F1: precET −0.052 *, macroP −0.016). The coarse grid is
regularising for the same reason: fewer candidate offsets means fewer chances to
fit validation noise.

## Direction matters, and it is not symmetric

`balanced accuracy` — the recall-only counterpart — is catastrophic for ET
precision (−0.236 *). Moving the objective toward recall hurts far more than
moving it toward precision. The current F1 setting sits close to the optimum of a
distinctly asymmetric curve.

## Standing

* **Do not change `tune_offsets`.** Macro F1 is not a bug; it is doing regularisation
  work that the target metric cannot do for itself at this validation size.
* **Do not refine the offset grid.** 9×9 over [−1, 1] is better than 21×21.
* This is now the third instance in this project of the same pattern: the obvious
  improvement to a small-data pipeline is *more* fitting, and more fitting is what
  the data cannot support. The others are feature unions (13 instances) and
  fine-tuning a pretrained encoder at ≤28 minority patients.
* Worth re-testing only if the validation split grows — a larger ET count would
  reverse the argument, since the objective mismatch is real and only the variance
  is stopping it from paying.
