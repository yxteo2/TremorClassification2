# Balanced bagging changes nothing, exactly as predicted in advance

## The idea and the two measurements behind it

The reported model averages **3 seeds trained on identical data**. The seeds
differ only in weight initialisation, so a stronger ensemble looked available:

1. **Randomly removing majority patients is free.** `prune_training.md` and
   `influence_prune.md` both measured it — dropping 10 random N/PD costs macroP
   −0.002 and +0.004 respectively, neither significant. Majority subsampling is a
   diversity knob at no cost.
2. **No harmful subset exists to find.** Difficulty-based and influence-based
   selection both failed. If no *particular* majority patients are the problem,
   the way to use that headroom is to drop different ones in every member and
   average over the choice.

That is balanced bagging, the standard ensemble for imbalanced data: each member
sees all of the scarce class and a different sample of the abundant ones. Every
bag keeps **all ET** and resamples only N and PD at 70 %.

Merged 3-class protocol, 20 splits, paired. `python -m experiments.balanced_bagging`.

## Result — null on every column

| arm | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **baseline (3 seeds)** | 0.639 | 0.655 | 0.685 | 0.660 | 0.593 | 0.068 |
| 6 seeds, full data | 0.636 | 0.666 | 0.661 | 0.654 | 0.601 | 0.068 |
| 6 bags, 70 % majority | 0.638 | 0.658 | 0.687 | 0.661 | 0.594 | 0.062 |

paired against the reported model:

| arm | precET | macroP |
|---|---|---|
| 6 seeds, full data | −0.024 [−0.102, +0.043] | −0.006 [−0.031, +0.018] |
| 6 bags, 70 % majority | +0.002 [−0.035, +0.048] | +0.001 [−0.015, +0.019] |

**bags vs seeds at the same ensemble size — the control that isolates the
subsampling:**

| | precET | macroP |
|---|---|---|
| bags − seeds | +0.026 [−0.037, +0.102] | +0.007 [−0.015, +0.032] |

Nothing anywhere. Bagging is +0.001 macro precision against the reported model,
and +0.007 against six seeds on the same data — both intervals comfortably
spanning zero.

## This one was predicted, in writing, before the run finished

`pooling_rules.md` and `ensemble_diversity.md` both recorded the prediction
before this result landed, on a specific measured basis: the six members
**already disagree** on 20.5 % of patient pairs (r(p(ET)) = 0.859, 23.7 %
disagreement across the two architectures). Bagging adds more of a kind of
diversity that is already present in quantity and is demonstrably not the binding
constraint.

The constraint is the one `ensemble_diversity.md` identified instead: 40.5 % of
patients sit on the boundary where the ensemble is at 0.443 balanced accuracy.
Adding members that disagree slightly differently does not move a patient off
that boundary. It is worth recording that this prediction held, because several
mechanism-derived predictions in this project have inverted on measurement
(`prune_training.md`, `one_vs_rest.md`) — this one was derived from a
*measurement* of ensemble diversity rather than from a story about why a method
ought to work.

## The second null is the more useful one operationally

**Doubling the ensemble from 3 seeds to 6 buys nothing** — macroP −0.006, and
precET actually −0.024. Ensemble size is not a lever here at all. The reported
model's 3 seeds are not a compromise that more compute would improve; they are
already past the point of return.

## Standing

* **Do not use balanced bagging.** Null on every column, and null against the
  matched seed control that isolates subsampling.
* **Do not raise the seed count.** 6 seeds costs twice the training for macroP
  −0.006. Keep 3.
* **Ensembling as a family is closed here.** Three separate attempts —
  combination rule (`pooling_rules.md`), ensemble size, and member diversity via
  data (this report) — are all null, and the reason is measured rather than
  guessed: the members already disagree plenty, and the patients they disagree
  about are on the boundary.
* Bagging is *free* rather than harmful (macroP +0.001), so it remains available
  if it is ever wanted for another reason, such as reducing per-member training
  cost.
