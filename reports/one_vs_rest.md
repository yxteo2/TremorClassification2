# A dedicated ET detector is worse than ET's column in the softmax

## The idea

Every model in this project is a **3-class softmax**. Its ET logit is fitted
against N and PD simultaneously, sharing trunk features and a single
normalisation across three logits. That looked like a handicap for the scarce
class: the gradient shaping the ET logit is diluted by two majority columns, and
the softmax ties ET's confidence to whatever PD's logit is doing on the same
patient.

One-vs-rest gives ET its own model. The ET detector gets its own trunk, its own
early stopping, its own class weighting, and it trains on **all 404 patients**
(49 vs 355) rather than the 237 tremor patients a PD-vs-ET model sees —
`pd_vs_et_transfer.md` having established PD-vs-ET as the hard sub-problem.

This had never been tried. The project's binary work always split N-vs-tremor
first and then PD-vs-ET; nothing had trained ET against the full remainder.

Merged 3-class protocol, 20 splits, paired. Each arm has the identical budget:
2 families × 3 seeds per head. `python -m experiments.one_vs_rest`.

## Result — it fails, and only on ET

| arm | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **3-class softmax** | 0.639 | 0.655 | **0.685** | **0.660** | **0.593** | 0.068 |
| one-vs-rest | 0.645 | 0.658 | 0.522 | 0.609 | 0.582 | 0.066 |
| blend (w on val) | 0.659 | 0.646 | 0.559 | 0.621 | 0.580 | 0.075 |

paired against the softmax:

| arm | precN | precPD | precET | macroP |
|---|---|---|---|---|
| one-vs-rest | +0.006 [−0.016, +0.028] | +0.003 [−0.026, +0.029] | **−0.162 [−0.249, −0.073]** * | **−0.051 [−0.081, −0.022]** * |
| blend | +0.019 [−0.009, +0.049] | −0.009 [−0.027, +0.009] | **−0.126 [−0.223, −0.047]** * | **−0.039 [−0.072, −0.009]** * |

**N and PD are untouched; ET collapses by 0.162.** The decomposition costs
nothing for the abundant classes and is catastrophic for exactly the class it was
built to help.

## Why — the diagnostic that was built in

The experiment measures the ET detector on its own terms, as a ranker, before any
combination step:

    ET-vs-rest detector AUC   0.750
    softmax's ET column       0.770
    paired difference        -0.020

The dedicated detector is **worse at ranking ET patients** than the ET column of
a model that was never asked to specialise, despite more capacity and the same
data. So this is not a calibration or combination failure — the underlying
score is worse.

The mechanism is the merge. One-vs-rest forces a **single** boundary between ET
and the union of N and PD. But the surface separating ET from PD is not the
surface separating ET from N — PD and ET are both tremor and differ in frequency
structure, while N differs from both by having little tremor at all. The 3-class
softmax keeps those two boundaries separate and lets each be shaped by its own
class. Collapsing them into one negative class throws that structure away, and
it is strictly less expressive.

The apparent handicap — ET's logit being "diluted" by two majority columns — was
the softmax *using* the distinction between those two columns. It was
load-bearing.

## The secondary finding, which is about method rather than tremor

The blend arm chooses its mixing weight on the validation split by macro F1,
exactly as the class priors are chosen. It picked a mean weight of **0.59**,
leaning toward one-vs-rest, and chose pure softmax (w = 0) in **0 of 20 splits**.

**Validation actively preferred the model that is 0.162 worse on ET precision.**
With ~64 validation patients of which ~8 are ET, the split cannot resolve a
difference that large. This is the same failure mode `prior_objective.md`
identified for tuning offsets on macro precision directly, and it is worth
stating in the writeup: validation-split model selection is reliable here for
2-parameter offsets and **not** reliable for choosing between architectures.

Anything selected on this validation split should be treated as unverified until
it has been through the 20-split paired test.

## Standing

* **Keep the 3-class softmax.** One-vs-rest is significantly worse on precET and
  macroP, and its blend is too.
* **Do not decompose the minority class against a merged negative.** The two
  boundaries ET has — against PD and against N — are different surfaces, and
  merging them destroys information the softmax was using.
* **Do not select architectures on the validation split.** It chose the worse
  model in 20 of 20 splits here.
* This is the second time a supposed handicap has turned out to be load-bearing.
  `prune_training.md` found the "hardest" majority patients were
  boundary-defining rather than noise; this finds the softmax's shared
  normalisation is structure rather than dilution. Both predictions were derived
  from plausible mechanism stories and both inverted on measurement.
