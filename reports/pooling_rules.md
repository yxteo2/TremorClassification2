# How the ensemble is pooled does not matter

## The lever

The reported model trains **two families × three seeds** and combines them with
`np.mean` of the softmax outputs. That arithmetic average was never a decision —
it is the default, and nothing in this project had ever compared it against an
alternative.

There was a specific reason to expect it to be the wrong default. **Precision at
12 % prevalence is read from the very top of the ranking.** Arithmetic pooling is
*permissive*: a single member that is confidently and wrongly sure a patient is
ET can carry the mean past the threshold on its own, because 0.95 pulls an
average much harder than 0.05 pushes it back. Geometric pooling — averaging in
log space — is *vetoing*: one member saying 0.05 drags the product down whatever
the others say. Vetoing costs recall and buys precision, which is the trade this
project wants for ET. Median and trimmed means make the same argument by
discarding the outlier outright.

Calibration is the other half. `tune_offsets` is the single largest measured gain
in the project (ET precision 0.475 → 0.612) and it is fitted on **uncalibrated**
outputs, where members disagree about scale as well as about the patient.

## What makes this the tightest comparison in the repo

Every arm is computed from **the same six fitted models** in each split. No
retraining, no extra seed, no capacity change — the arms differ only in the
arithmetic applied to six probability matrices already in memory. Any difference
is the pooling rule and nothing else, and the whole experiment costs one baseline
run. Each arm gets its own `tune_offsets` on its own pooled validation matrix;
test is never touched.

Merged 3-class protocol, 20 splits, paired. `python -m experiments.pooling_rules`.

## Result

| rule | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **arithmetic (current)** | 0.639 | 0.655 | 0.685 | 0.660 | 0.593 | 0.068 |
| geometric | 0.642 | 0.656 | 0.683 | 0.660 | 0.591 | 0.074 |
| median | 0.644 | 0.653 | 0.679 | 0.659 | 0.585 | 0.058 |
| trimmed mean | 0.634 | 0.653 | **0.693** | 0.660 | 0.587 | 0.066 |
| temperature → arith | 0.638 | **0.665** | 0.661 | 0.655 | 0.589 | 0.061 |
| temperature → geom | 0.647 | 0.651 | 0.689 | **0.662** | **0.602** | 0.067 |
| family weight on val | **0.655** | 0.632 | 0.656 | 0.648 | 0.590 | 0.070 |

paired against arithmetic pooling — **not one interval clears zero**:

| rule | precET | macroP |
|---|---|---|
| geometric | −0.002 [−0.043, +0.035] | +0.000 [−0.009, +0.008] |
| median | −0.006 [−0.057, +0.043] | −0.001 [−0.019, +0.015] |
| trimmed mean | +0.008 [−0.021, +0.038] | +0.000 [−0.011, +0.011] |
| temperature → arith | −0.024 [−0.065, +0.017] | −0.005 [−0.023, +0.011] |
| temperature → geom | +0.004 [−0.025, +0.036] | +0.002 [−0.011, +0.014] |
| family weight on val | −0.029 [−0.097, +0.031] | −0.012 [−0.035, +0.007] |

The largest macro-precision effect anywhere in the table is **0.012**, and it is
negative. Geometric pooling — the arm the mechanism argument predicted would
win — reproduces arithmetic pooling to **three decimal places** on macroP.

## Reading it

Two things are going on, and they are different.

**The temperature arms are absorbed by the priors, by construction.** A scalar
temperature is a global rescaling of the log-probabilities, and `tune_offsets`
re-fits two free logit offsets *afterwards*. Much of what temperature changed,
the offset search simply undoes. This is not a failure of calibration so much as
a statement that the pipeline already contains a calibration step, fitted on the
same split, with more freedom than a single scalar. Worth noting for the writeup:
the fitted temperatures average **0.82** (range 0.40–1.50), so the members are
mildly *under*-confident on average — sharpening was the right direction, and it
still bought nothing.

**Geometric vs arithmetic is not absorbed** — it is not a monotone rescaling, and
it genuinely reorders patients when members disagree. So why does it change
nothing?

The first explanation to reach for is that the members are near-copies: three
seeds within a family differ only in weight initialisation, so perhaps every
pooling rule is averaging over almost nothing. **That explanation is wrong, and
it was measured rather than assumed.** `ensemble_diversity.md` reports mean
pairwise correlation of p(ET) at **0.859**, and the six members disagree on the
argmax for **20.5 % of test patients** (23.7 % across the two architectures).
One patient in five is contested. There is plenty for a pooling rule to reorder.

The surviving explanation is therefore the other one: the contested patients sit
in a region where **no rule can do better than any other**, so reordering them
changes *which* errors are made rather than how many. That is tested directly in
`ensemble_diversity.md` by scoring the pooled prediction separately on unanimous
and contested patients.

**The split-level win rates are the other tell.** Most alternatives win on
*fewer* than half the splits (geometric 0.25 on macroP) while their mean deltas
are ~0.000. Losing slightly more often while averaging to zero is the signature
of **ties**: on many splits the arms produce identical predictions outright.

## Standing

* **Keep the arithmetic mean.** Nothing beats it, and two arms (family weighting,
  temperature → arithmetic) trend worse on ET precision.
* **Do not report a calibration step.** Temperature scaling before the priors is
  a no-op here because the validation-fitted offsets already do that job with
  more freedom.
* **The combination rule is not where the headroom is**, and the reason is not a
  lack of ensemble diversity — the members already disagree on 20 % of patients.
  This *lowers* the prior on `balanced_bagging.py`: bagging adds more of a kind
  of diversity that is already present in quantity and is demonstrably not the
  binding constraint. Recorded here as a prediction before that run finishes.
* The single positive point estimate in the table — trimmed mean, precET +0.008
  [−0.021, +0.038] — is a fifth the width of its own interval and should not be
  quoted as anything.

## Cost of the finding

One baseline training run. Seven decision rules were compared for the price of
the model that was already being trained, because they all read the same six
fitted members. Any future arm that is a pure post-hoc transform of the ensemble
outputs should be added to this file rather than given its own experiment.
