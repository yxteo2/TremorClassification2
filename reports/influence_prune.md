# No harmful-subject subset exists to remove

Two criteria for dropping majority-class training subjects have now been tested,
and neither improves the model. The second is the one that was actually asked
for, after the first was identified as the wrong question.

## Criterion 1 — difficulty (`prune_training.md`)

Drop the N and PD patients the model finds hardest to classify.

**Significantly worse than doing nothing, and significantly worse than random**:
precET −0.081 [−0.165, −0.009] * vs baseline, and hard-vs-random at k=5 is
precET −0.065 [−0.134, −0.008] *, macroP −0.030 [−0.053, −0.011] *.

The hardest majority patients turned out to be **boundary-defining** — hard
precisely because they sit near the PD/ET frontier, so removing them lets the
boundary drift into ET territory. Hard and *useful*.

## Criterion 2 — influence: subjects whose presence in training hurts

The right question. Not "which subjects are hard" but "which subjects, by being
in the training set, make the resulting model worse". Those are different sets.

Estimated as a Monte-Carlo Data-Shapley approximation: 240 random subsets of the
training fold per split, logistic-regression surrogate, scored on a held-out 30 %
of the training fold. Influence(i) = mean(score | i present) − mean(score | i
absent). Nothing outside the training fold is read; validation stays clean for the
priors; test is never touched; ET is never dropped.

**Validated before use** on synthetic data with 8 deliberately mislabelled
majority subjects: 7 of 8 recovered in the top-16 dropped, zero ET dropped.

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **k=0 (baseline)** | 0.639 | 0.655 | 0.685 | **0.660** | 0.593 |
| influence-drop 5 | 0.631 | 0.656 | 0.662 | 0.650 | 0.584 |
| random-drop 5 | 0.634 | 0.651 | **0.706** | 0.664 | 0.599 |

paired vs baseline — **nothing significant**:

    influence-drop 5   precET −0.023 [−0.095, +0.043]   macroP −0.010 [−0.035, +0.013]
    random-drop 5      precET +0.022 [−0.018, +0.071]   macroP +0.004 [−0.009, +0.018]

**influence vs random at k=5 — the comparison that decides it:**

    precN   −0.003 [−0.021, +0.015]
    precPD  +0.005 [−0.018, +0.031]
    precET  −0.045 [−0.108, +0.005]
    macroP  −0.014 [−0.037, +0.005]
    macroF1 −0.015 [−0.031, +0.001]

Not significant, but negative on four of five columns. **Selecting subjects by
measured harmfulness is no better than picking at random, and trends worse.**

## Why, most likely: the ranking is unstable

The selection frequencies are the diagnostic. Across 20 splits, at k=5 (10 slots
per split, 200 slots total), the single most-dropped subject is chosen in only
**8 of 20 splits**, and almost all the rest appear **2–5 times**:

    idx 109  PD  2015     8/20
    idx 230  PD  PADS     5/20
    idx 238  PD  PADS     5/20
    idx 270  PD  PADS     4/20
    idx 101  PD  2015     3/20
    ...

If a stable set of harmful subjects existed, the same names would recur in nearly
every split. They do not. With ~181 subjects scored from 240 subsets, the
influence estimate is noisy enough that the selection is close to random with
extra steps — which is exactly what the influence-vs-random comparison shows.

This is an observation about the estimator's stability, **not a tested claim**
that more subsets would fix it. Raising the subset count is the obvious check and
has not been run.

## The cohort signal, such as it is

Of the 30 most-dropped subjects: **PADS 17, 2015 7, NewData 6**. PADS is about
48 % of the capped merged cohort, so 57 % is a mild over-representation, and the
top of the list is dominated by **PADS PD** subjects (idx 230, 238, 270, 351,
212). That is the direction the Atypical Parkinsonism contamination would produce
— 20 PADS records labelled parkinsonian are PSP, MSA or vascular parkinsonism —
but the effect is weak and the selection unstable, so it is a hint rather than a
finding.

**If that contamination is worth removing, do it from the manifest, not from a
model-derived score.** `common/extract_pads.py` already parses the exact diagnosis
field. That is a deterministic, checkable removal; this is a noisy estimate of a
proxy.

## Standing

* **Do not prune majority-class training subjects**, by difficulty or by measured
  influence. Difficulty is significantly harmful; influence is no better than
  random and trends worse.
* Removing a small number of majority patients at random is **free** (macroP
  +0.004, precET +0.022, neither significant) — worth knowing if training cost
  ever matters, but it buys nothing.
* The broader reading: **this dataset has no identifiable harmful subset in N or
  PD.** Every majority patient is roughly equally useful to the model, which is
  consistent with the boundary-defining result from criterion 1 — the ones that
  look worst are the ones doing the most work.
* Untested follow-ups, in order of promise: remove the Atypical Parkinsonism
  records by diagnosis rather than by score; raise the subset count to test
  whether influence becomes stable.
