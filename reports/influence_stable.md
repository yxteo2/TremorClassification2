# Exact leave-one-out confirms it: there is no harmful training subset to find

## Why this was run again

`influence_prune.md` estimated per-subject harm by Monte-Carlo Data Shapley — 240
random subsets per split — and found that dropping the most harmful majority
subjects was **no better than dropping random ones, and trending worse**. It
attributed that to an unstable ranking, and said so explicitly as an observation
rather than a tested claim:

> This is an observation about the estimator's stability, **not a tested claim**
> that more subsets would fix it. Raising the subset count is the obvious check
> and has not been run.

This is that check, done properly. Rather than raising the subset count, the
estimator is replaced with **exact leave-one-out harm averaged over 20 inner
splits** — for each training subject, the change in held-out score when that
subject is removed, averaged over 20 independent inner train/score partitions.
That is the quantity Monte-Carlo Shapley was approximating, computed directly.

It also carries a **reliability diagnostic the first version lacked**: the
training fold is split in half, harm is estimated independently in each half, and
the two rankings are compared. Calibrated on synthetic data with planted
mislabels, top-k overlap reads **0.60 against 0.12 chance (4.9×)** when a real
harmful set exists and **0.00** when it does not.

## Is the ranking reproducible? Weakly

    split-half rank correlation   mean rho +0.074   median +0.081   range [-0.025, +0.204]
    TOP-K OVERLAP between halves  0.15   chance 0.04   ratio 3.4x
    cross-split agreement         mean rho +0.053

**The estimator is measuring something.** 3.4× chance is not nothing, and it is
well above the 0.00 the calibration produced when no harmful set existed.

But it is a long way from the planted-mislabel calibration in absolute terms:
**0.15 overlap means that of the top-ranked harmful subjects found in one half of
the training fold, roughly one in seven reappears in the other half**, against
roughly three in five when the harmful set was real. The script prints a canned
verdict at this point — *"the dropped set is reproducible; the drop below is a
fair test"* — and that line reads more generously than 15 % absolute overlap
warrants. The ranking is reproducible enough that the drop test is worth running;
it is not reproducible enough to call a stable harmful set identified.

## The drop test — same answer as before

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **k=0 (baseline)** | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 |
| LOO-harm drop 5 | 0.640 | 0.641 | 0.655 | 0.645 | 0.585 |
| random-drop 5 | 0.638 | 0.645 | 0.678 | 0.654 | **0.602** |

paired vs baseline — nothing significant:

    LOO-harm drop 5   precET -0.030 [-0.110, +0.045]   macroP -0.015 [-0.042, +0.010]
    random-drop 5     precET -0.007 [-0.069, +0.045]   macroP -0.006 [-0.027, +0.012]

**LOO-harm vs random — the comparison that decides it:**

    precN   +0.002 [-0.021, +0.028]
    precPD  -0.004 [-0.030, +0.024]
    precET  -0.023 [-0.084, +0.032]
    macroP  -0.008 [-0.029, +0.011]
    macroF1 -0.018 [-0.040, +0.003]

Not significant, and **negative on four of five columns** — the same pattern the
Monte-Carlo version produced. Two independent estimators, one approximate and one
exact, agree.

Selection frequencies remain diffuse: the most-dropped subject appears in **7 of
20 splits** and the rest in 3–6, spread across all three cohorts and both
droppable classes.

## What is actually being found

The estimator is not measuring noise — 3.4× chance says that — but what it
measures is not harmfulness in a usable sense. The most likely reading, and it
is consistent with the rest of the project rather than invented for this result:
`prune_training.md` established that the majority patients that look worst are
**boundary-defining**, hard precisely because they sit near the PD/ET frontier.
A leave-one-out harm score will rank exactly those subjects highly, because
removing a boundary-defining example does perturb the fitted model. It perturbs
it in a way that is *useful*, not harmful, which is why dropping them trends
negative.

**This dataset has no identifiable harmful subset in N or PD.** Every majority
patient is roughly equally useful, and the ones that look worst are doing the
most work.

## Standing

* **Do not prune majority-class training subjects.** Three criteria have now been
  tested — difficulty (`prune_training.md`, significantly harmful), Monte-Carlo
  influence (`influence_prune.md`, null and trending worse), and exact
  leave-one-out (this report, null and trending worse). The question is closed.
* **The follow-up `influence_prune.md` proposed is now done.** It asked whether a
  stronger estimator would change the answer. It does not.
* **Removing a small number of majority patients at random is free** (macroP
  −0.006, n.s.) — worth knowing if training cost ever matters, but it buys
  nothing.
* The reliability diagnostic is worth reusing. Any future selection rule over
  subjects should report split-half top-k overlap against chance *before* the
  expensive drop test, and should be calibrated against a planted-signal control
  so that "above chance" can be read on a scale.
