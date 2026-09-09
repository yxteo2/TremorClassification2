# Attention on the frequency axis: the fair test, and it is null

> **Correction to this report's own framing.** It was written as though small
> attention had never been reported here. It had: `frozen_vit.md` carries the
> same two arms under "Small attention on the current input: also no". What is
> new below is that those numbers were **pre-fix**, and this re-runs them after
> the axis / Q-factor / IF-trajectory corrections — so this is a *confirmation*,
> not a first test, and it is one of the `fragility_audit.md` re-runs done in
> passing. The deltas match closely (pre-fix −0.012 / −0.019 macroP; post-fix
> −0.011 / −0.013), while both baselines moved with the fixes (0.660 → 0.646).

## Why this needed running

Attention had been dismissed here twice, and **neither test was fair to it**:

    BilateralAttention   25-patient NewData, 61-bin raw spectra — at chance
    vit_b_16             85.8 M parameters, random weights (ImageNet
                         proxy-blocked), AUC 0.540 on 58 patients

Neither used the current input (16 log bins + IF trajectory) or the current
cohort (404 patients). `attention_test.py` was written to fix that and then
never run — one of the ten orphans `experiments/INDEX.md` names.

Both models here sit in the **9–35 k parameter band every model on this cohort
has peaked in**, not the 1e6+ of a vision transformer:

    SpectrumTransformer    17.3 k  frequency bins as tokens, sinusoidal
                                   positions — the "where in the band" cue a
                                   plain convolution lacks
    CrossStreamAttention    4.9 k  spectrum tokens attend to the IF trajectory,
                                   so the streams condition on each other
                                   instead of meeting only at the head

`assemble()` was asserted to reproduce `build()` **bit-exactly** on all three
feature blocks before anything was fitted — the assert-first rule from
`fragility_audit.md`, added after `estimator_smoothing` was found carrying a
stale duplicate of the frequency-axis bug.

## Result — 20 splits, paired

| model | params | precN | precPD | precET | macroP | macroF1 | sd(precET) |
|---|---|---|---|---|---|---|---|
| **TwoStreamNet CNN (current)** | **0.7 k** | 0.642 | 0.649 | **0.648** | **0.646** | 0.590 | 0.210 |
| + SpectrumTransformer | 17.3 k | 0.628 | 0.647 | 0.631 | 0.635 | **0.591** | **0.120** |
| CrossStreamAttention | 4.9 k | 0.622 | 0.632 | 0.646 | 0.633 | 0.579 | 0.171 |

| arm | precET | macroP |
|---|---|---|
| SpectrumTransformer | −0.017 [−0.111, +0.068] | −0.011 [−0.046, +0.021] |
| CrossStreamAttention | −0.002 [−0.105, +0.096] | −0.013 [−0.050, +0.023] |

**Null on every column, both slightly negative.** The recorded prediction — *"null
or slightly negative; at 17 k parameters it will not be catastrophic the way the
85 M backbones were"* — **held on both halves**, and the second half is the
informative one: 85.8 M parameters gave AUC 0.540 (chance), while 17.3 k gives
macroP 0.635 against the CNN's 0.646. **Attention is not the problem; scale was.**

## Why it cannot help here, stated as a measurement

The sequence a transformer would attend over is **16 frequency bins**. A 2-layer
kernel-5 convolution has a receptive field of 9 over that, so more than half the
sequence is already visible at every position — there is no long-range structure
for attention to add. The CNN does the job in **0.7 k parameters**; the
transformer needs 25× more to do it slightly worse.

The two standard arguments for attention — that a CNN misses long-range
dependencies, and that a recurrent net forgets the middle of a sequence — are
sound arguments about *long* sequences. They do not describe a 16-bin vector.

## An observation I cannot yet interpret

`SpectrumTransformer` roughly **halves the variance**: sd(precET) 0.210 → 0.120,
sd(macroP) 0.074 → 0.048. That is the same shape as `temporal_stability.md`'s
STAB, recorded there as *"the one null worth 40 splits"* because it halved precET
variance, and variance is the binding measurement constraint in this project.

**But this experiment scored only five columns and did not record `nETpred`**, so
**a lower variance cannot be distinguished from a model that simply predicts ET
less often and more uniformly.** The epoch-selection run demonstrated that trap
directly on the same day: its `steady` arm showed precET +0.036 and looked
promising until `nETpred −2.10 [−4.05, −0.55] *` revealed a threshold shift
rather than a gain.

So this is logged as an observation, not a finding. Anyone pursuing it should
re-run with `recET` and `nETpred` scored; if the variance drop survives with ET
prediction counts unchanged, a variance-reducing ensemble member is worth having
even at a slightly lower mean.

## Standing

* **Do not adopt attention on the frequency axis.** Null, slightly negative,
  25× the parameters.
* **The earlier dismissals were unfair but reached the right answer.** Worth
  saying in a writeup: the fair test at the right scale on the right cohort was
  run, and it agrees.
* **The size story is the useful one.** 85.8 M → chance, 17.3 k → within 0.011 of
  the CNN, 0.7 k → best. Capacity is not the axis; what the model is given is.
* **The variance observation is open**, and blocked only by two missing columns.
