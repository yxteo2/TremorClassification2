# The contested patients are not unclassifiable — signal survives there

## The question

`ensemble_diversity.md` established that the six ensemble members agree on 59.5 %
of patients and are 68.8 % correct there, while on the contested 40.5 % they are
at 48.5 % raw accuracy with a top-2 margin four times narrower. That explains why
every post-hoc decision rule ties, and leaves one question that decides where the
project goes next:

**Are those patients unclassifiable in principle, or only unclassifiable from the
multitaper spectrum?**

Each feature block `final_model.build()` already produces was fitted with a plain
logistic regression and scored on the contested and unanimous subsets separately.
A weak linear model is the right instrument: an LR that clears chance cannot be
memorising, so it is stronger evidence of usable structure than a deep model
would be. 10 splits. `python -m experiments.contested_specialists`.

## Result

Contested fraction 0.405. Deep ensemble on contested: raw accuracy 0.485,
**balanced accuracy 0.443**. Majority-class rate within the contested subset
0.582; balanced-accuracy chance is 0.333 by construction.

| block | bal acc \| contested | vs 1/3 | paired 95 % CI | bal acc \| unanimous† |
|---|---|---|---|---|
| **DESC** | **0.520** | **+0.187** | [+0.121, +0.253] * | 0.645 |
| wavelet_packet | 0.497 | +0.164 | [+0.104, +0.227] * | 0.637 |
| multitaper | 0.461 | +0.127 | [+0.072, +0.197] * | 0.653 |
| welch | 0.448 | +0.114 | [+0.045, +0.198] * | 0.624 |
| STAB | 0.405 | +0.072 | [+0.028, +0.119] * | 0.564 |
| all blocks | 0.385 | +0.052 | [+0.021, +0.081] * | 0.593 |
| ASYM+HAVE | 0.346 | +0.013 | [−0.066, +0.097] | 0.487 |
| TRAJ | 0.303 | −0.030 | [−0.087, +0.032] | 0.412 |

† raw accuracy on the unanimous subset, for context only.

**Six of eight blocks clear chance on the contested subset, significantly.** The
patients the deep ensemble argues about are *not* a random-label region. A
logistic regression on hand-built descriptors reaches 0.520 balanced accuracy
there against 0.333 chance.

## The bias that must be stated before anyone reads this as a win

The obvious reading — "an LR at 0.520 beats the deep ensemble's 0.443 on the same
patients" — is **not valid**, and the reason is structural.

The contested subset is *defined* by the deep ensemble's own disagreement.
Conditioning on that selects patients where the deep model happens to be
uncertain, but applies no such selection to the LR. Any second model will look
comparatively good on a subset carved out by the first model's uncertainty; it is
the same shape as regression to the mean. **The deep-vs-LR comparison on this
subset is confounded and should not be quoted.**

What is *not* confounded is each block against **chance**, because 1/3 does not
depend on which patients were selected. That comparison is what the table's
interval column tests, and it is the finding: **structure remains in the
contested region.** The ceiling there is not a labelling floor.

## Two secondary results worth keeping

**TRAJ contributes nothing on contested patients** — 0.303, below chance, with an
interval spanning zero. The instantaneous-frequency trajectory is one of the two
blocks the reported model adds beyond the spectrum, and on exactly the patients
that decide the ceiling it carries no usable signal on its own.

**This does not contradict the headline, and the two should not be read against
each other.** `headline_audit.md` verifies the trajectory stream at 40 splits and
it survives — precET **+0.068 [+0.030, +0.110]** *, macroP **+0.022 [+0.010,
+0.037]** *. Both statements are true, and together they locate where the
trajectory earns its keep: **on the patients the ensemble is already confident
about, not on the contested boundary.** A component can be significant overall
and useless in the hardest 40 %, and this one is. What follows is only that the
trajectory stream is not the thing that will rescue the contested set — not that
it should be removed, which the audit says clearly it should not.

**"all blocks" (+0.052) is far worse than DESC alone (+0.187)** — the
concatenation of everything underperforms its best member by a wide margin. This
reproduces, on a new subset and with a new instrument, the pattern already on
record across seven hand-built feature unions in this project: at 404 patients
with 49 ET, dimensionality binds harder than information.

## What this licenses, and what it does not

It does **not** license the claim that a logistic regression should replace or
augment the deep model. That requires showing the LR's signal is *complementary*
rather than the same signal the deep model already has — both are above chance
on the contested set (0.443 and 0.520), and two models can both be above chance
while carrying identical information.

It **does** license the specific architecture this measurement suggests, which is
testable and has never been tried here: **a gate**. Predict with the deep
ensemble where its members are unanimous, and with something else where they are
not. Crucially, unanimity is computable at test time from member outputs alone —
no labels — so this is a deployable rule and not an oracle.

That is tested in `contested_gating.md`, with the control that decides it:
**fusing the two models uniformly, ignoring the gate.** If uniform fusion does as
well as gated fusion, the gate is decorative and the effect is fusion.

## Standing

* **The contested 40 % contains real structure** — six of eight blocks clear
  chance there significantly, best +0.187 [+0.121, +0.253].
* **Never compare a second model against the first on a subset the first model's
  uncertainty defined.** Only comparisons to a selection-independent baseline
  (chance) are valid on this subset.
* **TRAJ is not the answer to the contested set** (−0.030, n.s.).
* **Concatenating blocks remains harmful**, now demonstrated on the contested
  subset with a linear model.
