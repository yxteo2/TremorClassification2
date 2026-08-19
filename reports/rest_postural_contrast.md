# The rest-tremor axis is not present in this data

**The hypothesis.** `task_averaging.md` showed that averaging rest recordings
into postural ones costs ET precision (−0.104 [−0.170, −0.035]) while gaining
healthy-class precision (+0.047 [+0.009, +0.088]). The natural explanation is the
central clinical fact about these two diseases:

> Parkinson's tremor is a **rest** tremor, damped by voluntary posture.
> Essential tremor is a **postural / kinetic** tremor, minimal at rest.

If that is why averaging hurts, then the fix is obvious: stop averaging the two
conditions and give the model their **difference**. All three cohorts have both
(2015 OUT/REST, NewData OUT/REST, PADS StretchHold/Relaxed), so the contrast is
computable for 398 of 404 patients.

**The hypothesis is wrong for this data, and the reason matters more than the
result.**

Run: `python -m experiments.task_contrast` and
`python -m experiments.amplitude_contrast`.

## The contrast does not point where physiology says it should

Before any model, the within-patient log ratio of postural to rest band power —
the one number a neurologist forms at the bedside — measured on 398 patients:

| class | n | mean | median | sd | fraction > 0 |
|---|---|---|---|---|---|
| N | 165 | +1.002 | +0.978 | 1.289 | 0.830 |
| **PD** | 184 | **+0.837** | +0.957 | 2.178 | **0.739** |
| ET | 49 | +1.567 | +1.264 | 2.435 | 0.816 |

**PD should be negative.** A rest tremor means more power at rest than in
posture. Instead 74 % of PD patients show *more* postural power, with a mean
ratio of e^0.837 ≈ 2.3×. Healthy controls — who have no rest tremor to damp —
show essentially the same thing.

ET is the highest, which is the expected ordering, and PD-vs-ET from this single
number gives **AUC 0.579** (n=233, 49 ET). That is below every feature family
already in use on the merged cohort (spectrum 0.667, ampmod 0.669, stability
0.653) and inside the permutation null for a cohort this size.

N-vs-Tremor from the same number is **AUC 0.499** — nothing, which is at least
coherent: the ratio is about tremor *type*, not tremor *presence*.

## Every contrast feature makes the model worse

Reported model, 20 splits, changing only the descriptor block:

| arm | precN | precPD | precET | macroP |
|---|---|---|---|---|
| postural only (reported) | 0.639 | 0.655 | **0.685** | **0.660** |
| + scalar contrasts (3 numbers) | 0.648 | 0.662 | 0.624 | 0.644 |
| + contrast spectrum (16 bins) | 0.660 | 0.648 | 0.579 | 0.629 |
| contrast **replaces** descriptors | 0.651 | 0.647 | 0.582 | 0.627 |

paired:

| arm | precET | macroP |
|---|---|---|
| + scalar contrasts | **−0.061 [−0.126, −0.009]** * | −0.015 |
| + contrast spectrum | **−0.106 [−0.185, −0.036]** * | **−0.031 [−0.060, −0.005]** * |
| contrast replaces descriptors | **−0.103 [−0.196, −0.020]** * | **−0.033 [−0.068, −0.001]** * |

Replacing rather than appending — the move the repo's feature-union rule predicts
should win — does not rescue it. The information is not there to be extracted.

## Why the rest condition is not a rest condition

Three explanations, all consistent with the numbers and with an earlier finding:

1. **The protocols do not elicit rest tremor.** Clinical rest tremor requires the
   limb fully supported *and* the patient cognitively distracted — counting
   backwards, serial sevens. None of the three protocols does this. An
   unsupported or attentive "relaxed" posture suppresses Parkinsonian rest tremor
   almost as effectively as voluntary posture does.
2. **Medication state.** PD cohorts are typically recorded ON dopaminergic
   medication, which suppresses rest tremor specifically.
3. **It agrees with `reemergent_tremor.md`.** That report found re-emergent
   tremor unmeasurable here — onset latency ≈ 0.000 s for every class. Re-emergence
   is also a rest-tremor phenomenon. **Two independent attempts to measure the
   rest-tremor axis have now both come back empty**, which is much stronger
   evidence about the recordings than either alone.

## The amplitude ratio, recovered from normalisation, is neutral

`amplitude_contrast.py` recomputes the ratio from **un-normalised** band power,
which is how the table above was produced. Added to the reported model:

| arm | precN | precET | macroP |
|---|---|---|---|
| postural only (reported) | 0.639 | **0.685** | 0.660 |
| + log amplitude ratio | 0.649 | 0.664 | 0.659 |
| + ratio and raw levels *(control)* | 0.665 | 0.681 | 0.668 |
| + ratio and shape contrast | 0.667 | 0.628 | 0.648 |

paired: the ratio alone is macroP −0.001 [−0.020, +0.016] and precET −0.021
[−0.072, +0.024] — **neutral**, neither the gain physiology predicts nor the
damage the shape contrasts caused. Nothing reaches significance on macroP or
precET in any arm.

**The pre-registered control fired.** The module docstring said, before the run:
*"log_amp_post and log_amp_rest are NOT scale-invariant across cohorts and should
help much less than their difference; if they help more, the model is reading a
cohort signature, not physiology."* They help more — the only significant effects
in the whole table are precN gains (+0.025 *, +0.028 *) in the two arms carrying
raw absolute power, while the scale-invariant ratio alone gives +0.009 (ns).

Absolute band power differs by cohort (sensors, units, placement) and the cohorts
have very different class mixtures (PADS is 72 % PD), so raw power is a proxy for
cohort, which is a proxy for the class prior. This is the **cohort-ID input
finding reached through a different door** — the repo already records that arm as
"best mean, CI spans zero, sd nearly doubles". Do not adopt it.

## What this is worth


It is a negative result with a clear, actionable cause, and it explains something
the project had only described:

* **It explains why PD-vs-ET is hard here.** The single most discriminating
  clinical sign between these diseases is absent from the recordings. Every model
  is being asked to separate PD from ET using only the *shape* of the postural
  tremor, which is a genuinely harder and more subtle problem.
* **It is a concrete recommendation for data collection.** A rest condition with
  the limb supported and the patient distracted, and OFF-medication recordings
  where ethically possible, would add the axis the current data lacks. That is a
  more specific and better-founded ask than "more ET patients", and it may be
  cheaper.
* **The 3-15 Hz sum-normalisation deletes the amplitude ratio.** Every spectrum in
  the repo is normalised to unit total power, so the raw `log total power`
  contrast is identically 0.000 for every patient. That is correct for
  across-patient comparison, where sensor gain differs, but it also removes the
  within-patient between-condition ratio, where gain cancels exactly.
  `amplitude_contrast.py` recovers it from un-normalised power — which is how the
  table above was computed.

## Standing

Do not re-try rest-vs-postural contrast features on the current cohorts in any
form: raw ratio, per-band contrast, appended or substituted. Re-open only if a
cohort is recorded with a distraction-based rest protocol.
