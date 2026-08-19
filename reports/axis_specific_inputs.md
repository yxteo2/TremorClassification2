# Different inputs for the two decisions: real gains on N and PD, none on ET

**The idea, and where it came from.** `task_averaging.md` measured a clean split
in what the extra recordings do to the reported model:

    precN   +0.047 [+0.009, +0.088] *      more recordings, same question
    precET  -0.104 [-0.170, -0.035] *      the PD-vs-ET contrast is averaged away
    macroP  -0.012                          the two cancel

The extra tasks are not useless and not useful — they are useful for **one** of
the two decisions and harmful for the other. A flat 3-class model must pick one
input and pay the other cost. Splitting the decisions removes that constraint:
give N-vs-Tremor every recording, and give PD-vs-ET the postural condition alone.

Run: `python -m experiments.axis_specific_inputs`. Merged cohort, n=404, 20
splits, composition by the chain rule P(PD) = P(tremor)·P(PD | tremor), with
validation-tuned priors applied to the composed 3-class vector exactly as in the
flat model.

## The control that makes it interpretable

A two-stage hierarchy was tried before and lost (macroP 0.568 vs 0.583), so a
gain here could be the hierarchy rather than the inputs. Arm 2 is that earlier
experiment reproduced inside this one — the identical hierarchy with **postural
inputs at both stages**. Only arm 3 minus arm 2 is attributable to the input
choice.

This is the discipline the SSL retraction was written about: when an arm changes
two things, the baseline has to change with it.

## Result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| flat 3-class, postural (reported) | 0.639 | 0.655 | **0.685** | 0.660 | 0.593 |
| two-stage, postural / postural | 0.663 | 0.647 | 0.645 | 0.652 | 0.596 |
| **two-stage, ALL-tasks / postural** | **0.697** | **0.706** | 0.610 | **0.671** | **0.618** |

**vs the flat 3-class model:**

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| two-stage, postural/postural | +0.023 | −0.008 | −0.039 | −0.008 | +0.003 |
| two-stage, ALL-tasks/postural | **+0.058 [+0.024, +0.092]** * | **+0.050 [+0.008, +0.090]** * | −0.075 [−0.160, +0.007] | +0.011 [−0.017, +0.040] | **+0.025 [+0.001, +0.049]** * |

**vs the same hierarchy with postural inputs — isolating the input choice:**

| | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| ALL-tasks stage A | **+0.035 [+0.003, +0.065]** * | **+0.059 [+0.032, +0.088]** * | −0.035 | +0.019 [−0.002, +0.039] | +0.022 [−0.001, +0.041] |

## Reading it honestly

* **The mechanism works as designed.** The hierarchy alone does nothing
  (macroP −0.008, confirming the earlier negative). Feeding stage A the extra
  recordings is what produces the gains, and it does so significantly on both
  precN (+0.035 *) and precPD (+0.059 *) against the matched hierarchy.
* **macroP 0.671 is the highest merged macro precision measured in this project**,
  above the reported 0.660. **But the paired gain is +0.011 [−0.017, +0.040] and
  is not significant.** The headline number improved; the comparison did not
  clear the bar.
* **ET precision is not improved.** It is −0.075 [−0.160, +0.007] against the flat
  model — not significant, but negative, and the interval is mostly below zero.
  Against the project's standing instruction to optimise ET precision above all,
  **this is not the win that was wanted.**

So the fair summary is: two significant per-class precision gains on the two
larger classes, a macro-precision point estimate that is the best measured but
not significantly better, and no improvement on the class that constrains the
project.

## Why ET does not benefit

Stage B is unchanged between arms 2 and 3 — same postural inputs, same model.
Its ET precision should therefore be roughly constant, and the −0.035 between
those two arms is the composition effect: stage A now passes a different set of
patients through to stage B. A better N-vs-Tremor gate admits more true tremor
patients, which raises the ET denominator without raising the numerator
proportionally.

That is a structural property of cascades, and it means **axis-specific inputs
cannot fix ET precision** — they redistribute where errors fall. Fixing ET still
requires either more ET patients or the missing rest-tremor axis
(`rest_postural_contrast.md`).

## Standing

* Worth adopting **if the objective is macro F1 or per-class precision on N and
  PD**: +0.025 macroF1 *, +0.058 precN *, +0.050 precPD *, at the cost of a
  non-significant ET decline.
* **Not** worth adopting if ET precision is the objective, which it has been.
* The generalisable lesson is the one that survives regardless: **when two
  sub-decisions want different inputs, measure them separately before choosing a
  single input for both.** The flat model's macroP was hiding two effects of
  opposite sign.
