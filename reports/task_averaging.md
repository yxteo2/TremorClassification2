# Averaging the extra tasks helps detection and destroys PD-vs-ET

**The chain.** `mil_recordings.md` found that averaging all of a patient's
recordings instead of the postural task alone improved a **spectrum-only** model
(macroP +0.040, precPD +0.057 *, macroF1 +0.049 *), while learned pooling
(attention, max) was significantly worse than that uniform mean. It flagged a
scope limit: the baseline there was macroP 0.599, not the reported 0.660, and a
gain on a stripped-down model need not survive on the full one.

It does not survive. And the way it fails is more informative than the gain was.

Run: `python -m experiments.alltasks_final`. The reported model (multitaper +
trajectory, soft-voted with `ResidualTCN`, validation-tuned priors), n=404, 20
splits, changing exactly one thing: which recordings the per-patient tables are
averaged over. 1,140 postural recordings against 3,081 across all tasks (2.7×).

## Result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| postural only (reported) | 0.658 | 0.661 | **0.641** | **0.653** | 0.594 |
| ALL tasks, spectrum | **0.705** | 0.681 | 0.537 | 0.641 | 0.604 |
| ALL tasks, spectrum + descriptors | 0.678 | 0.676 | 0.595 | 0.650 | **0.614** |

paired against postural only:

| arm | precN | precET | macroP | macroF1 |
|---|---|---|---|---|
| ALL tasks, spectrum | **+0.047 [+0.009, +0.088]** * | **−0.104 [−0.170, −0.035]** * | −0.012 | +0.010 |
| ALL tasks, spec + desc | +0.020 | −0.046 [−0.128, +0.035] | −0.004 | **+0.020 [+0.003, +0.038]** * |

**The extra recordings significantly improve healthy-class precision and
significantly damage ET precision, netting out to no change in macro precision.**

## Why, and it is the central clinical fact about these two diseases

> Parkinson's tremor is characteristically a **rest** tremor, damped by voluntary
> posture. Essential tremor is a **postural / kinetic** tremor, minimal at rest.

Rest-versus-postural is therefore not one task pair among many — it is the
primary bedside discriminator between PD and ET. Averaging a rest recording into
a postural one:

* **adds signal-to-noise for "is there any tremor at all"** — more recordings,
  same question, so precN rises;
* **destroys the contrast that carries "which tremor"** — the very difference
  between conditions is averaged out, so precET falls.

Both measured effects follow from one mechanism, and they point in opposite
directions on the two axes, which is why macro precision hides it. **This is a
case where the macro average is the wrong summary and the per-class columns are
the result.**

## It resolves a tension rather than creating one

`mil_recordings.md` flagged a conflict with the standing note *"averaging two
PADS tasks | precET 0.585 vs 0.612"*. There is no conflict. PADS's second task
is **Relaxed** — the rest condition. That note was the same effect, measured on
PADS alone two years of experiments earlier, and it was right. The MIL run saw a
net gain only because its stripped-down baseline made the precN term dominate.

The correction belongs to my MIL write-up, not to the older note.

## What it implies for using the discarded data

The pipeline still throws away roughly two thirds of its recordings, and that is
still worth attacking. But **not by averaging**. The result says the extra
conditions carry information *as contrasts*, not as additional samples of the
same quantity. Two follow-ups test that directly:

* `experiments/task_contrast.py` — feed the model the postural-minus-rest
  difference spectrum instead of their mean.
* `experiments/amplitude_contrast.py` — the within-patient power ratio, which
  sum-normalisation had been forcing to exactly 0.000 for every class.

## An aside worth keeping

The `spectrum + descriptors` arm recovers most of the ET loss (−0.104 → −0.046,
no longer significant) while keeping a macroF1 gain. Averaging the *descriptors*
over tasks is less damaging than averaging the *spectrum*, which makes sense:
descriptors are summary statistics that are individually more robust to which
condition produced them, whereas the spectrum's shape is the thing the condition
changes.
