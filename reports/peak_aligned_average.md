# Peak alignment: the mechanism is real, the correction buys nothing

## The setup

> **Correction (post `descriptor_trajectory_fix.md`).** Every `q_factor` figure in this report was computed with the pre-fix `describe()`, whose half-power width spanned every supra-half-max bin rather than the contiguous peak. Under the corrected definition the PADS ET-vs-PD Q gap is **not** 1.45 but ≈ 0 (ET 21.0, PD 22.1, N 22.8), so the "third of the sharpness gap" that averaging destroyed was mostly the definitional artifact — the old Q was tracking secondary spectral content and the absence of a peak, not sharpness. The model-level conclusion (alignment is null) is unaffected; the descriptor-level motivation is withdrawn.

`method_table` normalises each recording's spectrum then takes a plain mean over
the patient's recordings — for PADS, over the left and right wrist. Those spectra
do not peak at the same frequency. Measured across all 383 PADS patients, the
between-wrist peak mismatch has a **median of 0.781 Hz for N and PD and 0.391 Hz
for ET**, against a multitaper bin width of 0.391 Hz. A typical patient's two
spectra peak one to two bins apart.

That costs real, measurable structure. Peak sharpness is the strongest class
contrast in the project, and averaging the two wrists destroys a third of it:

| | PD | ET | ET−PD gap |
|---|---|---|---|
| per-wrist Q | 2.32 | 3.77 | **1.45** |
| after averaging | 1.86 | 2.82 | **0.96** |

The direction is the opposite of the obvious guess — PD is the asymmetric
disease, so averaging "should" blur PD more. It does not: **ET loses more in
absolute terms because it starts sharper**, and a given misalignment costs a
narrow peak more than a broad one.

The fix is a registration step *before* the same uniform mean — shift each
recording's spectrum onto the patient's own median peak, preserving absolute
tremor frequency, which is itself discriminative. **This is not a pooling rule**;
`mil_recordings.md` already refuted those, and the aggregator here is unchanged.

20 splits, paired, on the corrected frequency axis.

## Result — null for adoption

| arm | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **plain mean (current)** | 0.650 | **0.652** | 0.654 | **0.652** | **0.602** | 0.068 |
| peak-aligned | **0.654** | 0.633 | **0.661** | 0.650 | 0.594 | 0.059 |
| random-shift (control) | 0.587 | 0.613 | 0.543 | 0.581 | 0.538 | 0.080 |

**Adoption — aligned vs the plain mean:**

    precET  +0.007 [-0.065, +0.084]
    macroP  -0.002 [-0.035, +0.031]

Nothing. Correcting the misalignment changes nothing the model can use.

**Attribution — aligned vs random-shift:**

    precET  +0.119 [+0.018, +0.226] *
    macroP  +0.069 [+0.032, +0.106] *

Significant on four of five columns. And random-shift is significantly *worse
than the plain mean* (macroP −0.071 [−0.116, −0.026] *, precET −0.112 [−0.221,
−0.002] *).

## The interesting part: the response to misalignment is not linear

Random-shift adds displacement of the same magnitude in random directions, so it
roughly **doubles** the jitter. The three arms therefore sample zero, natural and
double misalignment:

| jitter | macroP |
|---|---|
| 0× (aligned) | 0.650 |
| 1× (plain, current) | 0.652 |
| 2× (random-shift) | 0.581 |

If the model's loss were linear in misalignment, the plain mean would sit halfway
between the other two, at ~0.616. **It does not — it sits level with the fully
aligned arm.** Going from natural jitter to none is free; going from natural to
double costs 0.071 macro precision.

So the model is genuinely sensitive to peak misalignment — that is what the
significant random-shift deficit establishes — but **the jitter this data
actually contains sits below the level where the model starts to lose.** The
preprocessing is fine as it stands, not by luck of an untested choice but now by
measurement.

## The prediction held, and it still did not produce an improvement

The prediction on record was deliberately narrow: *"aligned should beat
random-shift on precET if the mechanism is real"*, explicitly **not** a
prediction that either beats the plain mean. It held (+0.119 [+0.018, +0.226] *).

The caution that motivated that narrowness was `failed_predictions.md` #5 —
*"sub-component gains on this dataset are not evidence about the composite
task"* — and this is another instance of exactly that. A 33 % recovery of the
ET−PD sharpness gap, measured directly and reproducibly at descriptor level,
produces **+0.007 precET** in the model. The descriptor loss is real; the model
does not care about it.

## Standing

* **Keep the plain mean.** Peak alignment is null for adoption (macroP −0.002)
  and adds a resampling step for nothing.
* **The model is sensitive to peak misalignment** — doubling it costs macroP
  −0.071 [−0.116, −0.026] * — but the natural level is already below the knee.
  Worth knowing if a future cohort has noisier bilateral agreement than PADS.
* **A third data point for the sub-component composition rule.** Descriptor-level
  gains at this sample size have now failed to reach the model three separate
  times. Treat any future "this recovers X % of a class contrast" argument as
  motivation to test, never as evidence of a gain.
* Not to be confused with the pooling family: this changed the *registration*,
  not the aggregator, and it is still null. Both are now closed.
