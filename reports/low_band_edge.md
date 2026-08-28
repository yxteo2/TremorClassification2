# The 3 Hz band edge is not why slow patients are contested

## What was tested and why

`contested_profile.md` found that **lower tremor frequency means more contested,
in every class**, independent of cohort. It recorded two physical accounts, and
this tests the first.

**Note added after this run.** `contested_profile.md` also claimed the effect
could not be class confusion, on the grounds that confusion produces opposing
signs. That claim is now retracted — the class means are monotonically ordered
(N 8.16, PD 7.51, ET 7.04 Hz), so confusion produces the *same* sign for N and
PD and none for ET, which is exactly the measured pattern (rho −0.385 / −0.241 /
−0.051). The measurements in this report are unaffected; the motivation for
testing a physical mechanism is weaker than it appeared.

The analysis grid starts at 3.0 Hz and everything below is discarded by
`interp(..., left=0.0)` — which is where voluntary movement and postural drift
live. A 4 Hz tremor sits one octave from that edge; a 9 Hz tremor sits far from
it. If slow patients are lost because their signal is entangled with drift right
at the boundary, giving the network the sub-3 Hz region should help them
specifically: it can only learn to discount drift if drift is in its input.

Arms: low edge at **3.0 (current) / 2.0 / 1.5 Hz**, 64 grid points and a 15 Hz
top throughout. Only the spectral input changes — descriptors keep their own
3–15 Hz band, the trajectory stream is untouched, and asserts confirm labels and
descriptors are identical across arms. 20 splits, paired.
`python -m experiments.low_band_edge`.

## First, the frequency effect is confirmed at 20 splits

The baseline row of the stratified table is worth reading on its own, because it
restates `contested_profile.md`'s rank correlation as a directly interpretable
rate, on balanced terciles of 135 / 134 / 135 patients:

| mean-frequency tercile | contested rate |
|---|---|
| slow (< 7.37 Hz) | **0.515** |
| mid (7.37–8.22 Hz) | 0.416 |
| fast (> 8.22 Hz) | **0.253** |

**A clean monotone gradient, 2.0× from slow to fast.** Two different instruments
— a within-class Spearman correlation over descriptors, and a stratified rate —
agree. The effect is real and large.

## The result: nothing moves

| low edge | precN | precPD | precET | macroP | macroF1 | sd(macroP) |
|---|---|---|---|---|---|---|
| **3.0 Hz (current)** | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 | 0.068 |
| 2.0 Hz | 0.642 | 0.660 | 0.655 | 0.652 | 0.586 | 0.068 |
| 1.5 Hz | 0.631 | 0.664 | 0.673 | 0.656 | **0.598** | 0.066 |

paired vs the current edge — every interval spans zero:

| arm | precET | macroP |
|---|---|---|
| 2.0 Hz | −0.029 [−0.102, +0.025] | −0.008 [−0.032, +0.012] |
| 1.5 Hz | −0.011 [−0.050, +0.030] | −0.004 [−0.019, +0.013] |

**And the prediction fails outright:**

| edge | slow | mid | fast | slow vs base | fast vs base |
|---|---|---|---|---|---|
| 3.0 Hz | 0.515 | 0.416 | 0.253 | — | — |
| 2.0 Hz | 0.520 | 0.444 | 0.268 | **+0.005** | +0.015 |
| 1.5 Hz | 0.534 | 0.423 | 0.245 | **+0.019** | −0.008 |

The slow tercile's contested rate does not fall. It rises very slightly, and no
differential effect on slow patients appears at either edge. **Band-edge
contamination is not why slow patients are contested.**

## What the null actually tells us

**The model ignores the sub-3 Hz region.** Adding it neither helps nor hurts, at
either edge, on any column. Whatever lives below 3 Hz carries nothing this model
can use — consistent with it being drift and voluntary motion rather than
recoverable tremor. The 3 Hz edge is well placed, and that is now measured rather
than assumed.

**Spectral resolution inside the tremor band is not binding either.** The grid
keeps 64 points whatever the span, so the 1.5 Hz arm has ~14 % coarser resolution
across 3–15 Hz than the baseline. That cost nothing (macroP −0.004). Worth
knowing: this was flagged in advance as the way the experiment might backfire,
and it did not.

**By elimination, the cycle-count account is now the live explanation.** The two
candidates were band-edge contamination and cycle count — a slow oscillation
completes fewer periods in a fixed-length recording, so every frequency and
stability estimate rests on fewer cycles. Widening the band does nothing for
cycle count, so a null here does not touch that account while removing its rival.

This is elimination, not evidence *for* cycle count, and it should be read that
way until tested directly. The test is clean and untried: **truncate every
recording to a fraction of its length and see whether the slow tercile's
contested rate rises faster than the fast tercile's.** If cycle count is the
mechanism, halving the recording should hurt slow patients roughly as much as
halving their frequency would.

## Standing

* **Keep the 3–15 Hz band.** Extending it down to 2.0 or 1.5 Hz is null on every
  column, and does not touch the slow patients it was aimed at.
* **The frequency gradient is confirmed and quantified**: contested rate 0.515 /
  0.416 / 0.253 across mean-frequency terciles, 20 splits, balanced groups.
* **Band-edge contamination is refuted as the mechanism.** Recorded in
  `failed_predictions.md`.
* **The cycle-count follow-up named below is no longer the obvious next step.**
  Two things undercut it. `contested_profile.md`'s non-circularity claim is
  retracted, so the gradient may simply be class confusion; and the spectral
  estimators use a *fixed* analysis window (multitaper `nperseg=128`, welch
  `nperseg=512`), so relative frequency resolution does not depend on recording
  length at all — truncation would test temporal averaging, not resolution.
  `tf_window_length.md` separately measured that **shorter** windows help on
  PADS (AUC 0.825 at 0.64 s against 0.716 at 5.12 s), which is the opposite of
  what a resolution account predicts.
