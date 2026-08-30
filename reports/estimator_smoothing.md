# Estimator smoothing has an interior optimum, and the reported model sits on it

## The question

The reported model's spectral input is multitaper; the baseline it beats is
welch, macroP +0.043 at 40 splits. **Which property of multitaper produced that
gain had never been asked.**

Measured on a pure 6 Hz tone — true bandwidth zero, so any reported width is pure
instrument — the reported input is far blunter than the baseline it beats:

    ar16                        Q ceiling 31.00   (never used as an input)
    welch nperseg 512                     15.00   (the baseline)
    multitaper nw2.5 K4 n256               5.33   (the reported model)
    multitaper nw4 K7                      2.14
    multitaper nw6 K11                     1.36

That inverted the natural hypothesis and produced the recorded prediction:
**performance should rise monotonically as the estimator gets smoother**, since
at 404 patients with 49 ET a low-variance estimate of a blurred peak should beat
a noisy estimate of a sharp one.

The `nw` sweep is what makes this decisive. Welch versus multitaper changes
estimator family *and* smoothing together; varying `nw` holds family, window
length and frame count fixed and moves only the time-bandwidth product. The
reconstruction was asserted bit-exact against `build()`'s own multitaper
(`max|diff| = 0.00e+00`), so the "current" arm is the reported model and not an
approximation of it. 20 splits, paired.

## Result — an inverted U, peaking exactly where the model already is

| estimator | Q ceiling | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| ar16 | 31.00 | **0.658** | 0.604 | 0.626 | 0.629 | 0.579 |
| welch n512 | 15.00 | 0.627 | 0.643 | 0.639 | 0.636 | 0.585 |
| **MT nw2.5 K4 (reported)** | 5.33 | 0.639 | **0.655** | **0.685** | **0.660** | 0.593 |
| MT nw4 K7 | 2.14 | 0.645 | 0.645 | 0.657 | 0.649 | **0.594** |
| MT nw6 K11 | 1.36 | 0.660 | 0.645 | 0.626 | 0.644 | 0.592 |

paired against the reported model:

| arm | precET | macroP |
|---|---|---|
| ar16 | **−0.059 [−0.120, −0.001]** * | **−0.031 [−0.055, −0.008]** * |
| welch n512 | −0.046 [−0.111, +0.010] | **−0.024 [−0.046, −0.003]** * |
| MT nw4 K7 | −0.028 [−0.085, +0.032] | −0.011 [−0.032, +0.010] |
| MT nw6 K11 | −0.059 [−0.134, +0.010] | −0.016 [−0.042, +0.010] |

**Every arm is worse, and the two extremes are significantly worse.** macroP
traces 0.629 → 0.636 → **0.660** → 0.649 → 0.644 across the smoothing axis: an
interior maximum at the current setting, with both directions losing.

## The prediction fails, and so does its opposite

Smoothing is **not** monotone in either direction. Sharper loses (ar16 −0.031 *),
smoother loses (nw6 −0.016), and the optimum sits between them at the value
already in use.

**The script's own summary line misreads this.** It prints
`Spearman(Q ceiling, macroP) = −0.600 (negative = smoother is better, as
predicted)`. A rank correlation is the wrong instrument for a non-monotone
relationship: it is negative only because three of the five arms lie on the
sharp side of the peak. **Read the table, not the Spearman.** The canned line
should be ignored, as should the one in `influence_stable.md` for the same class
of reason.

## What this settles

* **`nw = 2.5` is the optimum**, not an arbitrary default. Both neighbours in
  the sweep are worse, so there is no free improvement one knob turn away — which
  is what the experiment was launched to find.
* **`ar16` is refuted as a spectral input.** It had sat unused in `METHODS` for
  the whole project and was the sharpest estimator available; it is significantly
  worse on macroP (−0.031) and precPD (−0.052), and no better on precET. That
  option is now closed rather than merely untried.
* **The headline's transform gain is confirmed independently and at the right
  size.** Welch against multitaper here, with everything else held fixed, is
  macroP −0.024 [−0.046, −0.003] *. `headline_audit.md` measured the transform
  contribution in isolation as +0.020 [+0.002, +0.039] at 40 splits. Two
  independent constructions agree on both sign and magnitude.
* **But it is not explained by smoothing.** The gain is not "multitaper is
  smoother than welch"; it is that multitaper at nw 2.5 happens to sit at an
  interior optimum that welch is on the wrong side of. Turning multitaper's
  smoothing up past that point loses the gain again.

## An observation, offered as numerology and not a finding

The optimal resolution bandwidth is 2W = 2·nw·fs/nperseg = **1.95 Hz**. The
narrowest class bandwidth measured in this project is **ET at 2.04 Hz**. The best
estimator is the one whose kernel width matches the narrowest structure it has to
resolve, which is what matched-filter reasoning would suggest.

That is one coincidence at one point and it is not evidence. It would become a
claim only if the optimum moved with the target — for example, if a PD-vs-N task
whose narrowest class is 2.48 Hz preferred a correspondingly broader kernel.
Untested.

## Standing

* **Keep multitaper at nw 2.5, K 4, nperseg 256.** It is the measured optimum
  across a 23× range of estimator sharpness.
* **Do not use `ar16`** as a spectral input. Significantly worse.
* **Do not describe the transform gain as a smoothing effect** in any writeup.
  The correct statement is that the estimator's resolution bandwidth has an
  interior optimum near 2 Hz and multitaper at nw 2.5 sits there.
* Registered as failed prediction 15 in `failed_predictions.md`. The smoke test
  on one split gave Spearman **+1.000** in the *opposite* direction to the
  20-split result — the fourth time this session a one-split result has inverted
  or evaporated.
