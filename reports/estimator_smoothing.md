# Estimator sharpness: the sweep was computed on a buggy axis, and its conclusion does not survive

> **This report replaces an earlier version.** That version reported an
> **inverted U peaking at the current nw 2.5**, with `ar16 −0.031 *` and
> `welch −0.024 *`. Both of those significant results are **withdrawn**, and so
> is the inverted U. See "What was wrong" below.

## What was wrong

`estimator_smoothing.mt_variant` reimplements the multitaper path rather than
calling `METHODS["multitaper"]`, and its copy still reconstructed the frequency
axis as `np.linspace(0.0, F_MAX, n_freq)` — **the exact 1.05 % stretch fixed in
`transforms.py` on 2026-09-02**. The duplicate was missed, so every multitaper
arm in the original sweep ran on the defective axis for three weeks.

The experiment's own bit-exactness assert caught it on the first re-run, at
**max|diff| = 3.54** against `build()`'s multitaper. That is not rounding. The
assert only fires when someone re-runs the experiment, which is why it sat
undetected; `verify_preprocessing.py` check 42 now tests the duplicate against
the canonical implementation on every verification run.

Containment: `mt_variant` is used only inside this module, and every other
caller of the shared `spec_for` passes the fixed `METHODS["multitaper"]`. One
report was affected — this one.

## The corrected result, 40 splits

| Q ceiling | estimator | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| 31.00 | ar16 | 0.638 | 0.632 | 0.632 | **0.634** | 0.578 |
| 15.00 | welch n512 | 0.646 | 0.643 | 0.661 | 0.650 | 0.579 |
| **5.33** | **MT nw2.5 K4 [reported]** | 0.648 | 0.654 | 0.654 | 0.652 | **0.593** |
| 2.14 | MT nw4 K7 | 0.642 | 0.644 | **0.685** | **0.657** | 0.588 |
| 1.36 | MT nw6 K11 | 0.653 | 0.644 | 0.659 | 0.652 | 0.589 |

**Paired vs the reported model:**

| arm | precET | macroP | macroP win rate |
|---|---|---|---|
| ar16 | −0.022 [−0.072, +0.029] | −0.018 [−0.037, +0.001] | 0.38 |
| welch n512 | +0.007 [−0.037, +0.054] | −0.002 [−0.019, +0.016] | 0.47 |
| MT nw4 K7 | +0.031 [−0.007, +0.071] | +0.005 [−0.009, +0.021] | 0.47 |
| MT nw6 K11 | +0.005 [−0.068, +0.070] | +0.000 [−0.028, +0.025] | 0.65 |

The one nominally significant cell is ar16's precPD, −0.022 [−0.044, −0.000],
whose upper bound touches zero.

## The candidate that did not survive doubling

At **20 splits** the corrected sweep looked like a real improvement:

    MT nw4 K7   precET +0.069 [+0.022, +0.125] *
    MT nw6 K11  macroP +0.019, win rate 0.75

At **40 splits** both collapse: nw4's precET halves to +0.031 and loses
significance; nw6's macroP goes to +0.000. **Nothing beats the reported nw 2.5.**

This is the fourth ~0.05 effect in this project to evaporate on doubling the
splits, and the first where the stricter bar was set *before* the second half
ran rather than after a claim had been made.

## What the sweep actually shows

Not an inverted U, and not "smoother is better" either:

    ar16   (Q 31.00)   macroP 0.634        <- clearly worst
    welch  (Q 15.00)          0.650    ┐
    nw2.5  (Q  5.33)          0.652    │  top four span 0.007,
    nw4    (Q  2.14)          0.657    │  against a resolution of ~0.025
    nw6    (Q  1.36)          0.652    ┘

**Only the sharpest estimator is distinguishable. Everything from welch to nw6
is one flat plateau, inside what 40 splits can resolve.** nw 2.5 sits *on* that
plateau — it is not a peak, and nothing is better.

The script previously printed `Spearman(Q ceiling, macroP)` with a canned
"smoother is better, as predicted". It has now misled twice: at −0.600 over an
inverted U, and at −0.900 over this plateau. A rank correlation ranks and cannot
see magnitude. It now prints the spread beside rho and states what the protocol
resolves.

## Standing

* **Keep nw 2.5.** Not because it is optimal — it is not measurably better than
  welch, nw4 or nw6 — but because nothing beats it and it is the incumbent.
* **Drop the "interior optimum" claim.** The reported recipe's estimator sits on
  a plateau, and the evidence that it was a peak was an artefact of the axis bug.
* **ar16 is the one real finding**: the sharpest estimator is worse, macroP
  −0.018 with a win rate of 0.38. Directionally solid, not significant.
* **Do not re-run this family without a reason.** The plateau is flat enough
  that 40 splits cannot separate four of the five arms; separating them would
  need the ET count from `data_plan.md`, not more splits.
