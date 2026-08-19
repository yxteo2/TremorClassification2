# Early fusion does not beat the reported model — the gain was split noise

**What was measured, and what it turned into.**

| comparison | splits | precET | macroP |
|---|---|---|---|
| early channels vs **late concat**, matched trunk | 20 | +0.091 [−0.003, +0.189] | **+0.036 [+0.004, +0.071]** * |
| early channels vs **reported model** | 20 | +0.054 [−0.024, +0.131] | +0.021 [−0.006, +0.048] |
| early channels vs **reported model** | **40** | **+0.005 [−0.064, +0.069]** | **+0.005 [−0.020, +0.028]** |

Run: `python -m experiments.early_fusion_confirm`. Two arms, 40 shared splits,
otherwise identical to `combined_best.py`.

At 40 splits:

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| A reported model | 0.651 | 0.654 | 0.663 | 0.656 | 0.595 |
| C early input channels | 0.655 | 0.661 | 0.668 | 0.661 | 0.594 |

**The gain collapses from +0.021 to +0.005 on macro precision, and from +0.054 to
+0.005 on ET precision.** C wins on 62 % of splits for macro precision and 47 %
for ET precision — an edge indistinguishable from a coin.

## What actually happened

Both statements are true and they are not in tension:

* **Early fusion beats late concatenation.** Inside a matched residual-TCN trunk
  it is macroP +0.036 [+0.004, +0.071] * (`tcn_fusion.md`), and against the
  log-binned reported model it is +0.047 [+0.013, +0.076] *.
* **It does not beat the reported model.** The reported spectrum stream is
  `Spectrum1DCNN`, not the residual dilated trunk `FusionTCN` uses. That
  architecture evidently already captures what early fusion recovers, so there is
  nothing left for the fusion change to add.

The `tcn_fusion.md` scope note was right to withhold the stronger claim: *"the
honest statement is 'early fusion beats late fusion within a matched trunk', not
'this is a better model than the one reported'."* This confirms that reading.

## Why 20 splits was not enough

precET has sd 0.183 across splits, so the standard error of its mean at 20 splits
is 0.041 — larger than the +0.054 being claimed. Doubling to 40 splits halves the
variance of the estimate and the effect disappears into it.

This is the project's standing warning firing at a case that had already passed
the usual bar: the comparison **was** paired, on shared splits, with a bootstrap
CI. Pairing removes the fold-composition noise the two arms share; it does not
remove the noise in how much *this particular set of 20 folds* happens to favour
one arm. **For a difference near 0.02 on this cohort, 20 splits is not enough
even when paired.**

## A correction the paper needs

Arm A at 20 splits gives precET 0.685 — the reported headline. At 40 splits the
identical model gives **0.663**.

That is not a discrepancy: sd is 0.183, so the 20-split standard error is 0.041
and the two agree comfortably. But it means **the published 0.685 is a
20-split point estimate carrying about ±0.04 of imprecision**, and the more
precise figure is 0.663. Any paper quoting 0.685 should quote the interval with
it, or use the 40-split number.

Macro precision is better behaved — 0.660 at 20 splits, 0.656 at 40, sd 0.065.

## Standing

* **No change tested this round improves the reported model.** Early fusion,
  log-frequency binning, principal-eigenvalue spectra, polarisation spectra, FiLM
  and channel gating are all null or negative against it.
* **Early fusion remains worth knowing about as an architecture fact**: where the
  descriptors meet a convolutional trunk matters more than how many parameters
  the fusion costs, ordered input > per-block modulation > post-pool > classifier.
  It is not a route to a better model *here* because the reported trunk already
  gets there.
* **Raise the split count before believing a difference under ~0.03.** 20 splits
  resolves ~0.04; 40 resolves ~0.025. Cheap relative to the cost of retracting.
