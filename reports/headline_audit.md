# Historical headline audit: split sensitivity before preprocessing fixes

> **Historical report.** Tables below predate the corrected frequency axis,
> descriptors and trajectory. Current point estimates are documented in
> `descriptor_trajectory_fix.md` (ET precision 0.654, macro precision 0.652).
> All intervals below resample splits and describe split sensitivity; they do
> not establish population uncertainty or account for historical model selection.
> The updated script exports patient predictions and computes conditional
> patient-bootstrap intervals on 40 splits. **That updated run is pending.**
> See `patient_level_ci.md` for the method and limitations. Later trajectory fixes
> removed the earlier component significance, so the historical claims below
> must not be used as current conclusions.

## Why it needed checking

The reported merged result — **macro precision +0.041 [+0.014, +0.067]** for
multitaper + IF trajectory over the welch baseline — was measured at 20 splits,
before invariant 6 existed. That invariant records that 20 splits resolves about
0.04, and that a paired **+0.021 [−0.006, +0.048] became +0.005 [−0.020, +0.028]**
on doubling (`early_fusion_confirm.md`).

+0.041 sits at that resolution limit with a lower bound of +0.014, and everything
in the paper rests on it. `kinetic_task_audit.md` had just shown what happens to a
load-bearing claim that predates the machinery to test it, so this one could not
be left unchecked.

Run: `python -m experiments.headline_audit`. Three arms, 40 shared splits,
otherwise the reported protocol exactly.

## Result — it holds

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| welch + desc + asym (baseline) | 0.638 | 0.636 | 0.566 | 0.613 | 0.570 |
| multitaper + desc + asym | 0.638 | 0.668 | 0.595 | 0.634 | 0.583 |
| **multitaper + trajectory (reported)** | 0.651 | 0.654 | **0.663** | **0.656** | 0.595 |

**THE HEADLINE — reported model vs welch baseline:**

| | 20 splits (reported) | **40 splits** |
|---|---|---|
| macroP | +0.041 [+0.014, +0.067] * | **+0.043 [+0.024, +0.062] *** |
| precET | — | **+0.097 [+0.047, +0.146] *** |
| macroF1 | — | **+0.025 [+0.008, +0.041] *** |

The point estimate is essentially unchanged (+0.043 against +0.041) and the
interval is **narrower**, not wider — the opposite of what happened to early
fusion. Split-level win rate for macro precision is **0.75**, and 0.78 for macro
F1, so the difference holds broadly rather than being carried by a minority of
splits.

## Both ranked components survive too

**Trajectory contribution** (reported as +0.056 precET):

    precET  +0.068 [+0.030, +0.110] *
    macroP  +0.022 [+0.010, +0.037] *

Larger than reported, and significant. This is the component the ranked list puts
third.

**Transform contribution alone** (multitaper over welch):

    precPD  +0.032 [+0.009, +0.056] *
    macroP  +0.020 [+0.002, +0.039] *
    precET  +0.029 [−0.020, +0.075]

Significant on macro precision and PD precision; the ET gain from the transform
alone is not significant, which is consistent with `band_truncation.md` finding
the multitaper advantage to be partly band coverage.

The two components sum almost exactly to the whole: +0.020 (transform) + +0.022
(trajectory) = +0.042 against the measured +0.043 headline.

## One number to correct for the paper

Absolute ET precision reads **0.685 at 20 splits and 0.663 at 40**. That is not a
discrepancy — precET has sd 0.183 across splits, so the 20-split standard error is
0.041 and the two agree comfortably. But the more precise figure is 0.663, and
**0.685 should not be quoted bare**. Macro precision is better behaved: 0.660 at
20, 0.656 at 40, sd 0.065.

Historical recommendation, superseded (do not quote as current):

    macro precision  0.656   +0.043 [+0.024, +0.062] over the welch baseline
    ET precision     0.663   +0.097 [+0.047, +0.146]

both at 40 splits, with the baseline at 0.613 / 0.566.

## Why this matters beyond the number

Many claims audited in this session dissolved — the kinetic-task lever, the
"below chance on 2015" note, the SSL gain, early fusion, four mechanistic
explanations, and a prediction built from measured sub-component gains. **This one
does not.** It is the project's central result, it was tested at double the split
count against the same protocol that killed the others, and it came back stronger.

That asymmetry is itself worth stating in the paper: the effects that survived
scrutiny are the input representation and the trajectory stream, and both are
signal-processing choices rather than architectural ones.

