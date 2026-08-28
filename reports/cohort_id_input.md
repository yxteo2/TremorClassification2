# Cohort ID buys N precision, not ET — and a control that was not a control

## Why this was retested

`cohort_strategies.md` tested four ways of combining 2015 / NewData / PADS and
concluded none beat the existing handling. Three arms were significantly worse.
**One was not** — feeding cohort identity to the network scored the only positive
row in that table (precET 0.690 vs 0.639, macroP 0.668 vs 0.649), reported as
"not better" because neither interval cleared zero.

But it was measured at **10 splits on the welch baseline**, not the reported
multitaper + trajectory model. Ten splits gives a precET interval ±0.10 wide
here, which cannot resolve an effect of +0.05 either way. That arm had never been
tested.

20 splits, reported model, paired. `python -m experiments.cohort_id_input`.

## The result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **baseline** | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 |
| + cohort ID | **0.663** | 0.659 | 0.630 | 0.651 | **0.602** |
| + rand ID fixed | 0.643 | 0.654 | 0.634 | 0.643 | 0.595 |
| + rand ID/split | 0.639 | 0.658 | 0.623 | 0.640 | 0.595 |

**Against the valid control** (random ID re-drawn every split), full ensemble:

| | cohort ID − valid random |
|---|---|
| precN | **+0.024 [+0.009, +0.041]** * |
| precPD | +0.001 [−0.009, +0.010] |
| precET | +0.007 [−0.021, +0.033] |
| macroP | +0.011 [−0.000, +0.021] |
| macroF1 | +0.006 [−0.007, +0.020] |

**Cohort information genuinely helps, and it helps the wrong class.** The gain is
real and significant on N precision and absent on ET, which is the class this
project optimises. macroP sits exactly on the boundary at +0.011 [−0.000,
+0.021].

## The arm where the input actually arrives

Only `TwoStreamNet` receives the cohort one-hot; `ResidualTCN` takes the spectrum
alone, so three of six members are unchanged by construction and a full-ensemble
null would be ambiguous. Scored on its own, cohort ID against the valid control:

    precET  +0.030 [-0.052, +0.108]
    macroP  +0.018 [-0.006, +0.044]
    macroF1 +0.023 [+0.005, +0.044] *

Same shape, roughly doubled, still not significant on precision. The dilution
argument holds — the effect is about twice as large where the input arrives — but
it does not turn into an ET result at either scale.

## The methodological finding, which is the more useful half

**A fixed random draw is not a null control, and this experiment's first version
used one.**

The original random-ID arm drew one 3-level label per patient and reused it
across all 20 splits. With the draw held fixed, any chance association between
that label and the class is **constant across splits**, so the paired bootstrap —
this project's standard safeguard — cannot see it. Pairing protects against
split-to-split noise, not against a fixed feature that is quietly informative.

Measured on the actual seed: the draw's association with ET reached **p = 0.051**
(ET rate 0.067 / 0.137 / 0.159 across levels, a 2.4× spread), and only **5.3 %**
of random draws are at least that ET-associated. The control was carrying a weak
ET prior.

The consequence, on the solo arm, vs baseline:

| arm | precET | macroP |
|---|---|---|
| + rand ID **fixed** | **+0.090 [+0.019, +0.168]** * | **+0.036 [+0.012, +0.064]** * |
| + rand ID **per split** | +0.031 [−0.020, +0.081] | +0.011 [−0.005, +0.029] |

Identical information content — none — and the fixed draw is significant on both
columns while the honest one is significant on neither. The direct paired
contrast between them is **+0.025 macroP [−0.001, +0.053]** and **+0.059 precET
[−0.003, +0.130]**, significant on macroF1 only.

**Stated precisely:** the head-to-head contrast is marginal rather than
established, so the strong claim is not that fixed-beats-honest is proven. It is
that a control which cannot be distinguished from a *real* effect at this sample
size is not doing its job — and the three independent signals (a p = 0.051 ET
association in the draw, significance against baseline that the honest control
lacks, and a positive point estimate on four of five columns head-to-head) all
point the same way.

## The prediction failed

Recorded before the run: cohort ID should cut NewData's contested rate more than
2015's, since `ensemble_diversity.md` found NewData contested at 0.573 against
2015's 0.307 and a domain shift would be what a cohort indicator absorbs.

| cohort | baseline | + cohort ID | change |
|---|---|---|---|
| 2015 | 0.325 | 0.323 | −0.002 |
| NewData | 0.514 | 0.505 | −0.009 |
| PADS | 0.409 | 0.401 | −0.007 |

Nothing moves. **Whatever the precN gain is, it is not the model resolving
NewData's domain shift**, and the mechanism that motivated the retest is
unsupported.

## Standing

* **Do not adopt cohort ID for this objective.** It is significant on precN
  (+0.024 over a valid control) and null on precET, and the project optimises ET
  precision.
* **macroP +0.011 [−0.000, +0.021] is the one number worth a higher-powered
  look.** If macro precision ever becomes the target, this arm deserves 40 splits
  the way `headline_audit.md` re-ran the main claim; at 20 it is undecidable.
* **This can never support a transfer claim.** A model given the cohort label
  cannot be applied to an unseen cohort. Legitimate for deployment, since site
  and device are always known; invisible to leave-one-cohort-out.
* **Never use a fixed random draw as a null control in this project.** Re-draw it
  per split. A frozen draw's chance label association is invisible to the paired
  bootstrap by construction, and at 404 patients with 49 ET a 5th-percentile
  draw is enough to manufacture significance.
