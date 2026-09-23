# A transformer as an ensemble *member*: the first gain to survive 40 splits

## What was tested

Architecture as a **replacement** is the most closed axis in this project — 33
parameters to 85.8 M, 15+ families, and the measured ordering says smaller is
better. But every *ensemble* variation added **more of the same**: 6 seeds
instead of 3 (null), balanced bagging (null), 7 pooling rules (all within
±0.012). The six members are two families × three seeds correlating at
**r = 0.859** on p(ET).

**Architecture as diversity had never been tested.** Ensemble gains come from
decorrelation, so a member that is individually *worse* but differently wrong is
what a correlated ensemble wants.

## Result — 40 splits

| arm | members | precN | precPD | precET | macroP | macroF1 | recET | nETpred | sd(precET) |
|---|---|---|---|---|---|---|---|---|---|
| reported | 6 | 0.648 | 0.654 | 0.654 | 0.652 | 0.593 | 0.440 | 7.65 | 0.189 |
| **+ transformer** | 9 | 0.650 | 0.658 | **0.694** | **0.667** | **0.609** | 0.448 | 6.88 | 0.182 |
| + bag-of-frames | 9 | 0.647 | 0.641 | 0.596 | 0.628 | 0.582 | 0.415 | 7.23 | 0.181 |
| + both | 12 | 0.644 | 0.650 | 0.660 | 0.651 | 0.604 | 0.457 | 7.38 | 0.162 |
| CONTROL + seeds | 9 | 0.649 | 0.648 | 0.662 | 0.653 | 0.592 | 0.430 | 7.28 | 0.188 |

**Adoption — vs the reported model:**

| | precET | macroP | macroF1 |
|---|---|---|---|
| **+ transformer** | **+0.040 [+0.004, +0.074]** \* | **+0.016 [+0.000, +0.030]** \* | **+0.016 [+0.003, +0.029]** \* |
| + bag-of-frames | **−0.058 [−0.119, −0.002]** \* | **−0.024 [−0.046, −0.004]** \* | −0.011 |

**Attribution — vs the matched seed control**, which adds members without
diversity:

| | precET | macroP | macroF1 |
|---|---|---|---|
| **+ transformer** | +0.032 [+0.000, +0.067] \* | +0.014 [−0.000, +0.029] | **+0.017 [+0.004, +0.030]** \* |
| + bag-of-frames | **−0.066 [−0.136, −0.005]** \* | **−0.025 [−0.049, −0.004]** \* | −0.010 |

## It survived doubling — the first thing here that has

At 20 splits the arm read precET +0.067 \*, macroP +0.029 \*. At 40 it reads
**+0.040 \* and +0.016 \***. The effect **halved** but stayed significant.

That matters because **four ~0.05 effects have evaporated on doubling in this
project**, including one the same week that went precET +0.069 \* → +0.031 n.s.
This is the first candidate to clear the project's own standard: 40 splits, a
matched control, and win rates reported.

## What the control does and does not license

**Half the adoption gain is ensemble size.** The seed control — three more seeds
of the existing families — reaches precET 0.662 on its own against the reported
0.654.

Against that control the transformer arm keeps:

* **macroF1 +0.017 [+0.004, +0.030] \***, win 0.57 — the most robust column,
  significant on both adoption and attribution;
* **precET +0.032 [+0.000, +0.067] \***, but the interval's lower bound is
  *exactly* zero and the **win rate is 0.45** — below half. On this project's
  own invariant 5 that is the pattern of a few favourable folds, not a method;
* **macroP +0.014 [−0.000, +0.029]**, null.

**So the defensible claim is macroF1, and precET should be quoted as suggestive
rather than established.** Adopting the arm is justified; attributing the
precision specifically to architectural diversity is not.

## The conservatism prediction, scored

Recorded before the run: *the transformer's variance drop is conservatism.*

**Half right, and the control is what separated the halves.** Against the
reported model the arm predicts ET less often (20 splits: nETpred −2.000 \*).
Against the seed control that shift is **null** (−0.400 [−1.275, +0.400]) — so
the conservatism comes from **adding members at all**, not from the transformer.

The quantisation-floor reading also held: sd(precET) falls monotonically with
member count — 0.189, 0.188, 0.182, 0.162 — tracking ensemble size rather than
architecture, exactly as `1/nETpred = 0.112` predicts.

## The bag-of-frames member is harmful, confirmed

precET **−0.058 \*** vs the reported model and **−0.066 \*** vs the control;
macroP −0.024 \* and −0.025 \*. It is worse than adding nothing and worse than
adding seeds. **Drop it.** Combining it with the transformer (`+ both`) cancels
the gain entirely (macroP −0.001 vs the control), which is the union-dilution
pattern this project has seen 16 times.

## Standing

* **Adopt `+ transformer`: 9 members = the current 6 + 3 `SpectrumTransformer`
  seeds.** Reported figures become **precET 0.694 / macroP 0.667 / macroF1
  0.609** at 40 splits.
* **Quote macroF1 as the established gain** (+0.016 adoption, +0.017
  attribution, both \*). precET +0.040 \* is real against the baseline but its
  attribution carries a sub-0.5 win rate.
* **Do not adopt the bag-of-frames member**, and do not combine the two.
* **Watch for further shrinkage.** The effect halved from 20 to 40 splits. If it
  is ever re-run at 80, expect it smaller again; the honest framing is "a small
  gain from a larger, more diverse ensemble", not a headline number.
* **This does not touch the ceiling.** precET 0.694 is still far from 0.90, and
  the label-noise bound is unaffected by anything in this report.
