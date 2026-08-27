# Routing contested patients elsewhere does nothing the same fusion does not

## The setup

`contested_specialists.md` established that the 40 % of patients the deep
ensemble argues about are **not** a random-label region: a logistic regression on
hand-built descriptors reaches 0.520 balanced accuracy there against 0.333
chance, +0.187 [+0.121, +0.253]. Six of eight feature blocks clear chance
significantly.

That licensed one specific architecture: a **gate**. Predict with the deep
ensemble where its six members agree, and with something else where they do not.
The gate is deployable rather than an oracle — unanimity is computed from the
members' own outputs, no labels, available for a single unseen patient.

20 splits, paired, mixing weight and priors both chosen on the untouched
validation split. `python -m experiments.contested_gating`.

## Result

| arm | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| **baseline (deep)** | 0.639 | 0.655 | **0.685** | **0.660** | 0.593 |
| LR everywhere | 0.663 | 0.669 | 0.399 | 0.577 | 0.550 |
| uniform fusion | 0.644 | 0.651 | 0.635 | 0.643 | 0.590 |
| gated hard | 0.656 | **0.670** | 0.589 | 0.638 | **0.603** |
| gated fusion | 0.637 | 0.664 | 0.636 | 0.646 | 0.598 |

paired against the reported deep model:

| arm | precET | macroP |
|---|---|---|
| LR everywhere | **−0.285 [−0.384, −0.188]** * | **−0.083 [−0.124, −0.041]** * |
| uniform fusion | −0.050 [−0.129, +0.024] | −0.017 [−0.048, +0.014] |
| gated hard | **−0.096 [−0.158, −0.042]** * | −0.022 [−0.047, +0.002] |
| gated fusion | **−0.049 [−0.105, −0.006]** * | −0.014 [−0.036, +0.002] |

**The arm that decides it — gated fusion vs uniform fusion:**

| | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| gated − uniform | −0.007 [−0.021, +0.005] | +0.013 [−0.001, +0.027] | **+0.001 [−0.066, +0.063]** | **+0.002 [−0.025, +0.025]** | +0.008 [−0.005, +0.020] |

Null on every column, and precET is +0.001 — as close to exactly nothing as this
protocol can measure.

## Reading it

**The gate is decorative.** Routing contested patients to a different model does
precisely as well as mixing the two models on *every* patient, ignoring the gate
entirely. Whatever small effect fusion has here, the contested/unanimous
distinction contributes none of it.

That is the outcome the control existed to detect, and it settles the question
`contested_specialists.md` deliberately left open. The LR's above-chance
performance on the contested subset is **not complementary signal** — it is the
same signal the deep model already has, showing up in a model that was not
selected against on that subset. Both models are above chance there; neither
knows anything the other does not.

`contested_specialists.md` explicitly refused to claim the LR beat the deep model
on that subset, on the grounds that conditioning on the first model's uncertainty
biases the comparison. **This confirms that refusal was correct.** Had the claim
been made, this experiment would have retracted it.

## Two details worth keeping

**Hard routing is significantly harmful to ET** (precET −0.096 *), and the reason
is visible one row up: the LR alone has precET 0.399 against the deep model's
0.685. Handing it 40 % of the patients outright hands it the ET decisions too.

**The gate helps the majority classes slightly and costs ET.** gated hard is
precN +0.017, precPD +0.014, macroF1 +0.010 — none significant — while precET
falls 0.096. The descriptor LR is a reasonable N/PD model and a poor ET model, so
any routing to it trades the scarce class for the abundant ones. That is the
wrong direction for this project's objective.

**The gate itself transfers fine.** Contested rate is 0.413 on validation and
0.392 on test, so the mechanism is stable across splits — it fails on merit, not
because the gate is unmeasurable at prediction time.

## Standing

* **Do not gate on ensemble disagreement.** Null against the matched fusion
  control that isolates the gate, and its hard variant significantly hurts ET.
* **Do not fuse the descriptor LR into the deep model**, gated or uniform. Both
  trend negative on precET and macroP.
* **The residual signal in the contested set is not complementary.** This closes
  the line `ensemble_diversity.md` opened: the contested region has structure,
  but nothing tried so far sees structure there that the deep model does not
  already see.
* Taken with `pooling_rules.md`, `balanced_bagging.md` and `one_vs_rest.md`, the
  whole family of *better uses of the current representation* is now closed —
  combination rule, ensemble size, member diversity, class decomposition and
  conditional routing are all null or harmful. What remains is a representation
  that separates the contested patients, not a better rule over this one.
