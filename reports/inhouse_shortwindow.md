# The short-window gain is PADS-dependent: nothing in-house

## What was asked

`tf_window_length.md` established that a spectrum built as the mean of 0.64 s
STFT frames beats the current multitaper representation on PD-vs-ET, paired over
30 repeats, under both linear and convolutional models:

    PADS    logreg AUC +0.034 * , precET +0.033 * ; CNN AUC +0.016 *
    MERGED  logreg AUC +0.030 * , precET +0.030 * ; CNN precET +0.013 *

**Both of those cohorts contain PADS.** The representation had never been tested
on in-house patients alone — the cohort the clinic actually has, and the one where
every method in this project has failed.

Run: `python -m experiments.inhouse_shortwindow`.

## In-house PD vs ET — neither representation clears the floor

logistic regression, 119 tremor patients, 21 ET, 200 permutations:

| representation | AUC | precPD | precET | macroP | null 95 % | p |
|---|---|---|---|---|---|---|
| multitaper 16 (current) | 0.556 | 0.827 | 0.190 | 0.509 | [0.304, 0.687] | 0.458 |
| short-window mean 16 | 0.557 | 0.847 | **0.286** | 0.566 | [0.321, 0.668] | 0.468 |

**AUC is identical to three decimals (0.556 vs 0.557) and both sit deep inside
their nulls.** The short-window advantage is entirely absent here.

### The precision trap, in one line

ET precision reads 0.190 against 0.286 — a 50 % relative jump, and exactly the
kind of number a paper would quote. It means nothing. The two models rank
patients identically (ΔAUC = 0.001) and **both are indistinguishable from chance**
(p = 0.458, 0.468). What differs is only where the prevalence-quantile threshold
happens to fall on two equally uninformative rankings.

This is the clearest single illustration in the project of why invariant 6 exists.
**Per-class precision is the right target metric, but a precision difference
between two models that are both inside the null is not a result.** Always print
the null next to the precision.

## In-house 3-class, 10 ET per test set (ET prevalence 0.101)

20 repeats, paired on the same test patients, only the spectrum swapped:

| spectrum | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| multitaper (current) | 0.652 | **0.769** | 0.193 | **0.538** | 0.471 |
| short-window | 0.674 | 0.754 | 0.153 | 0.527 | 0.493 |

paired: precN +0.022 [−0.037, +0.074], precPD −0.015 [−0.065, +0.032],
**precET −0.040 [−0.125, +0.037]**, macroP −0.011 [−0.045, +0.020], macroF1
+0.021 [−0.008, +0.049]. **Nothing is significant on any column.**

The multitaper arm reproduces the published in-house baseline exactly —
0.652 / 0.769 / 0.193 / 0.538 against the README's 0.652 / 0.769 / 0.193 / 0.538 —
which is what makes the comparison trustworthy.

## Reading it

* **The short-window representation is PADS-dependent.** It is the best PD-vs-ET
  representation measured on PADS and on the merged cohort, and it is worth
  nothing on in-house patients under either protocol.
* **This is the same pattern as everything else.** `pd_vs_et_transfer.md` found
  descriptors falling from AUC 0.794 within PADS to 0.519 in-house, and
  `own_data_reality_check.md` found PADS training adding nothing to in-house ET.
  A representation tuned on a cohort containing PADS inherits that dependence.
* **In-house PD precision 0.769 remains the strongest in-house figure**, and it
  belongs to the current representation. Nothing displaced it.

## Standing

* Do **not** adopt the short-window spectrum for in-house work — no effect on
  either protocol, and the apparent precET gain in the binary table is a
  threshold artifact between two chance-level models.
* Keep it for **PADS and merged PD-vs-ET**, where it is paired-significant under
  both model families.
* When quoting per-class precision in-house, quote the permutation null beside
  it. At 21 ET the PD-vs-ET null reaches 0.655 and every model measured sits
  inside it, so in-house precision differences describe thresholds, not skill.
