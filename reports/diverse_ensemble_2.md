# A fourth model type for the ensemble: diversity saturates at three

## What was tested

`diverse_ensemble.md` adopted a 9-member ensemble: CNN, TCN and SpectrumTransformer
families, three seeds each. Against a seed-only control, part of that gain was
genuinely *diversity*. This asks whether a **fourth** family keeps paying. The
candidates were chosen to read the input differently *and* to be decent on their
own, because the bag-of-frames member met the first condition but not the second
and was harmful.

## Result — 20-split screen, pooled as mean of family means

| arm | precN | precPD | precET | macroP | macroF1 | recET |
|---|---|---|---|---|---|---|
| **reported9 (adopted)** | 0.656 | 0.655 | **0.715** | **0.675** | 0.611 | 0.460 |
| + logistic regression | 0.648 | 0.654 | 0.673 | 0.658 | 0.600 | 0.415 |
| + CrossStreamAttention | 0.640 | 0.661 | 0.669 | 0.657 | 0.594 | 0.415 |
| CONTROL + 2nd CNN family | 0.662 | 0.647 | 0.708 | 0.673 | 0.612 | 0.445 |

`reported9` reproduces the earlier run's 20-split precET of 0.715 exactly, so the
family-mean pooling is equivalent to the member mean used there.

**Against the adopted model:** logistic regression gives macroP −0.017 n.s. and
recET −0.045 \*. CrossStreamAttention gives macroP −0.019 n.s., **macroF1
−0.017 \*** and recET −0.045 \*. **Against the family-count control:**
CrossStreamAttention gives **precN −0.023 \*** and **macroF1 −0.018 \***, and
logistic regression is null. The control itself is null against the adopted model
(macroP −0.003).

## Reading it

- **Adding members stops helping after the transformer.** 6 → 9 helped; 9 → 12
  does not, whether the fourth family is diverse (logistic regression,
  CrossStreamAttention) or not (a second CNN). The earlier `+ both` arm, which
  added bag-of-frames, showed the same saturation.
- **Being decent on its own is not enough.** Both candidates were individually
  0.62–0.63 macroP and still lowered the ensemble, and both cost ET recall.
- **No winner, so no 40-split confirmation.** A screen that finds nothing needs no
  confirmation run.

## Predictions, scored

*"Diversity saturates: neither candidate beats the control on precET or macroP."*
**Held.** *"Logistic regression is the better of the two, possibly positive on
macroF1."* It was the better of the two, but **not** positive on macroF1 (−0.012
vs control). *"CrossStreamAttention null."* It was slightly worse than null:
significantly negative on macroF1 and precN against the control.

## Standing

**The ensemble axis is closed at nine members / three families.** Keep the adopted
9-member model. Architecture as replacement, ensemble size and ensemble diversity
are now all measured, and the one gain found (+0.040 precET) is what they add up
to.
