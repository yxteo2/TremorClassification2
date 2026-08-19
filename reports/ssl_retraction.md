# Retraction: the self-supervised "gain" was freezing, not pretraining

**What was claimed.** `experiments/masked_pretrain.py` pretrained a small
encoder to reconstruct masked frequency bins on 3,081 unlabelled recordings, then
attached a linear head for PD-vs-ET. Against a random initialisation it measured,
paired over 10 repeats:

| | precET | macroP | AUC |
|---|---|---|---|
| PADS, frozen | +0.161 [+0.111, +0.218] * | +0.089 [+0.061, +0.120] * | +0.178 * |
| MERGED, frozen | +0.100 [+0.082, +0.120] * | +0.057 [+0.046, +0.068] * | +0.078 * |
| in-house, frozen | +0.095 [+0.033, +0.157] * | +0.058 [+0.020, +0.095] * | +0.082 * |

Significant everywhere, including in-house at 21 ET patients, where nothing else
in this project has produced a significant deep-model ET gain. I described it as
the largest deep-learning improvement measured here.

**It is not a pretraining effect.** The comparison was against a random
initialisation that was **fine-tuned**, while the winning arm was **frozen**. Two
things changed at once, and the one that mattered was freezing.

Run: `python -m experiments.ssl_leakage`. Same encoder, same folds, same
threshold rule, but the baseline is now *random init, frozen* — so pretraining is
the only difference between arms.

## PADS PD-vs-ET, frozen encoder + linear head, 5 repeats × 5 folds

| arm | AUC | precPD | precET | macroP |
|---|---|---|---|---|
| **random init, frozen** | 0.788 | 0.939 | **0.400** | **0.670** |
| SSL full corpus (transductive) | 0.803 | 0.938 | 0.393 | 0.666 |
| SSL, PADS excluded from pretraining | 0.788 | 0.937 | 0.379 | 0.658 |
| SSL, test patients excluded per fold | 0.791 | 0.936 | 0.371 | 0.654 |
| logreg on 10 descriptors | 0.786 | 0.946 | **0.464** | **0.705** |
| logreg on the same spectrum | 0.785 | 0.936 | 0.371 | 0.654 |

paired vs random init, frozen:

| arm | precET | macroP | AUC |
|---|---|---|---|
| SSL full corpus (transductive) | −0.007 [−0.029, +0.014] | −0.004 [−0.016, +0.008] | +0.015 * |
| SSL, PADS excluded | −0.021 [−0.064, +0.000] | −0.012 [−0.035, +0.000] | +0.001 |
| SSL, test patients excluded | **−0.029 [−0.036, −0.014] *** | **−0.016 [−0.020, −0.008] *** | +0.003 |

**A randomly initialised frozen encoder beats every pretrained one.** The
cleanest arm — pretrained with the test fold's patients removed from the
unlabelled corpus entirely — is significantly *worse* than random init on both
precision columns. Pretraining bought a small, real AUC gain in the transductive
arm (+0.015 *) that disappears once the test patients are removed (+0.003, ns),
which is what a leakage effect looks like.

## What the original number actually measured

Random init, **fine-tuned**, was precET 0.229 / AUC 0.625.
Random init, **frozen**, is precET 0.400 / AUC 0.788.

That single change — not training the encoder — is the whole +0.161. It is a
real and useful result, just not the one claimed: at 28 ET patients a 32-unit
encoder cannot be fine-tuned without being destroyed, so the best thing to do
with it is leave it alone. This is the ~28-minority-patient boundary again,
seen from a new angle, and it is consistent with every other capacity finding in
this project.

## And a frozen random encoder is just a linear model

`logreg on the same spectrum` gives precET 0.371 / AUC 0.785 — **identical to
three decimal places** to the SSL arm with test patients excluded, and within
noise of random-init-frozen. A frozen convolutional encoder with a linear head on
top is a fixed random feature map followed by a linear classifier, so it
approximates logistic regression on the input. The numbers say exactly that.

Meanwhile `logreg on 10 descriptors` beats all of them on precision
(0.464 / 0.705), by −0.093 precET * over the best SSL arm. **Ten hand-computed
spectral descriptors remain the best PD-vs-ET model on PADS**, and nothing in
this SSL line displaced them.

## Why I did not catch this before writing it down

The original script's arm list was

```python
(("random init (scratch)", None, False),      # freeze=False
 ("SSL pretrained, fine-tuned", enc_state, False),
 ("SSL pretrained, frozen", enc_state, True))  # freeze=True
```

The frozen arm was added as a variant of the pretrained arm, so its control
inherited `freeze=False` from the row above it without anyone deciding that. The
fine-tuned SSL arm *was* correctly controlled (+0.039 precET vs the same
fine-tuned baseline) — and that is the honest size of the pretraining effect
under the transductive corpus, before the leakage correction takes it to zero.

The general lesson, and the one worth carrying: **when an arm changes two things,
the baseline has to change with it.** A frozen treatment needs a frozen control.

## Standing after this

* Masked-spectrum SSL on ~3,000 unlabelled recordings: **no benefit**, and
  significantly negative on precision once test patients are excluded from
  pretraining. Do not re-try without a genuinely larger or richer unlabelled
  corpus.
* Freezing rather than fine-tuning a small encoder at ≤28 minority patients:
  **large and real**, but it converges to a linear model on the input, so the
  encoder is not earning its place.
## Closing the "but the pipeline was broken" objection

The SSL run also had a pretrain/downstream frequency-bin mismatch
(`reports/band_truncation.md`): the corpus was welch-61 binned to 3.1–12.3 Hz
while the downstream features were multitaper on 3–15 Hz, so bin *j* meant
11.9–12.3 Hz in one and 14.4–15.0 Hz in the other. "SSL failed" and "SSL was fed
the wrong axis" should not be left entangled, so
`python -m experiments.ssl_matched` rebuilds the corpus through the identical
pipeline and adds a cohort-held-out arm. All arms frozen, 10 repeats.

| | AUC | precET | macroP |
|---|---|---|---|
| **PADS** — random init, frozen | 0.789 | 0.393 | 0.666 |
| SSL mismatched corpus | 0.803 | 0.389 | 0.664 |
| SSL matched corpus | 0.804 | **0.414** | **0.677** |
| SSL matched, **PADS held out** | 0.788 | 0.379 | 0.658 |
| **MERGED** — random init, frozen | 0.652 | 0.302 | 0.605 |
| SSL mismatched corpus | 0.676 | 0.308 | 0.609 |
| SSL matched corpus | 0.714 | **0.384** | **0.651** |
| **in-house** — random init, frozen | 0.537 | 0.210 | 0.520 |
| SSL mismatched corpus | 0.564 | 0.229 | 0.532 |
| SSL matched corpus | 0.552 | 0.200 | 0.514 |
| SSL matched, **in-house held out** | 0.583 | 0.229 | 0.532 |

**Matching the pipeline does recover a gain, and it is large on MERGED** —
precET +0.082 [+0.063, +0.102] * over random-frozen, and +0.076 [+0.053, +0.100] *
over the mismatched corpus. So the bin mismatch was genuinely costing a lot, and
the earlier run was not a fair test of matched SSL.

**But the recovered gain is entirely transductive.** Every arm where the
evaluated cohort is removed from the pretraining corpus is flat or negative on
precision:

| held-out arm | precET vs random init, frozen |
|---|---|
| PADS held out (PADS eval) | −0.014 [−0.039, +0.014] |
| in-house held out (in-house eval) | +0.019 [−0.019, +0.052] |
| test patients excluded per fold (PADS eval) | **−0.029 [−0.036, −0.014] *** |

and the MERGED +0.082 has no held-out counterpart by construction — holding out
both cohorts leaves no corpus — so it is transductive throughout.

The consistent reading across both experiments: **masked-spectrum SSL on these
3,081 recordings gives no transferable benefit. What it gives is a transductive
one — it helps on patients whose unlabelled recordings were in the pretraining
corpus, and not otherwise.**

That distinction is not merely pedantic here. Transductive SSL is legitimate in a
deployment where the unlabelled recordings of the people you will classify are
already in hand. It cannot support a claim of generalising to new patients, which
is what a paper would be asserting.
