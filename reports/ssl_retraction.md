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
* Note this SSL run also had a pretrain/downstream frequency-bin mismatch
  (`reports/band_truncation.md`): the corpus was welch-61 binned to 3.1–12.3 Hz
  while the downstream features were multitaper on 3–15 Hz, so bin *j* meant
  different frequencies in each. `experiments/ssl_matched.py` closes that,
  because "SSL failed" and "SSL was fed the wrong axis" should not be left
  entangled.
