# MiniRocket loses to the model it was predicted to beat — and repairs a standing rule

## What was predicted, and why it looked well-founded

`time_domain_deep.md` closed *learned* time-domain models (TCN on the waveform,
macroP −0.034 *; on analytic channels −0.076 *) but explicitly did not close the
time domain. It closed **learning temporal filters from 404 patients**, and said:

> Time-domain information is only reachable here through estimators that do not
> have to be learned from this cohort.

The evidence was catch22 — 22 statistics fixed offline on 93 unrelated datasets —
which *ties* the spectral descriptors on PADS PD-vs-ET (AUC 0.798 vs 0.794) at
half the fold variance. MiniRocket's ~10 000 kernels come from a fixed
dictionary and are likewise not learned, and Donié et al. (*Scientific Reports*
2025) independently select ROCKET for small wearable-PD datasets on exactly that
reasoning.

**Prediction on record: MiniRocket should beat the learned TCN on the same
input.** It ran on the *identical* `patient_tensor` the TCN received, so this was
learned versus unlearned with the input held fixed.

## Result — it lost to the TCN, and badly to the reported model

| arm | precN | precPD | precET | macroP | macroF1 | recET | nETpred |
|---|---|---|---|---|---|---|---|
| **reported model** | 0.642 | 0.647 | 0.640 | **0.643** | 0.588 | 0.475 | 9.0 |
| MiniRocket + ridge | 0.578 | 0.628 | 0.458 | 0.555 | 0.512 | 0.345 | 8.1 |
| MiniRocket + logreg | 0.600 | 0.602 | 0.473 | 0.558 | 0.529 | 0.320 | 6.9 |
| score fusion (w on val) | 0.658 | 0.623 | 0.656 | 0.646 | 0.591 | 0.445 | 7.6 |
| *TCN on this waveform* | — | — | *0.579* | *0.626* | — | — | — |

paired vs the reported model:

| arm | precET | macroP |
|---|---|---|
| MiniRocket + ridge | **−0.182 [−0.299, −0.066]** * | **−0.088 [−0.130, −0.047]** * |
| MiniRocket + logreg | **−0.167 [−0.267, −0.066]** * | **−0.085 [−0.123, −0.044]** * |
| score fusion | +0.016 [−0.025, +0.069] | +0.003 [−0.014, +0.020] |

**The prediction is refuted.** MiniRocket at macroP 0.555–0.558 sits *below* the
learned TCN's 0.626 on the same input — the opposite of the ordering the standing
rule implied.

Two things are worth reading carefully. First, the **recall columns show this is
a real comparison, not the degenerate corner**: MiniRocket predicts ET about as
often as the deep model (8.1 and 6.9 against 9.0) and is simply wrong more often
(recET 0.345 / 0.320 against 0.475). Those columns were added after a 1-split
smoke test scored precET 1.000 from a single ET prediction; without them a
20-split mean could have hidden that. Second, the **fusion arm is null** (+0.003
macroP) with validation choosing weight 0 in **10 of 20 splits** — precisely what
`score_vs_feature_fusion.md` predicts when one member dominates.

## Repairing the rule: "unlearned" is the wrong abstraction

MiniRocket is unlearned and lost, so the standing conclusion needs replacing.
Two candidates made opposite predictions, and `rocket_dimensionality.py` decided
between them with PCA fitted train-only (unsupervised, so selecting *by label*
could not smuggle in the property being tested):

| arm | precET | macroP | macroF1 | recET |
|---|---|---|---|---|
| **PCA 22** | **0.559** | **0.605** | **0.570** | 0.480 |
| PCA 64 | 0.399 | 0.538 | 0.506 | 0.310 |
| PCA 256 | 0.505 | 0.540 | 0.475 | 0.225 |
| full 9 996 | 0.473 | 0.558 | 0.529 | 0.320 |

paired vs the full feature set:

    PCA 22    macroP +0.046 [+0.013, +0.076] *   precET +0.086 [+0.006, +0.155] *
    PCA 64    macroP -0.020 [-0.058, +0.019]     precET -0.074 [-0.165, +0.010]
    PCA 256   macroP -0.018 [-0.064, +0.022]     precET +0.032 [-0.097, +0.144]

**Dimensionality is a real and significant part of it, and the prediction that
the optimum would sit near 22 held exactly.** Cutting 9 996 features to 22 buys
macroP +0.046 * and precET +0.086 *.

But the relationship is **not monotone** — 64 and 256 are *worse* than the full
set — so "fewer dimensions is better" is too simple. The likely reading is that
L2 on 9 996 standardised features acts as heavy shrinkage and partially recovers,
while 64–256 components keep enough high-variance noise directions to hurt
without that shrinkage. That is an explanation of a shape, not a tested claim.

**And dimensionality does not rescue the method.** PCA 22 reaches macroP 0.605
against the reported model's 0.643 and the TCN's 0.626 — better than full
MiniRocket, still worse than both. So the corrected rule needs *both* halves:

> catch22 works here because its 22 statistics are **few** *and* **selected for
> classification**. Being unlearned is necessary but not sufficient: MiniRocket
> is unlearned, and at matched low dimensionality it still trails.

## Standing

* **Do not use MiniRocket/ROCKET as an input here**, alone or fused.
  Significantly worse than the reported model on macroP and precET, worse than
  the learned TCN it was meant to beat, and the fusion is null.
* **Correct the time-domain rule.** Replace "estimators not learned from this
  cohort" with "**few features, selected for classification**". `catch22`
  satisfies both; MiniRocket satisfies neither, and supplying only the
  low-dimensional half recovers a significant but insufficient part.
* **Dimensionality is confirmed as a first-order constraint**, now at its most
  extreme instance yet: 9 996 features on ~260 training patients, where reducing
  to 22 is worth +0.046 macroP *. That is the seventeenth measurement of this
  project's most reproduced finding.
* Registered in `failed_predictions.md`: the MiniRocket prediction failed, the
  dimensionality prediction held.
* Untested and cheap, if anyone returns to this: **supervised** selection of ~22
  MiniRocket features (fitted inside the training fold) would test whether the
  "selected" half can be supplied too, and would complete the repaired rule.
