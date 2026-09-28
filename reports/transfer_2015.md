# Transfer learning from PADS lifts 2015 ET precision (OUT task)

**Scope.** 2015 cohort only, OUT only, one action per model (user, 2026-09-27).
PADS StretchHold, also a postural task, is the transfer source.

**Result.** Pretraining the reported recipe on PADS and fine-tuning on 2015
raises 2015 ET precision from **0.249 to 0.333** (+0.084 [+0.035, +0.135] \*,
wins 72 % of 40 repeats), with macroP +0.023 \* and macroF1 +0.022 \*, at a PD
precision cost of −0.040 \*. It is the first method in this repo to move ET
precision significantly on in-house patients. The earlier verdict "PADS
pretrain/finetune is the worst thing tried" (`cohort_strategies.md`, precET
−0.188 \*) was measured on a merged test set containing PADS patients, which
fine-tuning is bound to forget; it does not apply to 2015 patients.

Run: `python -m experiments._inhouse_transfer_diagnostic` (pre-check), then
`TREMOR_DEVICE=cuda REPS=a-b python -m experiments.transfer_2015` in parts and
`python -m experiments.transfer_2015 report`.

## Pre-check

A PADS-fitted linear model applied unchanged to in-house patients (2015 +
NewData): N vs tremor AUC **0.850** (in-house own CV 0.844), PD vs ET 0.589
(null top 0.633). So PADS transfers tremor detection fully and PD-vs-ET weakly.

## Protocol

151 patients (61 N / 75 PD / 15 ET). Stratified 5-fold CV, every patient
predicted once out-of-fold, precision on all 151 at natural prevalence (ET
0.099); 40 partitions, all arms paired on the same partitions. A stratified
25 % validation split inside each training fold drives early stopping and the
logit offsets. PADS capped 90/class, redrawn per repeat, training only. Each
domain z-scored on its own statistics. Reported recipe (two-stream +
ResidualTCN x 3 seeds), GPU.

## Result (40 repeats)

| arm | precN | precPD | precET | macroP | macroF1 | ET preds/rep |
|---|---|---|---|---|---|---|
| scratch (2015 only) | 0.696 | **0.771** | 0.249 | 0.572 | 0.559 | 14.7 |
| pool (PADS in training) | 0.713 | 0.697 | 0.244 | 0.551 | 0.549 | 15.9 |
| **ft** (PADS pretrain → fine-tune) | 0.722 | 0.732 | **0.333** | **0.596** | **0.581** | 13.5 |
| ft_gentle (fine-tune lr 1e-4) | **0.745** | 0.671 | 0.286 | 0.567 | 0.562 | 18.2 |
| ft_NT (pretrain N-vs-tremor only) | 0.690 | 0.731 | 0.062 | 0.494 | 0.503 | 2.5 |
| ft_shuf (pretrain, labels permuted) | 0.674 | 0.741 | 0.211 | 0.542 | 0.529 | 15.1 |

Paired, bootstrap 95 % CI, split win rate:

| contrast | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| ft − scratch | +0.026 \* (0.75) | −0.040 \* (0.25) | **+0.084 [+0.035, +0.135] \*** (0.72) | +0.023 \* (0.65) | +0.022 \* (0.62) |
| ft − ft_shuf | +0.048 \* | −0.009 | **+0.122 \*** (0.82) | +0.054 \* | +0.051 \* |
| pool − scratch | +0.017 \* | −0.075 \* | −0.005 | −0.021 \* | −0.010 |

At 20 repeats the ft precET gain read +0.057 [−0.018, +0.126]; doubling to 40
strengthened it rather than flipping it.

## Reading it

* **Adoption and attribution agree.** ft beats the plain baseline (adoption)
  and beats the permuted-label pretrain by more (attribution): the gain is
  PADS *label* knowledge, not extra optimisation or a better initialisation.
* **How PADS is used matters more than whether.** Pooling PADS into training
  costs PD precision (−0.075 \*) and buys no ET; pretrain-then-fine-tune keeps
  most of PD and gains ET. Fine-tuning re-fits the decision rule to 2015's own
  class boundaries while keeping PADS's features.
* **Pretraining only the axis that transfers is the worst arm.** ft_NT almost
  stops predicting ET (2.5 per repeat): a network pretrained to merge PD and ET
  into "tremor" cannot re-separate them from 12 training ET.
* **Absolute level.** precET 0.333 at prevalence 0.099 is 3.4x chance, but one
  ET prediction in three is still right; ET recall is ~4 of 15. PD precision
  falls to 0.732. Whether ft is the better clinical trade depends on which
  error matters more.

## Exploration: can the transfer recipe be pushed further? (40 repeats)

`python -m experiments.transfer_2015_explore` -- every arm changes one thing
about `ft`, same 40 partitions, predictions recorded in its docstring first.
The `scratch` and `ft` arms reproduce the run above (same-split prediction
agreement 98.7 % / 99.3 %; GPU is not bit-exact) and ft − scratch replicates:
precET **+0.094 [+0.043, +0.146] \***.

| arm | change vs `ft` | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|---|
| scratch | no pretraining | 0.700 | **0.769** | 0.256 | 0.575 | 0.562 |
| **ft** | -- | 0.720 | 0.731 | 0.350 | 0.600 | 0.583 |
| ft_long | fine-tune 200 epochs | 0.721 | 0.734 | 0.328 | 0.594 | 0.580 |
| ft_head | classifier-only fine-tune | 0.728 | 0.679 | 0.256 | 0.554 | 0.552 |
| ft_uncap | all 383 PADS | **0.732** | 0.724 | 0.320 | 0.592 | 0.583 |
| ft_pn | PADS + NewData OUT | 0.705 | 0.731 | 0.335 | 0.591 | 0.575 |
| ens | ft + scratch soft vote | 0.715 | 0.746 | 0.308 | 0.590 | 0.578 |
| **ft_x2** | ft with 6 seeds | 0.718 | 0.740 | **0.370** | **0.610** | **0.589** |

Paired against `ft`:

| arm | precPD | precET | macroP |
|---|---|---|---|
| ft_long | +0.002 | **−0.021 \*** (win 0.40) | −0.006 |
| ft_head | **−0.052 \*** | **−0.094 \*** | **−0.046 \*** |
| ft_uncap | −0.008 | −0.029 | −0.009 |
| ft_pn | +0.000 | −0.014 | −0.010 |
| ft_x2 | +0.009 | +0.021 (win 0.60) | +0.009 (win 0.62) |
| ens − ft_x2 | +0.006 | **−0.063 \*** | **−0.020 \*** |

**The recipe sits at a local optimum on every axis tried.** Fine-tune strength
is bracketed from both sides (gentle / head-only too weak, 200 epochs too
long); more or different pretraining data is null; mixing in the scratch model
costs ET against a size-matched control. Only more seeds points up (ft_x2,
+0.021 precET, +0.009 macroP), and not significantly -- adopt 6 seeds as a
cheap default, not as a demonstrated gain.

## Where the gain lives: ranking, not threshold

From the saved out-of-fold probabilities (no refit; scores pooled across folds,
so calibration differences between fold models add some noise):

| arm | ET-vs-rest AUC | PD-vs-ET AUC | ET precision among the k most ET-like patients, k = 3 / 5 / 10 / 15 |
|---|---|---|---|
| scratch | 0.628 | 0.574 | 0.47 / 0.40 / 0.33 / 0.28 |
| ft | 0.693 | 0.628 | 0.51 / 0.49 / 0.43 / 0.35 |
| ft_x2 | 0.684 | 0.626 | 0.43 / 0.51 / 0.41 / 0.36 |

PD-vs-ET AUC ft − scratch **+0.055 [+0.042, +0.068] \***, win 0.93 -- the most
consistent effect in this study. Transfer improves how ET patients are
*ranked*, so a stricter ET threshold buys precision: flagging only the five most
ET-like patients per 151 reaches ~0.5 ET precision (vs 0.35 at the tuned
offsets) at ~2.5 of 15 ET found. The absolute PD-vs-ET AUC (0.63) is still
modest, and no permutation null has been fitted for it -- the paired
improvement over scratch is the claim, not the absolute level.

## Standing

* **2015 OUT model:** `ft` (PADS StretchHold capped 90/class, pretrain 200
  epochs, fine-tune all weights lr 1e-3 for 80 epochs), 6 seeds per member.
  Keep scratch if PD precision matters more than ET.
* If the clinical use is "flag likely ET for review", use a stricter ET
  threshold on `ft` rather than the macro-F1-tuned offsets.
* Closed on 2015 OUT: fine-tune length, head-only fine-tuning, uncapped PADS,
  NewData in pretraining, ensembling with the scratch model.
* Not yet tried: a separate REST model (never combined with OUT), with PADS
  Relaxed as its pretraining source.
