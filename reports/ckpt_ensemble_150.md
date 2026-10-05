# Checkpoint ensembling and gradient-disparity stopping on the 2015 transfer model: none beats `ft`

Run: `python -m experiments.ckpt_ensemble_150 check` (asserts the `ft` arm is
bit-identical to `common.protocol.train(pre=...)`, passed on both members), then
`REPS=0-20 python -m experiments.ckpt_ensemble_150` and
`python -m experiments.ckpt_ensemble_150 report` (CPU, one process, ~5
min/repeat). Corrected loader: **150 patients (61 N / 74 PD / 15 ET)**. The `ft`
arm is this run's own baseline on the same partitions.

## Why

`loss_curves_2015.md` showed that fine-tuning validation loss (about 30
patients) is flat after about 40 epochs, so the best-validation-loss
checkpoint is close to a random draw from that plateau. SWA (a weight average)
hurt. Averaging the *predictions* of several near-optimal checkpoints is the
standard cheap fix for a noisy checkpoint choice. Gradient disparity is a
stopping rule that does not use the validation set at all.

## Arms (all from ONE fine-tuning trajectory per member and seed, rules fixed before the run)

| arm | which fine-tuning epochs' softmax outputs are kept |
|---|---|
| `ft` | the best-validation-loss epoch (current recipe) |
| `top5` | mean over the 5 epochs with lowest validation loss (validation and test outputs) |
| `win` | mean over every epoch with validation loss ≤ 1.02 × minimum |
| `gd` | gradient disparity (Forouzesh & Thiran 2021): L2 distance between the class-weighted loss gradients of two stratified random halves of the training fold, averaged over 3 half-splits (fixed seed), eval mode; the epoch at the centre of the 5-epoch window with minimal mean disparity |

Protocol as in `transfer_2015`: 5-fold stratified CV (random_state = rep), 25 % inner validation, PADS capped at 90 per class and redrawn per repeat, `zfit` per domain, two-stream + ResidualTCN × 3 seeds. Each arm's offsets are tuned on its own validation probabilities. Seeds/repeats 0-19.

Chosen epochs (0-based, of 80):

| | two-stream median (IQR) | TCN median (IQR) |
|---|---|---|
| `ft` | 34 (14-67) | 22 (5-56) |
| `gd` | 70 (61-76) | 55 (34-70) |
| `win` size (epochs averaged) | 27 (10-42) | 26 (9-51) |

## Result (20 repeats, 150 patients)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET | sd(precET) | ET preds/rep |
|---|---|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 | 0.104 | 14.0 |
| top5 | 0.725 | 0.717 | 0.319 | 0.587 | 0.576 | 0.612 | 0.460 | 0.118 | 13.4 |
| win | 0.731 | 0.713 | 0.305 | 0.583 | 0.574 | 0.608 | 0.430 | 0.111 | 14.0 |
| gd | 0.716 | 0.728 | 0.296 | 0.580 | 0.574 | 0.595 | 0.440 | 0.098 | 15.1 |

### Paired vs `ft`, repeat-level bootstrap (20 repeats; win rate)

| contrast | precN | precPD | precET | macroP | macroF1 | AUC | top-5 ET |
|---|---|---|---|---|---|---|---|
| top5 − ft | −0.007 [−0.015, +0.000] (0.25) | +0.008 [−0.000, +0.017] (0.70) | −0.011 [−0.041, +0.015] (0.30) | −0.003 [−0.015, +0.006] | −0.005 [−0.014, +0.002] | −0.000 [−0.004, +0.004] | **+0.040 [+0.010, +0.080] \*** (4 up / 16 tie / 0 down) |
| win − ft | −0.001 [−0.008, +0.006] | +0.003 [−0.006, +0.011] | **−0.024 [−0.046, −0.004] \*** (0.25) | −0.007 [−0.016, +0.001] | **−0.007 [−0.014, −0.000] \*** | **−0.005 [−0.010, −0.000] \*** | +0.010 [+0.000, +0.030] (1 / 19 / 0) |
| gd − ft | **−0.016 [−0.029, −0.005] \*** (0.25) | **+0.018 [+0.005, +0.032] \*** (0.75) | −0.033 [−0.072, +0.008] (0.30) | −0.010 [−0.025, +0.004] | −0.007 [−0.020, +0.006] | **−0.017 [−0.034, −0.001] \*** | +0.020 [−0.060, +0.110] (4 / 13 / 3) |

### Paired vs `ft`, patient-level bootstrap (`bootstrap_dir`, 2000 class-stratified patient draws)

| contrast | precN | precPD | precET | macroP | AUC |
|---|---|---|---|---|---|
| top5 − ft | **−0.007 [−0.013, −0.001] \*** | +0.008 [−0.000, +0.015] | −0.011 [−0.037, +0.020] | −0.003 [−0.014, +0.008] | −0.000 [−0.009, +0.006] |
| win − ft | −0.001 [−0.007, +0.005] | +0.003 [−0.005, +0.011] | −0.024 [−0.052, +0.002] | −0.007 [−0.018, +0.003] | −0.005 [−0.013, +0.001] |
| gd − ft | **−0.016 [−0.030, −0.002] \*** | +0.018 [−0.001, +0.037] | −0.033 [−0.103, +0.033] | −0.010 [−0.039, +0.017] | −0.017 [−0.054, +0.015] |

With 150 patients and 15 ET, patient-level resolution is about ±0.03. Every
point estimate here except gd's precET is below that, so the starred
patient-level precN differences (−0.007 and −0.016) are real but small. The
patient bootstrap does not cover macroF1 or top-5 ET.

## Reading

* **Prediction ensembling over the validation-loss plateau does not help.**
  `top5` matches `ft` on AUC (−0.000) and macroP (−0.003) and is slightly worse
  on ET precision (−0.011, n.s.). Its only significant gain is top-5 ET
  precision, +0.040 \* at the repeat level, but that comes from 4 repeats going
  up with 16 ties. The patient-level bootstrap does not cover it, and the
  increase is one more ET in the top five in 4 of 20 repeats, which is well
  below the patient-level resolution. `win` averages about 27 epochs and costs
  ET precision (−0.024 \* repeat-level, [−0.052, +0.002] patient-level), macroF1
  and AUC. This is the same direction as SWA: averaging in late, drifted
  epochs hurts ET whether the average is over weights or over predictions,
  although it hurts less over predictions.
* **Gradient disparity picks late epochs** (two-stream median 70, TCN 55),
  because both halves are trained on and their gradients shrink together. It
  therefore behaves like the "last epoch" arm of `checkpoint_rule_2015`: AUC
  −0.017 \*, precN −0.016 \* (also at the patient level), precET −0.033 n.s.,
  and precPD +0.018 \* as a class trade. A validation-free stopping rule does
  not recover the TCN's early F1 peak (epochs 10-20).
* So the noisy best-validation checkpoint is not costing measurable precision
  that checkpoint averaging could recover. Its noise is already averaged over 3
  seeds × 2 members.

## Verdict

**None adopted; plain `ft` stays.** No arm has a patient-level interval that
excludes 0 in the right direction. `top5`'s one repeat-level gain (top-5 ET
+0.040) comes from ties plus 4 repeats, sits below patient-level resolution,
and comes with patient-level precN −0.007 \*. `win` and `gd` are worse on
ET/AUC. Prediction ensembling (top-k, window) and gradient-disparity stopping
join SWA, WiSE-FT, LP-FT, SAM and label smoothing as closed for the 2015
transfer model.

## Prediction (in the docstring, before the run) and outcome

* top5/win: AUC +0.005 to +0.015. **Failed** (top5 −0.000, win −0.005 \*).
* top5/win: precET |Δ| < 0.03 and not significant. **Held for top5** (−0.011).
  **Failed for win** (−0.024, significant at the repeat level).
* Nothing significant at the patient level. **Failed narrowly**: top5 precN
  −0.007 \*, which is tiny.
* `win` median ≥ 10 epochs. **Held** (27 / 26).
* gd picks late epochs (median ≥ 60). **Held for two-stream (70), failed for
  TCN (55).**
* gd: AUC −0.01 to −0.02 and precET null to negative, not adopted. **Held**
  (−0.017 \*, −0.033 n.s.).

For the coordinator: append to `reports/failed_predictions.md` (I did not edit
existing files).
