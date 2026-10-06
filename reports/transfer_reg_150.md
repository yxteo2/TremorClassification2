# Transfer regularisers (Co-Tuning, BSS) on the 2015 transfer model: Co-Tuning hurts ET, BSS does nothing

Run: `python -m experiments.transfer_reg_150 check` (bit-identity), then
`REPS=0-20 python -m experiments.transfer_reg_150` and
`python -m experiments.transfer_reg_150 report` (20 repeats, 150-patient table,
61 N / 74 PD / 15 ET; CPU, one thread, ~4.5 min per repeat for all four arms).
Runs: `transfer_reg_150_runs/rep{00..19}.npz`.

## Why

`training_methods_2015.md`: WiSE-FT, which pulls the weights back toward PADS,
collapsed the top of the ET ranking. That points to PADS's decision boundary
being wrong for 2015 while its features are useful. Two published transfer
regularisers act on features and the source head rather than on the weights:

* **cotune**, Co-Tuning (You et al., NeurIPS 2020). A copy of the pretrained
  PADS classifier is kept as a trainable source head on the shared features
  (the input to the final `nn.Linear`, via a forward hook). Its soft target
  for each patient is the row of the relationship matrix
  P(y_PADS | y_2015) for that patient's true 2015 class. The matrix is
  estimated from the pretrained model's softmax on the fold's 2015
  **training** patients. Loss = CE + 1.0 x soft-CE.
* **bss**, Batch Spectral Shrinkage (Chen et al., NeurIPS 2019). CE + 1e-3 x
  the smallest (k = 1) squared singular value of the full-batch penultimate
  feature matrix.
* **cotune_bss**: both terms.

For each seed, all arms branch from the same PADS-pretrained weights. The
checkpoint rule (lowest class-weighted validation CE of the target head) is
the same in every arm. The protocol is `transfer_2015`. **`ft` was checked
bit-identical to `common.protocol.train(pre=...)`** on rep 0 / fold 0 for both
members (`check`: CHECK_OK). Hyper-parameters (1.0, 1e-3, k = 1) were fixed
before the run.

## Result (20 repeats)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET | ET preds/rep | agree w/ ft |
|---|---|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 | 14.0 | 1.000 |
| cotune | 0.728 | 0.699 | 0.283 | 0.570 | 0.559 | 0.587 | 0.360 | 15.0 | 0.888 |
| bss | 0.727 | 0.712 | 0.331 | 0.590 | 0.578 | 0.611 | 0.430 | 14.3 | 0.980 |
| cotune_bss | 0.723 | 0.700 | 0.295 | 0.573 | 0.562 | 0.587 | 0.350 | 15.1 | 0.885 |

Paired against `ft`. Repeat level: 4000-draw bootstrap of the 20 per-repeat
differences, with the share of repeats won. Patient level: `bootstrap_dir`,
2000 stratified patient draws.

| contrast | metric | repeat level | win | patient level |
|---|---|---|---|---|
| cotune − ft | precET | **−0.046 [−0.073, −0.017] \*** | 0.15 | **[−0.091, −0.007] \*** |
| | macroP | **−0.020 [−0.030, −0.010] \*** | 0.20 | **[−0.038, −0.004] \*** |
| | precPD | −0.010 [−0.020, −0.002] \* | 0.30 | [−0.025, +0.005] |
| | precN | −0.004 [−0.017, +0.008] | 0.45 | [−0.014, +0.007] |
| | PD-vs-ET AUC | −0.026 [−0.036, −0.016] \* | 0.10 | [−0.064, +0.011] |
| | macroF1 | −0.022 [−0.030, −0.014] \* | 0.05 | – |
| | top-5 ET | −0.060 [−0.110, −0.020] \* | 0.00 | – |
| bss − ft | precET | +0.001 [−0.014, +0.022] | 0.20 | [−0.016, +0.022] |
| | macroP | −0.001 [−0.006, +0.007] | 0.35 | [−0.008, +0.008] |
| | precPD | +0.002 [−0.005, +0.011] | 0.35 | [−0.004, +0.009] |
| | precN | −0.005 [−0.013, +0.000] | 0.20 | [−0.010, −0.001] \* |
| | PD-vs-ET AUC | −0.001 [−0.003, +0.000] | 0.45 | [−0.005, +0.002] |
| | macroF1 | −0.003 [−0.008, +0.003] | 0.25 | – |
| | top-5 ET | +0.010 [+0.000, +0.030] | 0.05 | – |
| cotune_bss − ft | precET | −0.034 [−0.063, −0.003] \* | 0.20 | [−0.080, +0.006] |
| | macroP | −0.018 [−0.030, −0.006] \* | 0.20 | [−0.036, −0.001] \* |
| | precPD | −0.009 [−0.020, +0.000] | 0.40 | [−0.024, +0.007] |
| | precN | −0.009 [−0.023, +0.004] | 0.45 | [−0.020, +0.001] |
| | PD-vs-ET AUC | −0.025 [−0.036, −0.015] \* | 0.10 | [−0.064, +0.011] |
| | macroF1 | −0.020 [−0.029, −0.011] \* | 0.15 | – |
| | top-5 ET | −0.070 [−0.120, −0.020] \* | 0.00 | – |

`bootstrap_dir` covers precN / precPD / precET / macroP / AUC only, so macroF1
and top-5 ET have no patient-level interval. The bss win rates are low because
most repeats tie with `ft` (98 % identical predictions); a win is counted only
when bss is strictly ahead.

## Diagnostics

**Relationship matrix**, the mean over repeats, folds and seeds. Rows are
2015 N / PD / ET; columns are PADS N / PD / ET.

| member | 2015 N | 2015 PD | 2015 ET |
|---|---|---|---|
| two-stream | .53 / .35 / .12 | .20 / .45 / .35 | .16 / .38 / .47 |
| TCN | .62 / .30 / .09 | .29 / .40 / .32 | .23 / .36 / .41 |

The PADS head sees 2015 PD and 2015 ET as almost the same mixture: the rows
differ by about 0.1 in each column. Co-Tuning's soft targets therefore ask the
shared features to make PD and ET look alike to the source head. That fits the
measured AUC drop (−0.026 \*) and the loss at the top of the ET ranking. It is a
plausible reading, not an isolated mechanism. Co-Tuning also moves the chosen
checkpoint later (median epoch 63 vs 34 for the two-stream member, 28 vs 22
for the TCN). Fine-tuning longer has cost ET before (`transfer_2015.md`).

**BSS as specified is inactive on the two-stream member.** Its 90 x 96
penultimate matrix has 9 dead (all-zero) ReLU columns and rank 87 < 90. The
smallest singular value is about 1e-7, so the k = 1 penalty and its gradient
are zero (median 0.0000 at fine-tune epoch 0;
`python -m experiments._bss_rank_diagnostic`). On the TCN (90 x 16, full rank) the penalty is
about 15 x 1e-3 ≈ 0.015, against a CE of about 1. It moves 2 % of
predictions. `bss` is therefore close to `ft` by construction. A larger
weight or k would be a new hyper-parameter choice, and it was not made after
seeing the results.

## Prediction vs outcome (docstring, recorded before the run)

| prediction | outcome |
|---|---|
| cotune PD-vs-ET AUC within ±0.015 of ft | **failed** (−0.026 \*) |
| cotune precET \|Δ\| < 0.03, not significant | **failed** (−0.046 \*, also at patient level) |
| cotune top-5 ET not collapsed (\|Δ\| < 0.10) | held (−0.060) |
| cotune precPD / precN within ±0.02 | held (−0.010 / −0.004) |
| bss agreement with ft ≥ 90 %, every \|Δ\| < 0.015, nothing significant | held on agreement (98 %) and size; precN −0.005 reaches patient-level significance (negligible) |
| cotune_bss ≈ cotune | held |
| nothing significant at patient level | **failed**: cotune precET and macroP, and cotune_bss macroP, are significantly *worse* |

For `reports/failed_predictions.md` (not edited here; existing files are off
limits for this agent): the Co-Tuning direction was predicted null and came out
significantly negative. This is again a mechanism-style extrapolation from
L2-SP and WiSE-FT rather than a measured quantity. The relationship matrix,
which could have been inspected before the run, already showed the PD and ET
rows nearly identical.

## Verdict

**Not adopted. Closed: Co-Tuning, BSS and their combination for the 2015
transfer model.**

* Co-Tuning is **harmful at the patient level**: precET −0.046
  [−0.091, −0.007] \*, macroP −0.020 [−0.038, −0.004] \*. It is significantly
  worse at repeat level on every ET and ranking metric, and wins 0-20 % of
  repeats. Making the PADS head's view of 2015 part of the loss brings back
  PADS's PD/ET confusion, the same failure seen with WiSE-FT, here from the
  feature side.
* BSS at 1e-3 / k = 1 is a no-op: it is zero on the rank-deficient two-stream
  features and negligible on the TCN. No interval excludes 0 in the helpful
  direction.
* `ft` stays the recipe. Together with L2-SP, LP-FT, WiSE-FT, SWA, SAM and
  label smoothing, every regulariser that ties fine-tuning to the PADS model
  (weights, head or features) is now null or harmful on 2015 ET.
