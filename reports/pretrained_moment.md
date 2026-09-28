# Transfer from an online pretrained model: MOMENT-1-small helps the 2015 OUT model

**Question.** Does a publicly available pretrained model transfer to 2015 OUT
tremor classification? Earlier, an ImageNet ViT did not (macroP 0.501 vs 0.652,
`frozen_vit.md` in git history).

**Answer.** Yes, for a *time-series* foundation model. Frozen MOMENT-1-small
embeddings (AutonLab, MIT licence) beat the pipeline's own spectrum +
descriptor features under the same linear classifier, the gain is carried by the
pretrained weights (weight-permuted copies of the same network are far worse),
and soft-voting MOMENT with the PADS-transfer model **robustly improves
PD-vs-ET ranking (AUC +0.063 \*, win 0.93) and N / PD precision** on 40 fresh
partitions. **It does not improve ET precision** -- at the tuned offsets it
lowers it on fresh partitions (−0.058 \*), and it is no better among the k most
ET-like patients -- and its macroP gain (+0.019 \* on the first partition set)
did not replicate (−0.001). For the ET-precision target `ft` stays adopted; the
ensemble is the better model when balanced N / PD / ET performance matters.

Run (`experiments/pretrained_moment.py`, `experiments/moment_ensemble.py`):

```
python -m experiments.pretrained_moment export                     # project env
<moment env> -m experiments.pretrained_moment embed                # separate env
WEIGHTS=perm0 <moment env> -m experiments.pretrained_moment embed  # controls
python -m experiments.pretrained_moment eval
TREMOR_DEVICE=cuda REPS=0-8 python -m experiments.moment_ensemble  # parts; then `report`
```

## Setup

* **Model:** `AutonLab/MOMENT-1-small` from Hugging Face -- `config.json` and
  `model.safetensors` (151,615,328 bytes, size verified) only, loaded offline
  (`HF_HUB_OFFLINE=1`); all 78 encoder tensors verified identical to the file.
  The T5 backbone is built from the file's own `t5_config`; nothing else is
  downloaded.
* **Environment:** `momentfm` pins numpy 1.25 / transformers 4.33, so it runs in a
  separate conda env `moment` (CPU PyTorch, ~1 GB); the project env is untouched.
  Embedding 1,269 windows takes ~1 min on CPU. The embeddings are the only thing
  passed back.
* **Input:** 2015 OUT, lower-arm angular velocity (3 axes), 512-sample windows
  (5.12 s, hop 256), band-passed 3-15 Hz; 274 recordings -> 1,269 windows. MOMENT
  normalises each channel (RevIN) and averages its encoder output over patches
  and axes -> 512-d per window; mean over windows, then recordings, per patient.
  **It sees waveform shape, not amplitude.**

## 1. Frozen embeddings vs the pipeline's features (linear, argmax, 40 partitions)

| arm | precN | precPD | precET | macroP | PD-vs-ET AUC | N-vs-tremor AUC |
|---|---|---|---|---|---|---|
| spectrum + descriptors (26-d) | 0.717 | 0.796 | 0.228 | 0.580 | 0.621 | 0.885 |
| **MOMENT, band-passed** | 0.760 | 0.796 | 0.287 | 0.614 | **0.701** | 0.905 |
| MOMENT, raw | 0.790 | 0.791 | 0.289 | 0.623 | 0.672 | 0.915 |
| MOMENT, weights permuted (3 seeds) | 0.609-0.644 | 0.631-0.636 | 0.147-0.245 | 0.474-0.504 | 0.527-0.619 | 0.769-0.795 |

* MOMENT − spectrum: PD-vs-ET AUC **+0.080 \*** (win 1.00), precET +0.059 \*,
  macroP +0.034 \*.
* MOMENT − weight-permuted MOMENT: PD-vs-ET AUC +0.082 to +0.173 \*, macroP
  +0.111 to +0.141 \* (win ~1.00). Same architecture, same weight values, same
  scale -- **the gain is the pretrained structure**, i.e. genuine transfer.
* Label-permutation null (binary PD vs ET, same pipeline, 200 permutations):
  AUC **0.710**, null 95 % [0.300, 0.686], **p = 0.020**.
* Union (spectrum + MOMENT) vs the same union with MOMENT rows permuted across
  patients: every column \*, so the MOMENT columns are informative, not just extra
  dimensions (the permuted union is *worse* than spectrum alone, macroP 0.519).

## 2. Head-to-head with the adopted model, same decision rule (40 partitions)

Same partitions, folds, validation splits, PADS draws and validation-tuned logit
offsets as `transfer_2015`.

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| ft (PADS pretrain -> fine-tune) | 0.720 | 0.728 | **0.334** | 0.594 | 0.580 | 0.627 |
| mom_lr (MOMENT + logistic) | 0.744 | **0.775** | 0.299 | 0.606 | **0.605** | **0.693** |
| **ens** (ft + mom_lr, soft vote) | **0.755** | 0.760 | 0.324 | **0.613** | 0.603 | 0.691 |
| ens_ctrl (ft + weight-permuted MOMENT) | 0.665 | 0.724 | 0.209 | 0.533 | 0.528 | 0.591 |

Paired:

| contrast | precN | precPD | precET | macroP | PD-vs-ET AUC |
|---|---|---|---|---|---|
| mom_lr − ft | +0.024 \* | +0.048 \* | −0.035 | +0.012 | **+0.066 \*** (win 0.85) |
| **ens − ft** | +0.034 \* | +0.032 \* | −0.010 | **+0.019 \*** | **+0.064 \*** (win 0.93) |
| ens − ens_ctrl | +0.089 \* | +0.036 \* | +0.115 \* | +0.080 \* | +0.100 \* |

## Reading it

* **Pretraining on generic time series transfers; pretraining on images did not.**
  The frozen ViT saw a spectrogram as a picture; MOMENT sees the waveform as a
  time series, which is what it was trained on.
* **What MOMENT adds is ranking and PD/N precision, not ET precision.** ens keeps
  ft's ET precision (−0.010, n.s.) while recovering most of the PD precision the
  PADS transfer cost, and ranks ET against PD much better. Its gain over its
  control shows the second member must carry real information: a weight-permuted
  MOMENT member makes ft *worse*.
* MOMENT is amplitude-blind (RevIN), so it complements the descriptor stream,
  which carries amplitude -- plausibly why the ensemble works.

## Confirmation on 40 fresh partitions (seeds 100-139)

Section 2's partitions had been used by every earlier 2015 transfer run, so the
same comparison was repeated on partitions never used before. (No numeric
prediction was written for this stage; the expectation was replication.)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| ft | 0.720 | 0.737 | **0.377** | 0.611 | 0.600 | 0.628 |
| mom_lr | 0.743 | **0.762** | 0.310 | 0.605 | 0.601 | 0.686 |
| ens | **0.760** | 0.753 | 0.319 | 0.611 | **0.603** | **0.691** |
| ens_ctrl | 0.660 | 0.737 | 0.236 | 0.545 | 0.539 | 0.600 |

| contrast | precN | precPD | precET | macroP | PD-vs-ET AUC |
|---|---|---|---|---|---|
| ens − ft | +0.040 \* | +0.016 \* | **−0.058 \*** (win 0.28) | −0.001 | **+0.063 \*** (win 0.93) |
| mom_lr − ft | +0.023 \* | +0.025 \* | −0.066 \* | −0.006 | +0.058 \* |
| ens − ens_ctrl | +0.100 \* | +0.016 | +0.083 \* | +0.066 \* | +0.091 \* |

**Replicates:** the ranking gain (+0.063 vs +0.064), the precN / precPD gains, the
margin over the permuted-MOMENT control. **Does not replicate:** macroP
(+0.019 \* -> −0.001); precET goes from −0.010 (n.s.) to −0.058 \*.

### Stricter ET thresholds (post hoc, saved probabilities)

ET precision among the k most ET-like of 151 patients (15 ET):

| partitions | arm | k = 3 | k = 5 | k = 8 | k = 10 | k = 15 |
|---|---|---|---|---|---|---|
| 0-39 | ft | 0.517 | 0.490 | 0.450 | 0.425 | 0.352 |
| 0-39 | ens | 0.592 | 0.545 | 0.463 | 0.428 | 0.353 |
| 100-139 | ft | 0.433 | 0.495 | 0.463 | 0.430 | 0.362 |
| 100-139 | ens | 0.633 | 0.525 | 0.453 | 0.403 | 0.353 |

ens − ft at k = 5: +0.055 (win 0.40) and +0.030 (win 0.30), both n.s.; at
k = 10: +0.003 and −0.027. A positive mean with a sub-0.5 win rate is a few
favourable splits, not a gain (invariant 5). The PD-vs-ET ranking gain is spread
through the ranking rather than concentrated among the top ET candidates, and
the ET-vs-rest ranking also has to beat N patients.

## Fine-tuning MOMENT (LP-FT, 20 fresh partitions, seeds 200-219)

`experiments/moment_finetune.py`. Linear probe on frozen embeddings, then all
weights at lr 1e-5 (head 1e-4), up to 15 epochs, early stopping on validation
patients with the probe as epoch 0; hyperparameters fixed in advance. Control:
identical LP-FT from weight-permuted MOMENT. `ft` and frozen `mom_lr` re-run on
the same partitions. Trained in the `moment` env with CUDA PyTorch
(2.14.0+cu130) on the RTX 5070 Ti.

**A trap, caught before the run:** `momentfm` freezes the encoder and patch
embedder by default (`freeze_encoder` / `freeze_embedder` = True). The first
smoke run therefore "fine-tuned" only the head and produced plausible numbers;
a `check_backward_validity` warning and a direct gradient test exposed it. The
experiment now passes `freeze_encoder=False, freeze_embedder=False` and asserts
the encoder receives gradient on the first step. That smoke result was
discarded.

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| ft | 0.714 | 0.740 | 0.343 | 0.599 | 0.588 | 0.644 |
| mom_lr (frozen) | 0.744 | 0.758 | 0.324 | 0.609 | 0.608 | 0.703 |
| mom_ft (fine-tuned) | 0.703 | 0.768 | 0.306 | 0.592 | 0.586 | 0.693 |
| mom_ft_perm (control) | 0.570 | 0.747 | 0.249 | 0.522 | 0.483 | 0.577 |
| ens_lr (ft + frozen) | 0.746 | 0.756 | 0.363 | 0.622 | 0.613 | 0.708 |
| ens_ft (ft + fine-tuned) | 0.723 | 0.770 | 0.372 | 0.621 | 0.610 | 0.701 |

Early stopping kept the probe in 15 % of folds (median epoch 8), so fine-tuning
did move the encoder; for the permuted control it kept the probe in 51 %.

| contrast | precN | precPD | precET | macroP | PD-vs-ET AUC |
|---|---|---|---|---|---|
| mom_ft − mom_lr | −0.042 \* | +0.011 | −0.017 | −0.016 | −0.010 |
| mom_ft − mom_ft_perm | +0.132 \* | +0.021 | +0.057 \* | +0.070 \* | **+0.117 \*** |
| ens_ft − ens_lr | −0.023 \* | +0.014 | +0.009 | −0.000 | −0.007 |

**Fine-tuning adds nothing over the frozen encoder** (all three predictions
held): ~90 training patients per fold cannot improve a ~35 M-parameter encoder's
features, and it costs a little N precision. The pretrained start still matters
enormously when every weight trains (+0.117 AUC over the permuted start).
**Keep MOMENT frozen.**

## The ensemble across all 100 fresh-or-reused partitions

ft + frozen MOMENT minus ft, three independent partition sets:

| partitions | precET | macroP | PD-vs-ET AUC |
|---|---|---|---|
| 0-39 | −0.010 | +0.019 \* | +0.064 \* |
| 100-139 | −0.058 \* | −0.001 | +0.063 \* |
| 200-219 | +0.020 | +0.022 | +0.064 \* |
| **pooled (100)** | **−0.023** [−0.049, +0.002] | **+0.012 \*** [+0.002, +0.022] | **+0.064 \*** [+0.055, +0.072] |

The ranking gain is the most reproducible effect in this study (+0.063 to +0.064
on every set). The ET-precision effect swings from −0.058 to +0.020 with the
partitions -- the instability invariant 4 warns about -- and pools to a
non-significant −0.023.

## Standing

* **MOMENT transfer is real** -- above the spectrum features, far above
  weight-permuted MOMENT, above its label null -- and it is the first external
  pretrained model to help here.
* **For the ET-precision target, keep `ft`.** Over 100 partitions the ft + MOMENT
  ensemble raises PD-vs-ET ranking (+0.064 \*) and macroP (+0.012 \*) but not ET
  precision (−0.023, n.s.), at the tuned offsets or among the top-k ET candidates.
* **Keep MOMENT frozen**: LP-FT fine-tuning is null against the frozen encoder.
* **Use the ensemble** when the goal is balanced three-class performance or
  PD-vs-ET ranking.
* Not tried: MOMENT embeddings of PADS as an extra pretraining signal,
  MOMENT-base / -large (larger downloads), parameter-efficient fine-tuning (LoRA).

## Reproducing the weights

The weights are not in the repo. Fetch the two files (MIT licence) into a folder
and point `MOMENT_DIR` at it:

```
curl -L -o config.json       https://huggingface.co/AutonLab/MOMENT-1-small/resolve/main/config.json
curl -L -o model.safetensors https://huggingface.co/AutonLab/MOMENT-1-small/resolve/main/model.safetensors
```
