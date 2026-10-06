# Newer fine-tuning methods on the 2015 transfer model: none beats plain `ft`

Run: `REPS=a-b python -m experiments.training_methods_2015` in parts, then
`python -m experiments.training_methods_2015 report` (40 repeats, CPU ~10
min/repeat/process).

## Why these five

`transfer_2015.md` found fine-tuning longer loses ET (−0.021 \*) -- drift from
the PADS-pretrained weights costs what PADS taught -- while L2-SP raised ranking
only and head-only fine-tuning was too weak. Each arm changes one thing about
how `ft` is fine-tuned, branching from the **same** PADS-pretrained weights per
seed:

* **LP-FT** (Kumar et al., ICLR 2022) -- 40 epochs head-only (BN frozen), then
  the usual full fine-tune;
* **WiSE-FT** (Wortsman et al., CVPR 2022) -- 0.5 x pretrained + 0.5 x
  fine-tuned weights;
* **SWA** -- uniform weight average over fine-tune epochs 41-80, BN statistics
  recomputed, replacing the best-validation-epoch checkpoint (~30 validation
  patients);
* **SAM** (Foret et al., ICLR 2021), rho 0.05;
* **label smoothing** 0.1.

**Assert first:** the `ft` arm (a copy of `common.protocol.train` with hooks)
reproduces `segments_2015_runs`' saved `ft` predictions and probabilities
exactly on every repeat.

## Result (40 repeats, paired vs `ft`)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET | sd(precET) |
|---|---|---|---|---|---|---|---|---|
| ft | 0.723 | 0.737 | 0.350 | 0.603 | 0.590 | 0.625 | 0.495 | 0.129 |
| LP-FT | +0.002 | −0.002 | +0.008 | +0.003 | +0.000 | **+0.005 \*** | +0.015 | 0.126 |
| WiSE-FT | **+0.014 \*** | **−0.046 \*** | **−0.049 \*** | **−0.027 \*** | **−0.020 \*** | −0.006 | **−0.270 \*** | 0.090 |
| SWA | −0.001 | +0.003 | **−0.049 \*** | **−0.016 \*** | **−0.012 \*** | **−0.019 \*** | −0.005 | 0.107 |
| SAM | −0.001 | −0.008 | −0.012 | −0.007 | −0.006 | +0.001 | +0.000 | 0.108 |
| label smoothing | +0.007 | **−0.012 \*** | −0.025 | −0.010 | −0.008 | +0.001 | −0.020 | 0.111 |

## Reading

* **No method raises ET precision.** LP-FT is the only arm with no significant
  loss, and its only gain is +0.005 AUC.
* **WiSE-FT is harmful, and informatively so**: pulling the weights halfway back
  to the PADS-pretrained model cuts PD precision (−0.046 \*) and collapses the top
  of the ET ranking (top-5 0.495 -> 0.225 \*). The PADS-trained decision rule is
  wrong for 2015 patients -- consistent with the reversed PD/ET frequency
  relation between PADS and the in-house cohorts (`rest_replication.md`).
  Fine-tuning must move away from PADS's boundary while keeping its features;
  interpolation undoes the first.
* **SWA is worse than the best-validation checkpoint** (precET −0.049 \*, AUC
  −0.019 \*): averaging the late epochs averages in the drift that
  `transfer_2015.md` found costly. The noisy checkpoint is still the better
  choice.
* SAM and label smoothing are null or slightly negative.

## Verdict

**Closed for the 2015 transfer model: LP-FT, WiSE-FT, SWA, SAM, label
smoothing**, alongside the earlier fine-tune length, head-only, L2-SP, logit
adjustment, mixup/augmentation. The `ft` recipe sits at an optimum of the
training procedure; remaining gains are in inputs (REST / WING / coupling) and
data.

## Prediction (in the docstring)

* No method raises ET precision significantly -- **held** (AG); SAM null, label
  smoothing null-to-negative -- held.
* WiSE-FT and LP-FT raise PD-vs-ET AUC +0.01-0.02 -- **failed** (LP-FT +0.005 /
  WiSE-FT −0.006); SWA reduces spread but not the mean -- **failed** (mean −0.049 \*).
  #43.
