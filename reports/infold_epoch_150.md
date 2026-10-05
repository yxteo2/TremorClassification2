# Choosing the fine-tuning epoch inside the fold: the "TCN epoch 20" finding is not supported

Run: `python -m experiments.infold_epoch_150 check`, then
`REPS=0-20 OUT_DIR=<dir> python -m experiments.infold_epoch_150`, then
`... report` (150-patient table, 20 repeats; `ft` arm asserted bit-identical to
`common.protocol.train` on both members; agent-written, report completed by the
coordinator).

`checkpoint_rule_2015` found "TCN at fine-tuning epoch 20" worth PD-vs-ET AUC
+0.012 / +0.017 \*, but epoch 20 was read off test curves that, through CV,
include every patient. Here the epoch is chosen from the inner validation split
only. All arms come from the same trajectories (3 seeds per member):

* `pooled_loss` -- per member, the epoch minimising the weighted CE of the
  seed-averaged validation prediction;
* `pooled_f1` -- the epoch maximising seed-averaged validation macro-F1;
* `tcn_pooled` -- two-stream as `ft`, TCN as `pooled_loss` (in-fold analogue of
  the biased rule);
* `tcn_e20` -- the biased rule itself, reference only.

| arm | precN | precPD | precET | macroP | macroF1 | AUC | top-5 ET |
|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 |
| pooled_loss | 0.726 | 0.711 | 0.285 | 0.574 | 0.566 | 0.618 | 0.390 |
| pooled_f1 | 0.720 | 0.697 | 0.288 | 0.568 | 0.561 | 0.625 | 0.290 |
| tcn_pooled | 0.728 | 0.714 | 0.314 | 0.585 | 0.576 | 0.614 | 0.430 |
| tcn_e20 (biased) | 0.719 | 0.733 | 0.346 | 0.599 | 0.586 | 0.624 | 0.420 |

| vs ft | repeat level | patient level |
|---|---|---|
| pooled_loss | precET −0.045 \*, macroP −0.016 \*, AUC +0.005 \* | **precET −0.045 [−0.092, −0.000] \***, macroP \* |
| pooled_f1 | precET −0.042 \*, macroP −0.022 \*, top-5 −0.130 \* | macroP −0.022 \* |
| tcn_pooled | all n.s. (AUC +0.002) | all n.s. |
| tcn_e20 | precPD +0.024 \*, AUC +0.011 \*, precN −0.013 \* | precPD +0.024 [+0.009, +0.040] \*, precN −0.013 \*; AUC n.s. |

Chosen epochs: `pooled_loss` TCN median 21 (IQR 4-60), two-stream 24;
`pooled_f1` picks very early (medians 9-10). Validation macro-F1 on ~30
patients peaks early by chance.

## Verdict

* **The epoch-20 AUC gain is not supported by in-fold selection**: the in-fold
  analogue (`tcn_pooled`) is null on every metric. The biased rule's own AUC gain
  (+0.011 repeat level) is not significant at the patient level.
* Averaging seed curves or selecting on validation macro-F1 makes ET precision
  worse. **The per-seed best-validation-loss checkpoint stays.**
* Predictions (docstring): `tcn_pooled` null and not reproducing the biased gain
  -- held; `pooled_f1` picks earlier epochs -- held; `tcn_e20` AUC +0.005-0.015
  -- held (+0.011); `pooled_loss` precET within ±0.03 -- **failed** (−0.045).
