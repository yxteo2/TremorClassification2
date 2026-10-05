# Stopping PADS pretraining where it transfers best: worse than the full 200 epochs

Run: `REPS=0-20 python -m experiments.pretrain_stop_150` then `... report`
(150-patient table, 20 repeats, own `ft` baseline asserted bit-identical to
`common.protocol.train`; agent-written, report completed by the coordinator).

`loss_curves_2015.md`: the two-stream member's 2015 loss is lowest at
pretraining epoch ~18 and rises after; the TCN's ~76. All arms share one
200-epoch cosine pretraining trajectory (snapshots every 5 epochs) and then the
standard fine-tune. Stop epochs chosen inside the training fold:

* `stop_val` -- lowest weighted CE of the pretrained head on the fold's 2015
  validation patients, per member and seed;
* `stop_logme` -- highest LogME (You et al., ICML 2021) of penultimate features
  of the fold's 2015 training patients;
* `stop_fixed` -- 20 (two-stream) / 75 (TCN), read off test-including curves:
  biased reference only.

| arm | precN | precPD | precET | macroP | AUC | top-5 ET |
|---|---|---|---|---|---|---|
| ft (epoch 200) | 0.732 | 0.710 | 0.329 | 0.590 | 0.613 | 0.420 |
| stop_val | 0.708 | 0.727 | 0.299 | 0.578 | 0.591 | 0.330 |
| stop_logme | 0.710 | 0.729 | 0.268 | 0.569 | 0.595 | 0.380 |
| stop_fixed | 0.706 | 0.729 | 0.336 | 0.590 | 0.604 | 0.390 |

| vs ft | repeat level | patient level |
|---|---|---|
| stop_val | precN −0.024 \*, precPD +0.017 \*, AUC −0.022 \*, top-5 −0.090 \* | precN −0.024 [−0.043, −0.006] \* |
| stop_logme | precN −0.022 \*, precET −0.061 \*, macroP −0.021 \*, AUC −0.018 \* | precN \*, **precET −0.061 [−0.132, −0.005] \*** |
| stop_fixed | precN −0.026 \*, precPD +0.020 \* | precN −0.026 [−0.047, −0.007] \* |

Chosen epochs: two-stream `stop_val` median 40 (61 % <= 50), `stop_logme` 52;
TCN `stop_val` bimodal (median 90, 31 % in epochs 1-25), `stop_logme` 60. The
mean 2015 validation CE of the two-stream bottoms at epoch 40 (1.006) and rises
to 1.176 at 200, yet stopping there costs N precision and ranking.

## Verdict

**Not adopted; full 200-epoch pretraining stays.** Every early stop trades N
precision (significant at the patient level) for PD precision, and the
LogME-chosen stop also loses ET precision at the patient level. The rising 2015
loss reflects an over-confident PADS head, not worse features for fine-tuning.

Predictions (docstring): `stop_val` precPD +0.01-0.03, AUC −0.01 to −0.03,
precET within ±0.03 -- **held** (+0.017 / −0.022 / −0.030). `stop_logme` late
epochs (median >= 100) and null -- **failed** (median 52 / 60; precET −0.061 \*).
