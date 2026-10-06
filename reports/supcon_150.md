# Supervised contrastive PADS pretraining: null

Run: `python -m experiments.supcon_150 check`, `REPS=0-20 python -m
experiments.supcon_150`, `python -m experiments.supcon_150 report`
(150-patient table, 20 repeats; own `ft` asserted bit-identical to
`common.protocol.train`; agent-written, run resumed and report completed by the
coordinator after the agent hit an API limit).

Only PADS pretraining differs (200 epochs, same optimiser and schedule); the
2015 fine-tune is identical:

* `supcon` -- CE + 0.5 x supervised contrastive loss (Khosla et al., NeurIPS
  2020; temperature 0.1) on L2-normalised penultimate features, full batch;
* `supcon_aug` -- the same with a Gaussian-noise copy of each PADS row
  (sd 0.05, z-scored units) as an extra positive view.

Unlike the closed self-supervised pretraining (no labels, own-data corpus), this
uses PADS's labels to cluster classes in feature space.

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET |
|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 |
| supcon | 0.734 | 0.711 | 0.346 | 0.597 | 0.586 | 0.598 | 0.370 |
| supcon_aug | 0.721 | 0.721 | 0.333 | 0.591 | 0.584 | 0.602 | 0.380 |

Every contrast with `ft` is n.s. at the repeat and the patient level (supcon
precET +0.017 [−0.044, +0.068], AUC −0.014 [−0.049, +0.016]). supcon_aug vs
supcon: precN −0.014, significant at patient level, otherwise null.

**Verdict: not adopted.** CE pretraining already separates the PADS classes at
the head; a contrastive term tightens the same clusters one layer earlier and
adds no 2015 information, and the full fine-tune re-shapes the features anyway.

Prediction (docstring): null on every metric, supcon_aug ~= supcon -- **held**.
