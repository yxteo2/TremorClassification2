# Ranking-aware fine-tuning losses for ET: AUC surrogates hurt, AP is neutral

Run: `python -m experiments.rank_loss_150 check`, `REPS=0-20 python -m
experiments.rank_loss_150`, `python -m experiments.rank_loss_150 report`
(150-patient table, 20 repeats, own `ft` asserted bit-identical to
`common.protocol.train`; agent-written, report completed by the coordinator
after the agent hit an API limit -- the run itself finished).

During fine-tuning only, an auxiliary term on the ET margin m = z_ET −
logsumexp(z_N, z_PD) is added (weight 0.5) to class-weighted CE; pretraining and
the best-validation-CE checkpoint rule are unchanged:

* `auc` -- pairwise squared-hinge AUC surrogate over all (ET, non-ET) pairs
  (Yuan et al., ICLR 2022);
* `pauc` -- the same on the hardest 20 % of non-ET (one-way partial AUC; Zhu et
  al., ICML 2022);
* `ap` -- smoothed average-precision surrogate (Qi et al., NeurIPS 2021).

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET | ET preds |
|---|---|---|---|---|---|---|---|---|
| ft | 0.732 | 0.710 | 0.329 | 0.590 | 0.581 | 0.613 | 0.420 | 14.0 |
| auc | 0.718 | 0.693 | 0.266 | 0.559 | 0.551 | 0.608 | 0.320 | 15.6 |
| pauc | 0.724 | 0.677 | 0.233 | 0.545 | 0.540 | 0.620 | 0.280 | 17.2 |
| ap | 0.729 | 0.715 | 0.325 | 0.590 | 0.581 | 0.620 | 0.500 | 15.1 |

| vs ft | repeat level | patient level |
|---|---|---|
| auc | precET −0.064 \*, macroP −0.032 \*, top-5 −0.100 \* | **precET [−0.121, −0.010] \***, macroP \*, macroF1 \* |
| pauc | precET −0.096 \*, precPD −0.033 \*, macroP −0.046 \* | **precET [−0.172, −0.034] \***, precPD \*, macroP \* |
| ap | AUC +0.008 \*, top-5 ET +0.080 \*, precision null | all n.s. (top-5 ET [−0.010, +0.120]) |

## Verdict

**Not adopted.** The pairwise AUC losses push more patients into ET (15.6 /
17.2 ET calls per repeat vs 14.0) and lose ET precision significantly at the
patient level -- extra gradient on the few training ET (~9 per fold) that sit
inside the non-ET cloud fits them rather than finding structure. The AP
surrogate is harmless and lifts the top-5 ET ranking at the repeat level
(+0.080 \*, 6 of 20 repeats up, 14 tied), but not at the patient level: a lead
for a larger cohort, not a result here.

Predictions (docstring): no arm raises precET significantly at the patient level
-- **held**; precET within ±0.03 for every arm -- **failed** (auc −0.064, pauc
−0.096); ap closest to ft -- held.
