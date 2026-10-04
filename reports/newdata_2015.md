# Integrating NewData into the 2015 model: PD-vs-ET ranking up, precision flat

Run: `REPS=a-b python -m experiments.newdata_2015` in parts, then
`python -m experiments.newdata_2015 report` (40 repeats); stacking:
`python -m experiments.pdet_stack_2015`.

PADS is already in the model (`ft` pretrains on PADS StretchHold). NewData (56
patients: 27 HC / 23 PD / 6 ET, same OUT task, three arm sensors) had been tried
only in pretraining (`ft_pn`, null). Here: pooled into fine-tuning (2015 is the
only test set; NewData z-scored on its own statistics, asymmetry columns and
have-flag zeroed to match one-armed 2015), a label-shuffled control, and a
sequential PADS -> NewData -> 2015 stage. All arms branch from the same
PADS-pretrained weights per seed; **the `ft` arm reproduces the saved `ft`
predictions exactly on every repeat.**

## Result (40 repeats, selection partitions)

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC | top-5 ET |
|---|---|---|---|---|---|---|---|
| ft | 0.723 | 0.737 | 0.350 | 0.603 | 0.590 | 0.625 | 0.495 |
| **nd_pool** | 0.716 | 0.727 | 0.333 | 0.592 | 0.576 | **0.669** | 0.500 |
| nd_pool_shuf | 0.720 | 0.704 | 0.297 | 0.574 | 0.564 | 0.632 | 0.495 |
| nd_seq | 0.705 | 0.740 | 0.336 | 0.594 | 0.579 | 0.640 | 0.530 |

| contrast | precPD | precET | macroP | macroF1 | AUC |
|---|---|---|---|---|---|
| nd_pool − ft | −0.010 | −0.017 | −0.011 | **−0.015 \*** | **+0.044 \*** (win 0.95) |
| nd_pool − shuffled | **+0.023 \*** | +0.036 | **+0.018 \*** | +0.012 | **+0.037 \*** |
| shuffled − ft | **−0.033 \*** | **−0.053 \*** | **−0.029 \*** | **−0.027 \*** | +0.007 |
| nd_seq − ft | +0.003 | −0.014 | −0.010 | −0.012 | **+0.015 \*** |

* **NewData's labels carry PD-vs-ET information**: pooled with real labels the
  ranking improves +0.044 \* over `ft` and +0.037 \* over shuffled labels, which
  themselves hurt every precision.
* **It does not raise precision at the operating point** (macroF1 −0.015 \*,
  precET −0.017 n.s.) -- the same ranking-not-decision pattern as REST fusion.
* Sequential transfer gains less (+0.015 \* AUC) and costs control precision.

## The ranking gains stack (selection partitions, within-fold AUC)

| model | PD-vs-ET AUC |
|---|---|
| ft | 0.625 |
| nd_pool | 0.669 |
| fuse (ft + REST) | 0.660 |
| **fuse_nd (nd_pool + REST)** | **0.694** (+0.033 \* over fuse) |
| fuse + coupling (`pdet_2015`, 0.710 on fresh partitions) | 0.712 |
| **fuse_nd + coupling** | **0.722** (+0.011 \* over fuse + coupling; +0.097 \* over ft) |

Assembled after seeing the parts -- a candidate. Fresh-partition confirmation
(seeds 100-139) is running.

## Prediction (in `newdata_2015.py`'s docstring)

NewData adds nothing significant to ET precision; nd_pool vs shuffled |precET| <
0.04 n.s.; nd_seq null -- **held** on precision (precET −0.017, +0.036, −0.014,
all n.s.). The AUC gain (+0.044 \*) was not predicted. (AH)
