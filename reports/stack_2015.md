# What limits the 2015 OUT model, and a second-stage test of the fixable part

Run: `python -m experiments.stack_2015` (minutes; needs `segments_2015_runs/`
from `experiments.segments_2015`). `RULE=balanced` reproduces the first run.

## 1. Where `ft` errs (40 saved repeats, model-free follow-up)

27 of 151 patients are wrong in >= 75 % of repeats:

| error | n | log10 tremor RMS (lower arm, 3-15 Hz) | reading |
|---|---|---|---|
| PD -> N | **11** | OUT −1.96, REST −2.11, WING −2.03 (correct controls −2.04 / −2.26 / −2.08) | **no tremor in any 2015 task** |
| ET -> PD | 6 | OUT −1.57 (correct PD −1.53) | PD-like tremor; hand-forearm coherence 0.650 vs PD 0.911 |
| N -> PD | 5 | OUT −1.81, REST −2.14 (PD −1.85) | tremor at OUT only |
| ET -> N | 3 | OUT −2.03 | no tremor |
| PD -> ET | 2 | -- | -- |

* **14 of 27 consistent errors are patients with no measurable tremor** in OUT,
  REST or WING (3 of 10 PD->N exceed the control 90th percentile in either other
  task). No tremor feature or model on 2015 recordings can recover them -- the
  ceiling, and the same phenotype finding as `readjudication_list.md` on the
  merged cohort.
* **ET is recognised when it has the classic ET tremor**: correctly classified
  ET have peak sharpness 12.1, missed ET 4.0. 7 of 15 ET rank in the top 13 of
  90 tremor patients by P(ET); the other 8 sit at ranks 47-83, among PD.

## 2. Second stage on `ft`'s out-of-fold probabilities

Multinomial LR on `ft`'s 3 out-of-fold log-probabilities plus an extra block,
evaluated by an independent 5-fold CV per repeat. Decision rule = `ft`'s own
(class-weighted model, logit offsets tuned for macro-F1 on a 25 % inner
validation split).

| arm | precN | precPD | precET | macroP | macroF1 | recET |
|---|---|---|---|---|---|---|
| ft (reference) | 0.723 | 0.737 | **0.350** | **0.603** | 0.590 | 0.300 |
| stack (log-probs only) | 0.717 | 0.720 | 0.268 | 0.568 | 0.559 | 0.242 |
| + coh_hl | 0.721 | 0.740 | 0.265 | 0.575 | 0.575 | 0.343 |
| + coh_hl shuffled | 0.719 | 0.703 | 0.216 | 0.546 | 0.542 | 0.210 |
| + coupling8 | **0.756** | **0.757** | 0.274 | 0.595 | **0.595** | 0.360 |
| + REST rms (fusion) | 0.737 | 0.740 | 0.277 | 0.585 | 0.578 | 0.262 |

Paired (40 repeats):

| contrast | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| + coh_hl − shuffled | +0.001 | **+0.038 \*** | **+0.049 \*** | **+0.029 \*** | **+0.033 \*** |
| + coh_hl − stack | +0.004 | **+0.020 \*** | −0.003 | +0.007 | **+0.016 \*** |
| + coupling8 − + coh_hl | **+0.035 \*** | +0.016 | +0.009 | **+0.020 \*** | **+0.020 \*** |
| + REST rms − stack | **+0.021 \*** | **+0.019 \*** | +0.009 | **+0.016 \*** | **+0.018 \*** |
| stack − ft | −0.006 | **−0.016 \*** | **−0.082 \*** | **−0.035 \*** | **−0.031 \*** |
| + coupling8 − ft | **+0.033 \*** | **+0.020 \*** | **−0.076 \*** | −0.008 | +0.005 |

## Reading

* **The coupling information is real**: against a shuffled copy of itself it
  is significant on every precision but N's, and all 8 features beat the one
  chosen from the diagnostic.
* **Re-deciding on top of `ft` costs as much as coupling gives.** Refitting the
  decision on 151 patients (15 ET) alone loses macroP −0.035 \* and precET
  −0.082 \*; coupling recovers the macro scores (net vs `ft`: macroP −0.008,
  macroF1 +0.005, n.s.) and raises precN / precPD (+0.033 \* / +0.020 \*), but
  ET precision stays −0.076 \* below `ft`. **Not adopted for ET.**
* With a class-balanced argmax (`RULE=balanced`) every stacked arm moves toward
  ET recall (recET 0.38-0.53 vs `ft` 0.30); within-stack contrasts were larger
  there (coh_hl vs shuffled precET +0.071 \*), but no arm is comparable with `ft`.
* **REST amplitude** fixes part of the N->PD block, as the diagnostic predicted:
  precN / precPD +0.02 \*, ET flat. It is a second task -- outside the
  "one action per model" scope unless that is relaxed to late fusion.

## Predictions (in the docstring)

* + coh_hl beats stack and shuffled on precET by +0.03-0.06: **half** -- vs
  shuffled +0.049 \* (held), vs stack −0.003 (failed). #39.
* + coupling8 no better than + coh_hl: **failed** -- better on precN, macroP,
  macroF1. #39.
* + REST rms raises precPD and precN, precET flat: **held** (AB).
