# Tested precision–coverage results: no demonstrated three-class >0.90

Every precision figure elsewhere in this repo is at **100 % coverage** -- the
model must label all 404 patients, including genuinely ambiguous ones. Precision
is threshold-dependent, so that is the hardest operating point. This report
measures the full precision-coverage trade with an abstain option, to describe
the tested models and operating points. These historical experiments do not
establish a fundamental performance ceiling or rule out other models, features,
training data, calibration methods or thresholds.

## 3-class N/PD/ET, best model (two-stream, tuned priors)

Pooled out-of-fold predictions for 358 of 404 patients, margin abstention:

| coverage | n | precN | precPD | precET | macroP | n per class |
|---|---|---|---|---|---|---|
| 1.00 | 358 | 0.629 | 0.640 | 0.630 | 0.633 | 150 / 164 / 44 |
| 0.90 | 322 | 0.658 | 0.655 | 0.591 | 0.635 | 139 / 146 / 37 |
| 0.80 | 286 | 0.681 | 0.680 | 0.588 | 0.650 | 127 / 129 / 30 |
| 0.70 | 251 | 0.715 | 0.696 | 0.562 | 0.658 | 114 / 113 / 24 |
| 0.50 | 179 | 0.736 | 0.744 | 0.500 | 0.660 | 83 / 83 / 13 |
| 0.30 | 107 | 0.794 | 0.737 | 0.667 | **0.732** | 58 / 39 / 10 |

**The displayed three-class macro precision values do not reach 0.90.**
Abstaining on 70 % buys 0.732, and the curve is flattening far below target.

The `max_prob` rule behaves similarly (best 0.705 at 70 % coverage).

## ET precision gets WORSE under abstention

0.630 at full coverage, 0.562 at 70 %, 0.462 at 60 %. Confidence is not tracking
correctness for the minority class: the model is confidently wrong on some ET
patients and uncertain on some it gets right. **The tested abstention rules did not reliably improve ET precision.**
This warrants class-specific calibration analysis; it does not prove that every
threshold fails or isolate calibration as the cause.

## A high binary value at low coverage needs independent validation

N-vs-Tremor reports macro precision 0.971 at 25 % coverage. The class counts
there are 13 N against 94 tremor -- only 13 true N patients in the retained subset. Precision denominators
are predicted-class counts and must be reported separately. At 30 % coverage
it is 0.858. The 0.971 is exploratory: low counts and threshold exploration
make it unsuitable as a validated headline, but do not prove it is an artifact.

## Binary axes at full coverage

| axis | class precisions | macroP |
|---|---|---|
| N vs Tremor | N 0.704, Tremor 0.734 | 0.719 |
| PD vs ET (tremor patients only) | PD 0.852, ET 0.654 | 0.753 |

PD-vs-ET at 0.753 macro precision, with PD precision 0.852, is the most
defensible headline available from this data.

## Directions to evaluate; none guarantees >0.90

1. **More ET patients.** 49 across three cohorts limits minority-class evaluation;
   it does not prove sample size is the sole cause of observed errors.
2. **The questionnaire.** The published PADS baseline reaches 91 % on PD-vs-HC
   using smartwatch IMU *plus* a 30-item non-motor symptom instrument
   (`pads_literature.md`). It is the one lever the state of the art uses that
   this pipeline does not.
3. **A narrower question.** Binary PD-vs-ET, reported honestly at ~0.75 macro
   precision, is defensible; a 3-class >0.90 claim is not supportable.

The tested architecture changes had limited gains. Five rounds moved macro
precision from 0.583 to 0.675; the remaining gap to 0.90 is larger than
everything that work bought combined.

Reproduce: `python -m metrics.selective` helpers, `scratch/selective_run.py`.

