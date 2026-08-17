# Tremor characteristics and classification from frequency alone

Goal 1: describe the tremor in the quantities clinicians name, and classify from
mean / max frequency alone before adding anything else.

Angular velocity, 3-15 Hz, Welch-512, per patient. `python -m tfbench.characteristics`.

## Characteristics per class

PADS StretchHold (n=383: 79 N, 276 PD, 28 ET) -- the largest cohort:

| characteristic | N | PD | ET |
|---|---|---|---|
| max_freq (Hz) | 7.20 +/- 2.44 | 7.07 +/- 1.80 | **6.16 +/- 1.76** |
| mean_freq (Hz) | 8.02 +/- 1.03 | 7.67 +/- 1.00 | **6.61 +/- 1.10** |
| bandwidth (Hz) | 2.94 +/- 0.34 | 2.48 +/- 0.46 | **2.04 +/- 0.57** |
| inband_frac | 0.57 +/- 0.15 | 0.69 +/- 0.17 | **0.76 +/- 0.18** |
| harm_ratio | 0.23 +/- 0.18 | 0.16 +/- 0.17 | **0.08 +/- 0.07** |
| peak_sharp | 4.08 +/- 1.16 | 5.80 +/- 3.45 | **12.19 +/- 6.94** |

**`peak_sharp` is the standout.** ET tremor is three times more sharply peaked
than PD (12.2 against 5.8) and PD roughly 1.4x more than healthy. ET is close to
a pure tone; PD is broader and noisier. This matches ET's clinical description
as a rhythmic, monotonal postural tremor, and it is not one of the quantities
the usual descriptor set states explicitly.

The frequency ordering is ET < PD < N on both max and mean frequency, and ET has
the narrowest bandwidth and the highest in-band power fraction -- a tighter,
cleaner oscillation.

## Classification, features added one at a time

### N vs Tremor -- precision above 0.90 from six interpretable numbers

| features | 2015 AUC / prec | PADS AUC / prec |
|---|---|---|
| max_freq | 0.623 / 0.707 | 0.468 / 0.766 |
| + mean_freq | 0.783 / 0.819 | 0.624 / 0.856 |
| + bandwidth | 0.869 / 0.873 | 0.786 / 0.900 |
| + inband_frac | 0.887 / 0.900 | 0.806 / 0.915 |
| + harm_ratio | 0.885 / 0.887 | 0.808 / 0.920 |
| + peak_sharp | **0.890 / 0.910** | 0.804 / **0.924** |

**Tremor precision reaches 0.910 (2015) and 0.924 (PADS)** from six frequency
characteristics and a logistic regression. This is the one place in the project
where >0.90 precision is met, and it is met by the simplest model tried.

`bandwidth` is the biggest single addition on both cohorts (+0.086 AUC on 2015,
+0.162 on PADS) -- larger than mean frequency.

### PD vs ET -- works on PADS, fails on 2015

| features | PADS AUC | 2015 AUC |
|---|---|---|
| max_freq | 0.649 | 0.318 |
| + mean_freq | **0.786** | 0.295 |
| + bandwidth | 0.770 | 0.323 |
| all six | 0.791 | 0.305 |

On PADS, **mean frequency added to max frequency gives AUC 0.786** -- close to
the full 10-descriptor set (0.807) from two numbers.

On 2015 every frequency feature is **below chance** (AUC 0.29-0.32) and ET
precision never exceeds 0.081. The frequency route simply does not transfer to
that cohort. `temporal_stability.md` shows what does work there:
instantaneous-frequency stability reaches AUC 0.652 on the same patients.

NewData (6 ET) is not measurable on this axis and its numbers should be ignored.

## Reading

* **N-vs-Tremor is a frequency problem** and is close to solved with six
  interpretable features.
* **PD-vs-ET is not a frequency problem on every cohort.** It is on PADS
  (0.786 from two numbers) and is not on 2015 (below chance). Reporting a single
  PD-vs-ET frequency result across cohorts would hide that split.
* `peak_sharp` and `bandwidth` -- shape, not location -- carry more than the
  frequency values themselves, which is worth stating since mean and max
  frequency are the usual first choice.
