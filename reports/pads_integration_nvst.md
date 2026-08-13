# Integrating PADS — it helps N-vs-Tremor

Three-cohort merging was previously tested only on **PD-vs-ET**, where it hurt
(precision 0.393 → 0.163). It was never tested on **N-vs-Tremor**, which is the
axis where merging works. That gap is now closed.

OUT condition (PADS `StretchHold` is the equivalent), lower_arm / wrist,
stft512 descriptors, patient-level LOSO. **All rows scored on the same 151 2015
patients**, so they are directly comparable.

| training cohorts | bal-acc | AUC | precision | recall | F1 |
|---|---|---|---|---|---|
| 2015 only | 0.812 | **0.902** | 0.877 | 0.789 | 0.830 |
| 2015 + NewData | 0.821 | **0.905** | 0.887 | 0.789 | 0.835 |
| **2015 + PADS** | **0.826** | 0.894 | **0.889** | **0.800** | **0.842** |
| ALL THREE | 0.804 | 0.896 | 0.866 | 0.789 | 0.826 |

Cohort sizes: 2015 61 N / 75 PD / 15 ET; NewData 27 / 23 / 6;
PADS 79 / 276 / 28.

**2015 + PADS is best on balanced accuracy, precision, recall and F1.** Adding
383 PADS patients improves classification of the 2015 patients. This is the
first time PADS has improved anything in this project.

## Two findings inside that table

**More data is not monotonically better.** All three cohorts together (0.804) is
worse than either pair (0.826, 0.821) and barely better than 2015 alone. The two
external cohorts pull in different directions and combining both dilutes rather
than reinforces. Use one external cohort, not both.

**The device probe explains why it works.** 2015 vs PADS:

| class | probe AUC | verdict |
|---|---|---|
| N | 0.805 | **confounded** |
| PD | 0.658 | **pass** |

The cohorts differ in how *healthy controls* were recorded but their *tremor
patients* look alike. N-vs-Tremor needs the tremor side to generalise, and it
does. That asymmetry was invisible until NewData gained HC and PD made
per-class probes possible — earlier probes tested ET only, the least separable
class, and produced the misleading "PADS cannot be pooled" verdict for every
axis rather than just PD-vs-ET.

## Position by axis

| axis | best configuration | result |
|---|---|---|
| **N vs Tremor** | **2015 + PADS, OUT** | **bal-acc 0.826, AUC 0.894, precision 0.889** |
| PD vs ET | 2015 alone, REST | bal-acc 0.730, precision 0.393 |

Merging helps N-vs-Tremor and hurts PD-vs-ET, consistently, across every
combination tried. The recommendation is therefore per-axis, not per-project.

**Before reporting:** the gain over 2015 + NewData (0.826 vs 0.821) is small
enough that a paired CI is needed before claiming PADS beats NewData. Both
clearly beat 2015 alone.

# Per-class precision, and fixing ALL THREE by balancing the cohorts

The binary table above hides the per-class picture. 3-class (N/PD/ET), OUT,
scored on the same 151 2015 patients:

| training cohorts | prec N | prec PD | **prec ET** | **macro P** | **macro F1** |
|---|---|---|---|---|---|
| 2015 only | 0.758 | 0.745 | 0.118 | 0.540 | 0.518 |
| 2015 + NewData | 0.750 | 0.769 | 0.097 | 0.539 | 0.517 |
| 2015 + PADS | 0.710 | 0.686 | 0.161 | 0.519 | 0.509 |
| ALL THREE (raw) | 0.725 | 0.680 | 0.156 | 0.520 | 0.509 |
| ALL THREE + per-cohort z-score | 0.581 | 0.667 | 0.164 | 0.470 | 0.337 |
| **ALL THREE, PADS capped 60/class** | **0.765** | 0.750 | **0.185** | **0.567** | **0.562** |
| ALL THREE, PADS capped 30/class | 0.736 | 0.736 | 0.154 | 0.542 | 0.534 |
| ALL THREE, both capped | 0.761 | **0.782** | 0.160 | **0.567** | 0.560 |

**The problem with ALL THREE was cohort size, not domain shift.** PADS
contributes 383 patients against 2015's 151 and NewData's 56, so it dominated
training. Capping it at 60 per class turns ALL THREE from the *worst*
configuration (macro F1 0.509) into the **best** (0.562), beating 2015 alone on
every macro metric.

**ET precision improves 0.118 → 0.185**, a 57 % relative gain and the best
3-class ET precision at OUT. Still low absolutely — and note 2015-only at
**REST** reaches 0.219, so REST remains the better condition for ET.

**Per-cohort z-scoring is harmful** (macro F1 0.337; PD recall collapses to
0.027). The same result appeared on PD-vs-ET: removing the domain offset does
not help, and here it destroys the PD class.

## The rule for merging

**Balance the cohorts; do not normalise them.** Every earlier three-cohort test
failed because PADS was allowed to contribute 4–7× the patients of the others.
That is a simpler and better-supported rule than the per-axis guidance above,
and it explains the earlier failures without appealing to domain shift.

Cap ≈ 60/class works, 30 does not — the optimum appears to be roughly 1–2× the
target cohort's per-class count. A proper cap sweep, and re-running the binary
N-vs-Tremor table with capping, are the obvious follow-ups.
