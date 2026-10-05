# Multi-segment tremor coupling (hand / lower arm / upper arm) on 2015 OUT

Run: `python -m experiments.multisegment` (feature level, minutes) and
`REPS=a-b python -m experiments.segments_2015` in parts, then
`python -m experiments.segments_2015 report` (deep model, `transfer_2015`
protocol, 40 repeats, CPU ~13 min per repeat).

## Why

IEEE TNSRE 2023 ("A Wearable Multi-Segment Upper Limb Tremor Assessment System
for Differential Diagnosis of PD Versus ET", DOI 10.1109/TNSRE.2023.3306203)
separates PD from ET by the **phase relationships between limb segments** and
loses 9-10 % when sensors are removed. 2015 and NewData record three arm
segments; the pipeline reads only the lower arm.

## Features (8, fixed before looking)

Each sensor's 3-axis angular velocity, band-passed 3-15 Hz and projected on its
own principal axis (sensor headings are not reliably shared, and the
rotation-invariant 3x3 coherence is biased to ~1 at ~7 Welch segments). At the
lower-arm tremor peak, +-0.5 Hz: coherence and |cos(phase)| for hand-lower,
lower-upper and hand-upper, plus log band-power ratios hand/lower and
upper/lower. Validated on a synthetic arm with known phase, coherence and gain
(all seven checked values recovered).

## Feature level (logistic regression, 5-fold x 20 repeats, permutation null)

| task, axis | 2015 | pooled with NewData |
|---|---|---|
| OUT, PD vs ET | **0.686**, null [0.307, 0.697], p = 0.045 | 0.551, p = 0.264 |
| OUT, N vs tremor | **0.786**, null [0.379, 0.614], p = 0.005 | -- |
| REST, PD vs ET | 0.409, p = 0.816 | 0.436, p = 0.697 |
| REST, N vs tremor | **0.693**, p = 0.005 | **0.616**, p = 0.005 |

On 2015 OUT the PD-vs-ET signal is driven by **hand-forearm coherence, lower in
ET** (univariate AUC 0.28). A mechanism fits: PD tremor is largely forearm
pronation-supination, which carries the hand with the forearm; ET is more wrist
flexion-extension, the hand moving against the forearm. NewData (6 ET) does not
show it (0.51), so pooled it is inside the null -- the same pattern as REST
frequency (`rest_replication.md`).

## Deep model: 2015 OUT, `transfer_2015` protocol, 40 repeats

`transfer_2015.fit` reused unchanged; the block is appended to the descriptor
stream. `seg_shuf` permutes the block's rows among the 151 patients, re-drawn
each repeat.

| arm | precN | precPD | precET | macroP | macroF1 | PD-vs-ET AUC |
|---|---|---|---|---|---|---|
| scratch | 0.702 | 0.774 | 0.258 | 0.578 | 0.561 | 0.572 |
| seg | 0.703 | **0.784** | 0.294 | 0.594 | 0.575 | 0.591 |
| seg_shuf | 0.696 | 0.772 | 0.258 | 0.576 | 0.561 | 0.585 |
| ft | **0.723** | 0.737 | 0.350 | 0.603 | **0.590** | 0.625 |
| ft_seg | 0.714 | 0.745 | **0.368** | **0.609** | **0.590** | **0.639** |

**Harness check.** ft - scratch reproduces `transfer_2015.md`: precET +0.092
[+0.040, +0.144] \* (there +0.084 / +0.094 \*), AUC +0.053 \* (there +0.055 \*).

Paired, 40 repeats:

| contrast | precPD | precET | macroP | macroF1 | AUC |
|---|---|---|---|---|---|
| seg - scratch (adoption) | +0.010 | +0.036 [−0.002, +0.076] | **+0.016 \*** | **+0.013 \*** | **+0.019 \*** |
| seg - seg_shuf (attribution) | **+0.012 \*** | +0.035 [−0.001, +0.083] | **+0.018 \*** | **+0.014 \*** | +0.007 |
| seg_shuf - scratch (control) | −0.002 | +0.001 | −0.002 | −0.000 | **+0.012 \*** |
| ft_seg - ft (on the adopted model) | +0.008 | +0.018 | +0.006 | −0.001 | **+0.015 \*** |

## Reading

* **On the 2015-only model, coupling is an attributable gain over a shuffled control,
  robust to re-partitioning but not patient-bootstrapped** (and below the ~0.03
  patient-level resolution): macroP
  +0.016 \* and macroF1 +0.013 \* over scratch, and the same against the
  shuffled control (+0.018 \*, +0.014 \*), which itself moves nothing but AUC.
  ET precision +0.035-0.036 sits just at the edge of its CI on both contrasts.
* **The AUC gain is not the coupling's.** The shuffled control lifts PD-vs-ET
  AUC +0.012 \* by itself, and seg beats it by only +0.007 (n.s.) -- extra
  columns move ranking; the real content moves decisions.
* **On top of transfer (`ft`) it adds nothing significant to precision**:
  precET +0.018, macroP +0.006, macroF1 −0.001. ft_seg is the best arm on
  precET (0.368), macroP (0.609) and AUC (0.639), but its AUC +0.015 \* over ft
  is the size the extra-column control produced and was not attribution-tested
  (no ft + shuffled arm). Transfer and coupling appear to capture overlapping
  information about which 2015 patients are ET.
* 20 repeats read seg - seg_shuf precET +0.064 \*; at 40 it is +0.035, n.s. --
  the halving on doubling seen repeatedly in this project.

## Verdict

* **2015 model without transfer:** add the coupling block (small, attributable
  macro gain; costs no PD precision: +0.010 n.s. vs scratch, +0.012 \* vs shuffled).
* **2015 model with transfer (`ft`, the adopted recipe):** coupling is **not
  adopted** -- no significant precision gain; the ranking gain is within what
  extra columns give.
* Feature-level PD-vs-ET from coupling is a 2015 result, not replicated on
  NewData's 6 ET.

## Prediction (in `segments_2015.py`'s docstring) -- failed

"seg beats seg_shuf on PD-vs-ET AUC by +0.02 to +0.05 (significant), but not on
ET precision." The reverse: AUC +0.007 (n.s.), while macroP / macroF1 / precPD
are significant and precET borderline. Registered as #37. The feature-level
prediction that coupling separates N from tremor at OUT held (AA).
