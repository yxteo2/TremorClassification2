# 2015 OUT: 40 % of the tremor patients show no tremor at OUT

Run: `python -m experiments.tremor_present_2015` (seconds; saved predictions from
`experiments.segments_2015`, 40 repeats, no refit).

**Tremor present** = log10 OUT tremor RMS (lower arm, 3-15 Hz) above the
controls' 90th percentile (−1.74), applied to every patient whatever their label.
A reporting stratification only: the model was trained and scored on all 151.

| class | tremor present |
|---|---|
| N | 6 of 61 |
| PD | **45 of 75 (60 %)** |
| ET | **9 of 15 (60 %)** |

| subset | arm | precN | precPD | precET | macroP | recPD | recET | ET prevalence |
|---|---|---|---|---|---|---|---|---|
| all 151 | scratch | 0.702 | 0.774 | 0.258 | 0.578 | 0.633 | 0.223 | 0.099 |
| all 151 | **ft** | 0.723 | 0.737 | 0.350 | 0.603 | 0.678 | 0.300 | 0.099 |
| tremor present (60) | scratch | 0.486 | 0.848 | 0.326 | 0.553 | 0.788 | 0.281 | 0.150 |
| tremor present (60) | **ft** | 0.485 | **0.826** | **0.382** | 0.564 | 0.779 | **0.400** | 0.150 |
| tremor absent (91) | scratch | 0.734 | 0.616 | 0.164 | 0.505 | 0.399 | 0.138 | 0.066 |
| tremor absent (91) | **ft** | 0.753 | 0.595 | 0.298 | 0.549 | 0.527 | 0.150 | 0.066 |

## Reading

* **Two in five PD and ET patients have no tremor above the control range in
  the postural task.** This is the ceiling `stack_2015.md` found among the
  consistent errors, quantified over the cohort.
* **Where tremor is present, PD is well classified** (`ft` precision 0.826,
  recall 0.78). Where it is absent, PD precision falls to 0.595 -- the model still
  finds about half of tremor-free PD (recall 0.53), from spectral shape beyond
  amplitude.
* **ET is not easier when tremor is present, relative to chance.** ET precision
  0.382 at prevalence 0.150 is 2.5x chance; over all patients 0.350 at 0.099 is
  3.5x. The stratification moves the base rate, not the separability.
* `ft` gains over scratch in both strata on ET (0.326 -> 0.382; 0.164 -> 0.298).

## For the paper

Report the all-patient headline **and** this split, and state that a tremor
classifier cannot score tremor-free patients from tremor. The threshold uses the
control distribution, not the outcome, and is fixed before scoring; an
alternative (e.g. the controls' 95th percentile) should be shown as a
sensitivity check if this table is published.
