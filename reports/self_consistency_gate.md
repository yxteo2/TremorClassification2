# Recording agreement: descriptive evidence, not a label-noise bound

## Correction

The earlier **55% upper bound on the label-noise share of errors** and derived
**19% mislabelling bound are withdrawn**. The ratio
`(0.733 - 0.554) / (0.882 - 0.554)` is a normalised agreement contrast, not a
mixture proportion. No identified mixture model or validated component rates
justify interpreting it as a fraction or upper bound. Stable model errors can
occur with correct labels, and incorrect labels need not yield stable predictions.
Agreement cannot divide errors into label noise and signal insufficiency.

The earlier recommendation to split funding according to that ratio, the claim
that the remaining 45% cannot benefit from relabelling, and the estimated list of
50–60 patients are also withdrawn. The script did not export an adjudication list.
Repeated test-fold counts cannot be converted into a unique patient count.

## Historical output — incompatible with the corrected statistic

| repeat kind | all-recordings A_correct | all-recordings A_wrong | old pairwise control |
|---|---|---|---|
| same-arm (2015 + NewData) | 0.882 | 0.733 | 0.554 |
| PADS left/right | 0.773 | 0.643 | 0.501 |

These are archived observations, not corrected results. The original statistic
required **all** recordings to agree, but the control compared **two** recordings.
The control also failed to match cohort/class composition and correctness subgroup.
Consequently, the original observed-minus-control contrasts are not comparable.

## Corrected analysis — rerun pending

Run `python -m experiments.self_consistency_gate`.

- Compute the fraction of unordered recording pairs agreeing within each patient.
- Give every eligible patient equal weight, regardless of recording count.
- Pair different patients within exact cohort, recorded-label class and
  patient-level correct/incorrect subgroup; average recording-pair agreement
  within each patient pair, then weight stratum means by eligible patient count.
- Report all-patient agreement and **matched** observed/control agreement
  separately. Exclude singleton strata from both matched columns and report
  eligible and matched patient counts. Interpret each contrast only on its
  matched subset; sparse subgroups can have no estimable control.
- Keep same-arm and PADS left/right results separate. PADS includes bilateral
  differences, so it is not a same-arm repeatability estimate.

The new checkpoint `self_consistency_pairwise_v2` prevents old statistics being
silently reused. Confidence remains unadjusted maximum model probability and
has no matched control or calibration guarantee. The analysis still broadcasts
patient-level asymmetry descriptors to recording rows, so recordings are not
independent model inputs. Correctness-stratified comparisons are descriptive
and conditional on the fitted model, not causal evidence about diagnoses.

No corrected numerical result has been produced as part of this code correction.
An independently adjudicated sample, including unflagged patients, would be needed
to estimate label-error prevalence. Model disagreements may guide a review but
must not automatically change labels or remove difficult patients.
