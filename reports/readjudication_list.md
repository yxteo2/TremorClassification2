# Who the model is consistently wrong about — and why most of it is not label noise

## What was produced

`reports/readjudication_list.csv` lists, ranked, the patients that the **adopted
9-member ensemble** gets *consistently* wrong. A patient is flagged when three
things hold:

- the patient-level prediction disagrees with the label;
- every recording of that patient, scored separately, gets the same wrong class;
- this happens in at least two thirds of at least two test appearances across 20
  splits.

`self_consistency_gate.md` measured this population and `selective_90.md` showed
it cannot be abstained away, but neither recorded who is in it. **Nothing here
changes a label.**

**62 of 404 patients are flagged (15 %).**

## The breakdown contradicted the recorded prediction

| confusion (label → model) | flagged |
|---|---|
| **N → PD** | **26** |
| **PD → N** | **17** |
| ET → PD | 13 |
| ET → N | 5 |
| PD → ET | 1 |

The prediction was that PD↔ET would dominate. It does not: **43 of 62 flagged
patients (69 %) are N↔PD**, and only 14 are PD↔ET. ET is still over-represented,
at 29 % of flagged patients against 12 % of the cohort, which was predicted.

By cohort: PADS 34, 2015 17, NewData 11. NewData, the smallest cohort, has the
highest flagged rate at 11 of 56 (20 %).

## A model-free test says most of these are phenotype, not mislabels

Tremor amplitude is sum-normalised out before the model sees anything, so the
in-band RMS of the raw recording is an **independent** check on what the model
concluded:

| group | log10 in-band RMS | compared with | reference |
|---|---|---|---|
| **PD flagged → N** (17) | **−1.97** | other PD −1.70, p < 0.001 | **N median −1.94** |
| **N flagged → PD** (26) | **−1.78** | other N −1.97, p < 0.001 | **PD median −1.72** |
| ET flagged → PD (13) | −1.54 | other ET −1.45, p = 0.86 | — |

In-band fraction moves the same way: 0.66 vs 0.74 for PD → N (p = 0.003), and
0.67 vs 0.58 for N → PD (p = 0.004). The PD → N contrast replicates within 2015
(p = 0.011) and within PADS (p < 0.001).

**For the N↔PD 43, the model reads the recording correctly:**

- **PD → N (17):** these patients have PD with tremor at the *Normal* level. On
  PADS, where a rest recording exists (n = 14), they are also tremor-free at rest
  (−1.98 against N −2.04, p = 0.66). They look like PD without tremor, a
  well-known subtype. **No wrist-tremor recording of any task can classify them
  as PD.** The label is almost certainly right; the task cannot see their
  disease.
- **N → PD (26):** these controls have tremor at the *PD* level. That fits
  enhanced physiological tremor, with a correct label. Occasionally it may be an
  undiagnosed case; the recording cannot tell the two apart.

**Only the 19 ET-involved patients show no amplitude anomaly.** The 13 ET → PD
patients have ET-typical amplitude and a PD-like spectrum. They are the
candidates that fit the clinical misdiagnosis account, or genuine spectral
overlap.

## What this does to the label-noise claim

The gate called ~55 % an **upper bound** on the label-noise share of errors,
because a consistently-wrong patient could be mislabelled *or* genuinely
atypical. This measurement separates the two, at least in part, and most of the
flagged set lands on the **atypical** side:

    consistently wrong, amplitude-explained (phenotype)   43 / 62   69 %
    consistently wrong, no amplitude explanation          19 / 62   31 %
                                                          = 4.7 % of the cohort

**The label-noise bound tightens a lot.** At most ~19 patients (~5 % of the
cohort) are consistently wrong without a physiological explanation, well below
the ~19 % implied mislabelling the gate's arithmetic allowed. Consistently-wrong
errors are one part of all errors; this bounds that part rather than every
error. It is still a bound: amplitude explains a phenotype but does not prove the
label right.

**The ceiling is mostly phenotype–task mismatch.** Patients with PD but no
tremor, and controls with tremor, are indistinguishable from the other class in
a wrist-tremor recording. That is an established clinical phenomenon, measured
here without a model, and it is a stronger account of the ceiling than label
noise.

## What it means for improving performance

- **The 17 PD → N patients are out of reach for any tremor recording**, postural
  or rest. Recovering them needs a non-tremor PD marker (bradykinesia, rigidity,
  for example a finger-tapping task). The fix is new data, not a new model. I
  did not train a rest-aware model: the rest test above rules it out.
- **The clinician's review list shrinks from 62 to 19.** Re-adjudicate the 19
  ET-involved patients. For the 43 N↔PD patients the question is a chart lookup
  (does this PD patient have tremor? is this control's tremor physiological?),
  not a re-diagnosis.
- **Say what the classifier measures.** It recognises a *tremor phenotype*, and
  maps that to a diagnosis only where the two coincide. That framing is more
  honest and more defensible in a paper than reporting a diagnostic ceiling.

## Predictions, scored

1. *"40–70 flagged"*: **held** (62).
2. *"ET over-represented relative to its 12 % share"*: **held** (29 %).
3. *"the dominant flagged confusion is PD↔ET, not either-vs-N"*: **failed**.
   N↔PD is 43 of 62. The reasoning assumed that because the binary N-vs-Tremor
   screen reaches 0.91–0.92, N-vs-PD confusion must be rare among hard cases.
   That confuses a screen's overall precision with the make-up of its residual
   errors: a small error rate can still be concentrated in one direction.

## A hazard found along the way

2015 subject ids carry the task suffix (`ET 10_OUT` vs `ET 10_REST`), so any
cross-task join on 2015 matches nobody unless the suffix is stripped. The rest
test above therefore covers PADS only; 2015 contributes just 3 PD → N patients,
so the conclusion does not change.
