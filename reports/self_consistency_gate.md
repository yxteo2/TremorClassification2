# The gate: about half the errors look like label noise, and it names which patients

## What was measured

Every patient has more than one recording and the pipeline averages them away.
This scores each recording **separately** under the reported model — trained
exactly as reported, recording rows passed as a third output matrix standardised
with the same train-fold statistics, so it measures the reported model rather
than a new one.

    label noise           -> model CONSISTENT with itself, disagrees with LABEL
    signal insufficiency  -> model INCONSISTENT on the same patient too

Controlled against agreement between recordings of **two different patients of
the same true class**, resampled per split — without that, the model's class
prior alone would manufacture high self-agreement.

## Result — 20 splits

| repeat kind | A_correct | A_wrong | control | conf_cor | conf_wrong | n_cor | n_wrong |
|---|---|---|---|---|---|---|---|
| **same-arm** (2015 + NewData) | 0.882 | **0.733** | 0.554 | 0.595 | 0.542 | 21.9 | 11.2 |
| PADS L/R | 0.773 | 0.643 | 0.501 | 0.599 | 0.538 | 23.8 | 16.2 |

|  | A_wrong − control | A_correct − A_wrong |
|---|---|---|
| same-arm | **+0.179** | **+0.149** |
| PADS L/R | +0.141 | +0.130 |

## The verdict: both accounts, in almost equal measure

On patients the model gets **wrong**, it still gives both recordings the same
answer **73 %** of the time, against **55 %** for two random patients of the same
class. So there is stable, patient-specific structure even where the model is
wrong — the label-noise signature. But A_wrong sits **0.149 below** A_correct,
so misclassified patients are also genuinely more ambiguous than correctly
classified ones — the signal-insufficiency signature.

Scaling A_wrong between the guessing floor (the control) and the working ceiling
(A_correct):

    same-arm   (0.733 - 0.554) / (0.882 - 0.554) = 55 %
    PADS L/R   (0.643 - 0.501) / (0.773 - 0.501) = 52 %

**~54 % of the way from "guessing" to "fully self-consistent", and the two very
different repeat types agree to within 3 points.** That concordance is the main
reason to trust the number: a same-arm retest and a two-limb comparison have no
reason to land together unless both are reading the same underlying property.

## The limitation that bounds the claim

**Self-consistency is not proof of mislabelling.** A patient can be consistently
misread because they are genuinely *atypical* — a PD patient with an unusually
tonal, ET-like tremor — in which case the model is consistently wrong and the
label is right. So:

> consistently wrong = {mislabelled} ∪ {genuinely atypical}

**~55 % is therefore an upper bound on the label-noise share of errors, not an
estimate of it.** Nothing in this data separates the two, and it should not be
quoted as "half our labels are wrong".

A second limitation, from a design gap: the confidence columns have **no matched
control**, unlike the agreement columns. conf_wrong 0.542 vs conf_cor 0.595 is a
small separation with no baseline to read it against, so those two columns
should carry little weight.

## Predictions, scored

1. *"Neither account wins cleanly — A_wrong clearly above the control but
   clearly below A_correct."* — **held, and closely.** +0.179 above, +0.149
   below; the two gaps are nearly equal.
2. *"PADS L/R self-agreement < same-arm self-agreement."* — **held.** 0.773 vs
   0.882 correct, 0.643 vs 0.733 wrong. Consistent with the PADS pair spanning
   two limbs and tremor being genuinely asymmetric.
3. *"The label-noise half will look stronger on confidence than on agreement."*
   — **failed.** Agreement carried the entire result (+0.179 against a control);
   confidence separated by only ~0.05 and has no control at all. The prediction
   also revealed the design gap in §"limitation" — I measured confidence without
   a baseline, so it could not have been decisive either way.

## A consistency check, offered as arithmetic not measurement

The model errs on ~35 % of patients. If ~55 % of those errors are the
consistently-wrong kind, the implied upper bound on mislabelling is
0.35 × 0.55 ≈ **19 %**, which lands inside the clinical PD-vs-ET misdiagnosis
band of 15–35 %. That is a rough cross-check arriving independently at the
literature's number, not a measurement — the upper-bound caveat above applies to
every term in it.

## What this changes about the plan

`data_plan.md` §0 asked this gate to choose between funding adjudication and
funding collection. **It does not choose — it says do both — but it converts
adjudication from a blanket exercise into a targeted one, which is the cheaper
half of the answer.**

* **Re-adjudicate the consistently-wrong set, not all 404.** The measurement
  identifies specific patients: those whose recordings *agree with each other*
  and *disagree with the label*. That is ~11 patients per test fold, so of order
  50–60 patients across the cohort — a tractable review list rather than a full
  re-read.
* **Re-adjudication also resolves the bound.** Every patient on that list
  resolves to either "label was wrong" or "label was right, patient is
  atypical". Doing the review therefore *measures* the label-noise share that
  this experiment can only bound — and atypical-but-correctly-labelled patients
  are themselves worth having identified, because `prune_training.md` showed
  the hardest patients are boundary-defining and must not be dropped.
* **Collection is still needed**, because the other ~45 % is genuine ambiguity
  that no amount of re-labelling touches. §1's in-house ET target stands.
* **Order it adjudication-first**, reversing the earlier draft's implicit
  ordering. The review is cheap, it is targeted, and its outcome changes the
  expected return on collection: if the list resolves mostly to wrong labels,
  collection with a better label protocol becomes much more valuable; if it
  resolves mostly to atypia, the ceiling is more physical than clinical and
  collection buys less than §1 implies.
