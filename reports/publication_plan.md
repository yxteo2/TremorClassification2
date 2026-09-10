# A Q1 direction, given what is actually established

## The honest starting point

**There is no performance contribution here, and the paper must not pretend
otherwise.** precET 0.654 / macroP 0.652 on 404 patients is below clinical
usefulness and below published claims — and this project has spent 76
experiments establishing that the gap is not a modelling failure. Seventy-six
experiments, ~25 closed families with matched controls, and every architecture
family tried is null or worse.

A paper whose headline is a number will be rejected, correctly. A paper whose
headline is **a measurement the field is not making** can go to a Q1 venue,
because the measurement is novel, cheap, transferable, and answers a question
every wearable-biosignal group has and none reports on.

## The recommended paper

> **How much of a classifier's error is the labels? A self-consistency test for
> repeated-measurement biosignal cohorts.**

**Primary claim.** When a dataset gives each subject more than one recording,
the model's *agreement with itself across those recordings* separates two
otherwise indistinguishable accounts of a performance ceiling — noisy labels
versus insufficient signal — without any additional data, annotation, or
model.

**The result that makes it interesting.** On PD-vs-ET, a contrast with no gold
standard and a documented 15–35 % clinical misdiagnosis rate, the two accounts
hold in **almost equal measure**: on patients the model gets wrong it still
gives both recordings the same answer **73 %** of the time against a **55 %**
same-class control, but that sits **0.149 below** its rate on correct patients.
Scaled between the guessing floor and the working ceiling, misclassified
patients are **~54 % of the way to fully self-consistent**.

**The internal validation that makes it credible.** Two structurally different
kinds of repeat agree to within 3 points — a same-arm test–retest (55 %) and a
two-limb left/right comparison (52 %). They have no reason to land together
unless both are reading the same underlying property.

**The actionable output.** The test does not just apportion; it *names* the
patients — those whose recordings agree with each other and disagree with the
label. ~50–60 of 404 here: a tractable re-adjudication list rather than a full
re-read.

## The two things that must be added before submission

### 1. Calibrate the gate against known label noise — **this is the paper**

As it stands the statistic is a plausible heuristic with a bound:
`consistently wrong = mislabelled ∪ genuinely atypical`, so ~55 % is an **upper
bound**, not an estimate. That is a fair reviewer objection and it is fatal to a
Q1 methods claim.

**Fix it synthetically, which costs one experiment.** Flip a *known* fraction
`k` of training-and-test labels (k = 0, 10, 20, 30 %), re-run the model, re-run
the gate, and plot the recovered statistic against `k`. If it tracks `k`
monotonically, the gate becomes a **calibrated instrument** rather than a
heuristic, and the observed 54 % can be mapped to an implied noise fraction with
an interval.

This is the single highest-value experiment left in the project. It converts
"we measured something suggestive" into "we validated a measurement and applied
it", which is the difference between Q2 and Q1.

*(A rough arithmetic cross-check already points the right way: 35 % error rate ×
55 % consistently-wrong ≈ 19 % implied mislabelling, inside the clinical
15–35 % band. Calibration would replace that arithmetic with a curve.)*

### 2. Partial clinical adjudication — turns the bound into an estimate

Have a neurologist review even **20 of the ~50–60 flagged patients**. Each
resolves to *label was wrong* or *label was right, patient is atypical*. That
directly measures the split the synthetic calibration models, and it is the
difference between a methods paper and a methods paper with clinical grounding.

If this is not feasible, the paper still stands on §1 — but say so explicitly
rather than leaving the bound unaddressed.

## The second pillar: a power analysis the field needs

`data_plan.md` §1 contains a result worth its own section:

| ET total | PD-vs-ET null p95 | precET step | resolution @ 40 splits |
|---|---|---|---|
| 21 (in-house) | **0.757** | 0.317 | 0.084 |
| 49 (this cohort) | 0.676 | 0.136 | **0.055** |
| 150 | 0.618 | 0.044 | 0.032 |

**At 49 ET, 40 splits resolves 0.055, while every candidate improvement measured
here lands at +0.02 to +0.04. The instrument is coarser than the effects being
chased.** At 21 in-house ET the PD-vs-ET chance ceiling is AUC **0.757** — higher
than most effects in the literature, so in-house PD-vs-ET is *unmeasurable*,
not merely hard.

This explains, quantitatively, why the wearable-tremor literature is full of
unreplicated small gains — and this project produced two of its own on the same
day, a +0.069 \* and a −0.032 \*, both of which evaporated when re-run. That is
an unusually honest and unusually useful thing to put in a paper.

## The third pillar: the negatives, as defence not headline

~25 closed families, each with a matched control, plus a **pre-registered
prediction register (24 failed / 16 held)** and explicit retractions. This is not
the contribution; it is what makes the ceiling claim credible. Every reviewer
will ask "did you try X?" — having a pre-registered answer for 25 X's, including
the ones that failed, is a strong defence and a rare one.

Include as a compact table + supplement. Lead with the two most striking:
* **85.8 M ImageNet parameters (frozen ViT, real weights) reach macroP 0.501
  where ten numbers off a Welch spectrum reach 0.619 and a 5 k model reaches
  0.652.** Capacity is not the axis.
* **Nine transformer/attention configurations from 4.9 k to 85.8 M, all null or
  worse**, with the reason stated as a measurement: the sequence is 16 bins,
  shorter than a 2-layer kernel-5 receptive field.

## The fourth pillar: reproducibility infrastructure

Two absolute verification suites, both of which caught real defects that every
relative safeguard missed:

* `verify_preprocessing.py` — **43 checks**, synthetic signals with known answers
  through every stage. Found a 1.05 % frequency-axis stretch, a Q-factor that
  spanned every supra-half-max bin, and IF-trajectory end points that were
  filter transients — all of which had survived 68 reports.
* `verify_data.py` — **12 checks** on the real recordings: no duplicate
  recordings, no cross-cohort subject-id collisions, units consistent within
  1.4×.

Plus the finding that motivates them: **relative safeguards — patient-level
splits, paired bootstraps, permutation nulls — cannot see a defect every arm
shares.** That is a genuinely transferable methodological point and belongs in
the discussion.

## What to leave out

* **The 0.654 headline as a claim.** Report it, frame it as the ceiling, never
  as a contribution.
* **Every architecture result as a positive.** They are evidence for the ceiling,
  not results.
* **The bag-of-frames finding**, unless it survives 40 splits *and* a comparison
  against the reported model rather than a standalone CNN. Right now it is
  macroP +0.027 \* over a weaker baseline with a **precET win rate of 0.45** —
  the exact pattern that has burned this project before.
* **In-house PD-vs-ET numbers of any kind.** The null reaches 0.757 at 21 ET.
  Report the floor, not a point estimate.

## Venue

| venue | fit | note |
|---|---|---|
| **IEEE JBHI** | **best realistic target** | methods + clinical data, values evaluation rigour, Q1 |
| IEEE TBME | good | more engineering; the power analysis fits well |
| IEEE TNSRE | good | tremor/neuro fit, slightly narrower |
| npj Digital Medicine | stretch | only with §2 clinical adjudication done |
| Movement Disorders | poor fit | wants clinical validation and n we do not have |

## Ordering

1. **Run the synthetic label-noise calibration** (§1). One experiment. Without
   it there is no Q1 paper; with it there is.
2. Request the 20-patient adjudication (§2) in parallel — it has the longest
   lead time and does not block drafting.
3. Draft around the four pillars above.
4. Only then decide whether the bag-of-frames result is a fifth pillar or a
   footnote — it will have 40 splits by then.

## The one-sentence pitch

*We show that a classifier's agreement with itself across repeated recordings
separates label noise from signal insufficiency without new data, calibrate it
against known noise, apply it to a contrast with no gold standard, and quantify
why the field's usual response — another architecture — cannot work at the
sample sizes in use.*
