# Method rules — each one learned from a wrong conclusion in this repo

The point of listing the incident is so the rule is understood rather than
obeyed. Every one of these produced a confident, wrong result before it was a
rule.

## Controls and attribution

**A matched control decides attribution; the plain baseline decides adoption.**
Peak alignment beat its random-shift control by +0.250 precET on one split while
*losing* to doing nothing by −0.050. The docstring had called the control "the
comparison that decides it". Beating a deliberately scrambled arm is a low bar.
Report both, label which is which.

**Never use a fixed random draw as a null control — re-draw it per split.** One
random 3-level label reused across all 20 splits happened to be ET-associated at
p = 0.051 (top 5 % of draws). Against baseline it manufactured precET +0.090
[+0.019, +0.168] from three meaningless columns, while the honest re-drawn
control reached nothing. A frozen draw's chance association is *constant* across
splits, so the paired bootstrap cannot see it — pairing removes split-to-split
noise, not a quietly informative fixed feature.

**Never judge a second model on a subset the first model's uncertainty defined.**
Conditioning on where model A is unsure selects patients A does badly on and
applies no such selection to B, so B looks good by construction. A logistic
regression scored 0.520 balanced accuracy on the contested set against the deep
model's 0.443 — and the gating experiment showed the LR carried no complementary
signal at all. Only comparisons to a selection-independent baseline (chance) are
valid on such a subset.

**Do not select architectures on the validation split.** Asked to weight a
one-vs-rest decomposition against the softmax, validation chose the arm that is
0.162 worse on ET precision in 20 of 20 splits. With ~8 ET patients per
validation fold it can tune two offsets; it cannot choose between models.

**When an arm changes two things, the baseline must change with it.** The SSL
study compared a frozen pretrained encoder against a fine-tuned random one and
reported +0.161 precET; against a frozen random encoder the effect was zero.
Welch-vs-multitaper changes estimator family *and* smoothing — the `nw` sweep
with family fixed is what showed the gain is an interior optimum, not "smoother
is better".

**A gain measured against a re-implemented baseline is a claim about the
re-implementation.** Log-frequency binning showed precN +0.031 against a
baseline at macroP 0.641; against the reported model at 0.660 it was −0.030.
Reproduce the reported number first — assert bit-exactness where possible.

## Reading results

**A rank correlation is the wrong instrument for a non-monotone effect.** The
estimator sweep printed Spearman −0.600 with a canned "smoother is better, as
predicted" line, while the table showed an inverted U peaking at the current
setting. Read the table. Distrust any canned verdict line, including
`influence_stable`'s "the dropped set is reproducible" at 15 % absolute overlap.

**A sign test is weak evidence when a magnitude ordering is available.** "Lower
frequency means more contested in every class, so it cannot be class confusion"
held only if class means straddled the range. They are monotonically ordered
(8.16 / 7.51 / 7.04 Hz), so confusion gives the same sign for N and PD and none
for ET — and the effect sizes (−0.385 / −0.241 / −0.051) fell exactly in that
order. Check whether the rival account predicts the magnitudes, not just the
signs.

**Relative safeguards cannot see a defect every arm shares.** The 1 % frequency-
axis stretch survived 68 reports because patient-level splits, paired
bootstraps and permutation nulls all compare arms. Then a synthetic-signal
sweep of every stage found two more that had also survived everything: a
Q-factor that spanned every supra-half-max bin instead of the peak (a tone with
a 0.8-amplitude harmonic read Q 0.94 instead of 15, and under the correct
definition PADS has *no* ET-vs-PD Q gap), and IF-trajectory end points that
were band-pass transients (2.7 Hz of noise on a 0.5 Hz signal, at points 0 and
63 of every patient's input). `experiments/verify_preprocessing.py` is the
absolute check; run it after touching `signal_processing/`, `frequency/` or
`common/cohorts.py`, and add a check whenever a stage is added.

**A synthetic test can be wrong too — check its own physics first.** The
quaternion path initially "failed" at 18.7 % error. The code was correct; the
test's Euler integrator advanced with the previous sample's rate, a half-sample
lag that the code's phase-free central difference faithfully reported. An
analytic quaternion sequence gave 2.35 %, exactly the central-difference bound.

**Below-chance AUC at small n is the low tail of a wide null, not
anti-prediction.** [0.298, 0.655] at 21 ET; [0.195, 0.819] at 6 ET. NewData once
gave AUC exactly 0.000 and it was not a finding.

**Precision is prevalence-dependent.** A cap sweep "showed" ET precision falling
with more PADS; it was the test fold's ET fraction moving from 7 % to 32 %.
Report prevalence or lift. Print the permutation null beside any in-house
precision — at 21 ET every model sits inside it, so differences describe where
the threshold fell.

**Sub-component gains do not compose.** Measured three times: a
measurement-derived two-stage prediction (#5), the short-window binary gains,
and a 33 % descriptor-level sharpness recovery that gave +0.007 precET. At this
n the model does not care about what a descriptor says it should.

## Predictions

**Write the prediction in the docstring before launching; append the outcome to
`reports/failed_predictions.md`.** Seventeen failed, eight held. The eight that
held were all derived from a *measurement of this dataset* (ensemble
disagreement, uniform axis distortion, mechanism checks, "small and uncertain"
calls for the onset trim and the Q fix, a guard on two class-agnostic points);
the fifteen mechanism stories all failed — most recently "the trajectory's gain
was its transient end points reading the PADS onset", which measured at rho
+0.03. Two supposed handicaps for the minority class — hard patients, ET's
diluted logit — turned out to be load-bearing.

**Design the experiment so either outcome is informative.** The band-edge test
was built so that a null *eliminated* one of two named accounts. A prediction
that only pays out when it holds is worth less.

**Narrow the prediction to what the mechanism actually implies.** "Aligned beats
random-shift if the mechanism is real" held; it was deliberately not a prediction
of improvement, and there was none.

## Physics and code

**Sanity-check physics, not just metrics.** The IF-trajectory extractor once
z-normalised away the quantity it existed to measure; no accuracy number looked
wrong. `‖ω‖` has fundamental 2f for a linear oscillation — use the principal-axis
projection. Measure an estimator's own resolution on a pure tone before
interpreting a bandwidth it reports (multitaper nw 2.5 has a Q ceiling of 5.33;
welch 15; ar16 31).

**Check what a reshape keeps.** `logbin` silently dropped 21 % of the band on
61-column input and nothing on 64-column input for the whole project.

**Pretraining corpora leak even without labels.** Hold the evaluated cohort out
of any unlabelled corpus or state the claim as transductive.

**A canned print line is not a conclusion.** Two scripts here print verdicts
that misread their own tables. Write the number; let the report interpret it.
