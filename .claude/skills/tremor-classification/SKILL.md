---
name: tremor-classification
description: Work inside the TremorClassification2 repo — classifying wearable IMU recordings as Normal (N), Parkinson's (PD) or Essential Tremor (ET) across three cohorts. Use whenever the user loads tremor recordings, computes time-frequency features (STFT/CWT/multitaper/HHT/SST/wavelet-packet), works on mean/max frequency or tremor characteristics, builds or trains the CNN/TCN/BiLSTM/two-stream models, merges the 2015 / NewData / PADS cohorts, runs patient-level splits, or debugs low ET precision. Trigger on any reference to quaternion data, angular velocity, spectrograms, tremor stability, or the `models` / `signal_processing` / `frequency` / `common` / `metrics` / `experiments` packages — do not reinvent conventions the repo already fixes.
---

# Tremor Classification (TremorClassification2)

Three-class N / PD / ET classification from wearable IMU, across three cohorts.
**ET is the constraint**: 49 patients across all three cohorts. When a trade
exists, optimise **per-class precision** — especially ET precision — not accuracy.

## Package layout

```
models/             architectures.py — every model (CNN/TCN/BiLSTM/two-stream/attention)
signal_processing/  transforms, tfd, quaternion, preprocessing, spectral,
                    stability (Tremor Stability Index, IF trajectories), signal_features
frequency/          characteristics (mean/max freq), descriptors, tables, biomarker, report
common/             data, quaternion_data, load_2025, extract_pads, loaders,
                    cohorts (merge assembly), training (loops), protocol (splits+priors),
                    datasets, cache
metrics/            evaluate, stats, selective (precision at coverage), benchmark, merged
experiments/        final_model.py, audio_techniques.py — runnable studies
*.ipynb             01_tremor_characteristics, 02_deep_model (repo root)
```

Entry points: `python -m frequency.characteristics`, `python -m experiments.final_model`.

## Non-negotiable invariants

1. **Patient-level splits only.** Every recording of one patient stays in one
   fold. In the merged tables each row IS a patient, so `StratifiedShuffleSplit`
   on the row index is already patient-disjoint.
2. **Split first, then normalise.** Fit scaling on the training fold only.
3. **Augmentation and oversampling on the TRAIN fold only.**
4. **Report per-class precision with the test set's class prevalence.**
   Precision is not comparable across differently-composed test sets — see the
   prevalence artifact below.
5. **Paired bootstrap CIs for every comparison.** Unpaired differences of ~0.04
   sit inside the per-config sd here and have repeatedly evaporated when paired.
   The split-level bootstrap is sound for this: on the merged 20-split protocol it
   comes within 1.1x of a patient-level bootstrap, because every patient is tested
   ~4 times under different fold compositions (`patient_level_ci.md`).
6. **Raise the split count before believing a difference under ~0.03.** 20
   splits resolves ~0.04, 40 resolves ~0.025. A paired bootstrap over 20 splits
   removes the fold-composition noise the two arms share but NOT the noise in how
   much that particular set of folds favours one arm: a paired macroP +0.021
   [−0.006, +0.048] became +0.005 [−0.020, +0.028] on doubling
   (`early_fusion_confirm.md`). Also note precET has sd 0.183 across splits, so
   the reported **0.685 carries about ±0.04**; at 40 splits the same model gives
   0.663.
7. **Use a PERMUTATION null for any single-model claim.** Bootstraps hold the
   fitted model fixed and cannot see fitting variance, which dominates at 21 ET.
   The in-house PD-vs-ET null spans [0.298, 0.655] — nothing below AUC 0.66 there
   is distinguishable from chance (`permutation_null.md`).

## Cohorts

| cohort | n | N / PD / ET | notes |
|---|---|---|---|
| 2015 | 151 | 61 / 75 / 15 | `Data/`, 3 sensors, single limb, OUT/REST/WING |
| NewData | 56 | 27 / 23 / 6 | 2025 Moveo, both limbs, 7 tasks |
| PADS | 383 | 79 / 276 / 28 | both wrists, 100 Hz, 10 tasks (2 extracted) |

**PADS labels must be exact-matched.** A substring match once put 13 non-ET
records (etiology, asymmetric, Retrocollis, hypokinetic) into ET, and 20 PD
records are Atypical Parkinsonism. `common/extract_pads.py` re-derives class
from the manifest by exact diagnosis. Strict mode gives 79 / 276 / 28.

**NewData is a training cohort, not an evaluation one.** At 6 ET, a 20 % test
split holds ~1.2 ET patients; per-cohort ET metrics there are one-patient
artifacts (an identical 0.100 appeared across three structurally different
models).

## How to merge the cohorts (settled)

* Cap PADS at **90/class**, pool, one **global** set of validation-tuned priors.
* **No distribution alignment.** `spectrum_table` output is already
  scale- and rotation-invariant; the cohort probe sits at |acc − majority|
  = 0.003–0.035 with nothing applied. z-score, rank and CORAL all RAISE it to
  0.24–0.48, because centring each cohort on its own class mixture (PADS is
  72 % PD) injects the signature they were meant to remove.
* **Merge on the postural task** (OUT / OUT / StretchHold), not REST:
  LOCO macroF1 0.451 vs 0.399.
* **Dropping PADS is catastrophic**: ET precision 0.519 → 0.065.
* **Sample weighting does not substitute for capping** (0.492 vs 0.649). Uncapped,
  precPD rises to 0.742 while precET collapses to 0.221 — PADS at 72 % PD turns
  the model into a PD detector regardless of loss weights.
* **Transfer learning from PADS is the worst thing tried** (precET −0.188
  [−0.315, −0.088]). PADS is where the ET signal lives; pooling is correct.

## What works

**Frequency characteristics solve N-vs-Tremor.** Six numbers
(`frequency/characteristics.py`) + logistic regression reach **precision 0.910
(2015) / 0.924 (PADS)**. `bandwidth` contributes more than mean frequency.
`peak_sharp` is the standout descriptor: ET 12.19, PD 5.80, N 4.08 on PADS — ET
tremor is close to a pure tone.

**PD-vs-ET works on PADS and is UNMEASURABLE in-house.** On PADS, five of six
families beat a permutation null (descriptors 0.794, spectrum 0.791, stability
0.757, harmonics 0.726, ampmod 0.700, all p ≤ 0.010; `axes` p = 0.279). In-house
**not one family reaches significance** — best is `axes` p = 0.085
(`permutation_null.md`).

The permutation null for in-house PD-vs-ET AUC spans **[0.298, 0.655]** at 21 ET,
so an in-house model must reach **AUC ≈ 0.66 to be distinguishable from chance at
all**; the best measured is 0.629. Quote that detection floor rather than a
precision figure — it depends on neither prevalence nor threshold.

The earlier claim that "on 2015 every frequency feature is BELOW chance
(0.29–0.32)" is **withdrawn**: 2015 descriptors measure 0.492, p = 0.995. There is
no below-chance effect to explain. Never report one cross-cohort PD-vs-ET
frequency number.

**The final deep model** (`experiments/final_model.py`): two-stream —
`Spectrum1DCNN` on the log-binned **multitaper** spectrum plus `TrajectoryEncoder`
(dilated TCN) on the instantaneous-frequency trajectory, soft-voted with
`ResidualTCN`. **Re-audited at 40 splits and it held** (`headline_audit.md`):
precN 0.651 / precPD 0.654 / **precET 0.663** / **macroP 0.656**, paired
**+0.043 [+0.024, +0.062]** macroP and **+0.097 [+0.047, +0.146]** precET over the
welch baseline, winning on 75 % of splits. The interval **tightened** from the
original 20-split [+0.014, +0.067] — the opposite of early fusion.

Both ranked components survive at 40 splits: trajectory precET +0.068
[+0.030, +0.110] *, transform-alone precPD +0.032 * / macroP +0.020 *, and the two
sum to the whole (+0.020 + +0.022 vs +0.043 measured).

**Quote precET 0.663, not 0.685** — sd is 0.183, so the 20-split figure carries
about ±0.04.

Ranked by measured contribution:

1. **Input representation.** Log-scale and coarse-bin the spectrum to 16–32 bins.
   61 bins at n=404 is 15 % of the sample count; every model collapses there
   (TCN 0.505 at 16 bins vs 0.412 at 61).
2. **Validation-tuned class priors** — per-class logit offsets fitted on val,
   applied to test. ET precision 0.475 → 0.612, the single largest gain.
   **Leave `tune_offsets` alone.** It maximises validation macro F1 while the
   target metric is precision, and that mismatch is load-bearing: optimising
   precision directly lowers macroP and doubles its spread, because the
   validation split holds ~11 ET patients (`prior_objective.md`).
3. **Instantaneous-frequency trajectory** (+0.056 precET paired).
4. **Transform choice**: multitaper over welch — but read `band_truncation.md`
   first. That ranking was confounded with band coverage (multitaper was binned
   from 64 columns, welch from 61, and the old `logbin` dropped the remainder).
   With coverage equalised, on **N-vs-Tremor the estimator is worth almost
   nothing** (0.780 vs 0.797 PADS, 0.829 vs 0.833 in-house); on **PD-vs-ET it
   inverts by cohort** — multitaper better on PADS (precET 0.391 vs 0.339),
   significantly worse in-house (0.169 vs 0.250, −0.057 [−0.086, −0.029]).
   HHT is the WORST of eight.
5. Residual connections in the TCN; bilateral asymmetry as a missing modality;
   attention pooling for the BiLSTM (+0.012, does NOT transfer to the TCN).

## What does not work — do not re-try without new evidence

| tried | result |
|---|---|
| mixup | worse in 7 of 9 configs |
| spectrum augmentation (shift+noise) | paired CI spans zero |
| SpecAugment frequency masking | −0.021 |
| rich descriptors (10 → 34) | 0.584 vs 0.583 |
| two-stage hierarchy | 0.568 vs 0.583 |
| minibatch training | worse at every batch size |
| 1-D ResNet | 0.571 vs ResidualTCN 0.587 |
| ImageNet backbones (ResNet18/WideResNet/ViT) | at chance; ResNet18 from scratch trades macroF1 for precET |
| HHT / hht_imf2plus | worst of eight transforms |
| averaging two PADS tasks | precET 0.585 vs 0.612 |
| per-cohort priors | significantly worse (overfits ~11 val patients) |
| cohort-ID input | best mean, CI spans zero, sd nearly doubles |
| frequency-aware conv (CoordConv/FDY) | +0.010, not significant |
| masked-spectrum SSL on 3,081 unlabelled recordings | **no transferable benefit** — every cohort-held-out arm is flat or negative; the apparent gain is transductive (`ssl_retraction.md`) |
| MIL attention / max pooling over a patient's recordings | significantly WORSE than the uniform mean they replace (precET −0.117, −0.147) — `mil_recordings.md` |
| averaging non-postural tasks into the spectrum | precN **+0.047** * but precET **−0.104** * — PD is a rest tremor and ET a postural one, so averaging conditions deletes the PD-vs-ET contrast (`task_averaging.md`) |
| rest-vs-postural CONTRAST features (ratio, per-band, appended or substituted) | all significantly worse, precET −0.061 to −0.106 (`rest_postural_contrast.md`) |
| principal-eigenvalue spectrum (λ₁ of the per-frequency cross-spectral matrix) | macroP −0.000. The physics verifies (synthetic SNR 39.7→80.5, rotation-invariant to 6e−16) but sum-normalisation discards a gain that lives in absolute amplitude (`spectral_representation.md`) |
| polarisation spectrum (λ₁/trace per frequency) | worst arm tried, macroP −0.020, sd 0.089 |
| FiLM conditioning / channel gating of the TCN by descriptors | +0.019 and −0.004 macroP, neither significant (`tcn_fusion.md`) |
| early-fusion TCN (descriptors as broadcast input channels) | beats LATE CONCAT in a matched trunk (macroP +0.036 *) but **does not beat the reported model** — +0.021 at 20 splits collapses to +0.005 [−0.020, +0.028] at 40 (`early_fusion_confirm.md`) |
| TCN over TIME on the raw waveform | macroP **−0.034 [−0.066, −0.004] ***; soft vote with the reported model −0.024 (`time_domain_deep.md`) |
| TCN on analytic channels (envelope + IF stability) | precET **−0.192 ***, macroP **−0.076 ***; the worst deep input tried |
| tuning the class priors for macro precision (the target metric) | macroP −0.049 and sd 0.068 → 0.149; F1's recall term is regularising the offset search (`prior_objective.md`) |
| refining the offset grid 9×9 → 21×21 | slightly worse; coarseness is regularisation |
| fine-tuning a small encoder at ≤28 minority patients | destroys it — frozen beats fine-tuned by precET +0.161 on PADS |

**Feature unions dilute.** Eight have underperformed their best member
(concat+asym 0.554 vs 0.709; descriptors+stability 0.754 vs 0.807;
multitaper+traj+stability 0.639 vs 0.660). At 404 patients with 49 ET,
**dimensionality binds harder than information**. Prefer replacing a feature
family over appending one.

**Two unions do work**: `axes + stability` and `logreg + one-class Mahalanobis`
rank-averaged (`oneclass_hybrid.md` — in-house precET +0.023 [+0.005, +0.042] at
21 ET). Both are **two-member** and both members are individually decent.

An earlier version of this file generalised that to "combine at the score level
when the models differ in kind, at the feature level almost never". That is too
broad — see `score_vs_feature_fusion.md`, which tested it on all seven families:

* score-level beats feature-level on PADS and MERGED but **loses in-house**;
* **neither beats the best single family within one cohort** (PADS −0.063 precET,
  in-house −0.138, both significant) — averaging in members that are near chance
  on that cohort drags the good one down, and at 7 members it holds 1/7 weight;
* combination pays **only when no member dominates**: on MERGED `rank-avg ALL` is
  the best model tried, AUC +0.065 [+0.059, +0.071] over the best family, though
  its precision edge (+0.011) is not significant;
* AUC-weighting recovers about half the loss and is the safer default;
* the one-class member helps as 1 of 2 and is neutral as 1 of 8 — the ingredient
  was two strong dissimilar members, not the one-class model.

**Corrected rule: combine when the members are comparable in strength and you
cannot tell in advance which will win; do not combine when one dominates, and
adding members makes it worse.**

## Measurement traps that have produced wrong conclusions here

* **Precision is prevalence-dependent.** A cap sweep appeared to show ET
  precision falling monotonically with more PADS (0.282 → 0.156). It was an
  artifact: capping also shrank the PADS *test* fold from 7.3 % to 31.8 % ET.
  With the test cohort fixed, the trend reverses. Always report prevalence, or
  lift = precision / prevalence.
* **One fold split proves nothing.** Split-to-split sd is 0.008–0.025 here. A
  single split said "BiLSTM loses to logreg"; over 5 splits it wins.
* **Pooled k-fold overstates deep models on merged cohorts.** The pooled-to-LOCO
  gap grows with capacity (−0.037 logreg → −0.149 CNN). Report LOCO for any
  generalisation claim.
* **Two protocols answer different questions.** Mixed-cohort (all sources in
  train/val/test) = "how well at sites we trained on"; LOCO = "will it transfer".
  Cohort-ID as an input is legitimate in the first and leaks in the second.
* **Window-level metrics do NOT inflate results here** — tested because a paper
  reports them; window-level is LOWER than patient-level on N-vs-Tremor.
* **When an arm changes two things, the baseline must change with it.** The SSL
  study compared a *frozen* pretrained encoder against a *fine-tuned* random one
  and reported +0.161 precET for pretraining. Against a *frozen* random encoder
  the effect is zero — the whole gain was freezing. A frozen treatment needs a
  frozen control (`ssl_retraction.md`).
* **A frozen random encoder plus a linear head IS a linear model.** It reproduces
  logistic regression on the same spectrum to three decimals (precET 0.371, AUC
  0.785 both ways). Any "deep" result at that configuration should be checked
  against logreg before it is believed to be deep.
* **Unlabelled pretraining corpora leak.** Pretraining on a corpus containing the
  test patients' recordings is transductive even with no labels. Hold the
  evaluated cohort — or the fold's patients — out of the corpus, or state the
  claim as transductive.
* **Check what a reshape actually keeps.** `logbin` dropped 21 % of the band for
  61-column input (`X[:, :61//16*16]`) and nothing for 64-column input, so the
  same line was exact on one path and lossy on another for the whole project.
* **A patient bootstrap over fixed CV predictions is anti-conservative.** It
  resamples patients while holding the fitted model's out-of-fold scores fixed,
  so it cannot see the variance of the fitting procedure — which dominates at 21
  ET. It produced three confident "ANTI-predictive" verdicts that a permutation
  test calls ordinary chance (p = 0.41–0.44). **Use a permutation null for any
  single-model claim**; keep the bootstrap for paired differences, where the
  shared fold noise cancels.
* **Below-chance AUC at small n is almost never anti-prediction.** It is the low
  tail of a very wide null — [0.298, 0.655] at 21 ET, [0.195, 0.819] at 6 ET.
  NewData once gave AUC exactly 0.000 (all 6 ET below all 23 PD) and it was not
  a finding.
* **Sanity-check physics, not just metrics.** The IF-trajectory extractor
  initially z-normalised away the fluctuation magnitude it existed to measure; a
  stable and a wandering 6 Hz tremor came out identical. No accuracy number
  looked wrong.

## Transfer

**PD-vs-ET does not transfer between cohorts, in either direction**
(`pd_vs_et_transfer.md`). Fit on PADS, test in-house: not one family's CI excludes
0.5, and `descriptors` falls from AUC 0.794 within PADS to 0.519 in-house. Fit
in-house, test PADS: same. Three of 14 cells have intervals excluding 0.5 with
lower bounds of 0.52-0.53, which is what 14 tests produce by chance.

This explains why adding PADS to in-house training *degrades* in-house PD
precision (−0.082): the boundary PADS teaches does not hold in-house.

**Never present the PADS PD-vs-ET number as a method that works.** It is a result
about PADS.

## Axis-specific inputs (the one deep-learning gain that held)

The two decisions want **opposite inputs**: N-vs-Tremor benefits from every
recording, PD-vs-ET needs the postural condition kept clean. A flat 3-class model
must pick one and pay the other cost.

Two-stage with **all tasks for stage A, postural only for stage B**
(`axis_specific_inputs.md`): macroP **0.671** — the highest merged macro precision
measured here — with precN +0.058 *, precPD +0.050 *, macroF1 +0.025 *.

**But**: the macroP gain is +0.011 [−0.017, +0.040], **not significant**, and
precET is −0.075 [−0.160, +0.007]. Adopt only if the objective is macroF1 or
N/PD precision. **It does not help ET**, and it cannot: stage B is unchanged, so a
better gate just admits more tremor patients and enlarges the ET denominator.

The hierarchy alone does nothing (macroP −0.008), reproducing the earlier
two-stage negative. The inputs are the active ingredient.

## Input representation, revisited

**Log-frequency binning is RETRACTED.** It measured precN +0.031 * against an
STFT-derived baseline (macroP 0.641), but that baseline is weaker than the
reported multitaper one (0.660). Against the reported model it is **significantly
worse**: precET −0.086 [−0.146, −0.029] *, macroP −0.030 [−0.049, −0.012] *
(`spectral_representation.md`). A gain measured against a re-implemented baseline
is a claim about the re-implementation — check the baseline reproduces the
reported number first.

**An SNR gain that lives in absolute amplitude is invisible here.** Every
spectrum is sum-normalised per patient, so a uniform enhancement of the signal
band is removed before the model sees it. This has now discarded two physically
correct quantities: the principal-eigenvalue spectrum, and the within-patient
rest/postural amplitude ratio (`rest_postural_contrast.md`). Check whether a
proposed gain survives normalisation before building it.

## Time-frequency: window length is a real knob, but it splits by model

Every transform collapses its TF surface to the frequency marginal (`P.mean(0)`)
and **window length was never swept** — all are pinned at nperseg 256 or 512.
Sweeping it moves PADS PD-vs-ET AUC by 0.11 (`tf_window_length.md`).

**A spectrum built as the mean of 0.64 s STFT frames, 16 bins**, paired over 30
repeats against the current multitaper 16:

* **logreg PD-vs-ET**: PADS AUC +0.036 *, precET **+0.088 [+0.073, +0.102] ***,
  macroP +0.049 *; MERGED precET +0.032 *. Best PD-vs-ET representation measured.
* **the reported 3-class deep model**: **macroP −0.033 [−0.057, −0.007] ***,
  losing on 77 % of splits. **Do not swap it in.**

**Resolved**: re-run on the binary axis, the **CNN gains too** (PADS AUC +0.016 *,
MERGED precET +0.013 *), about half the linear magnitude. So it is a **task**
effect, not a model one — short-window is better for PD-vs-ET under both model
families, and also better for N-vs-Tremor (logreg AUC 0.774 → 0.810).

**It is a binary-axis representation and its gains do not compose.** A two-stage
model that decomposes the 3-class problem into exactly those two winning axes is
*also* significantly worse (macroP −0.033 *, precET −0.106 *), landing exactly
where the flat short-window model did. **Why the 3-class model loses is
unexplained**: four mechanism stories and one measurement-derived prediction were
all tested and all failed.

**Sub-component gains here are not evidence about the composite task.** Do not
report a binary-axis improvement as if it will carry into the 3-class model —
it has now been shown not to, under both flat and hierarchical combination.

**And it is PADS-dependent.** On in-house patients alone it does nothing:
PD-vs-ET AUC 0.556 vs 0.557 (both p ≈ 0.46, inside the null), and the in-house
3-class protocol shows no significant difference on any column
(`inhouse_shortwindow.md`). Both cohorts where it won contain PADS.

**A precision trap worth remembering.** In that in-house binary table precET reads
0.190 against 0.286 — a 50 % relative jump — while ΔAUC is 0.001 and both models
are indistinguishable from chance. **Always print the permutation null beside
in-house precision**: at 21 ET every model measured sits inside it, so in-house
precision differences describe where the threshold fell, not skill.

**Coarseness accounts for much of it on PADS**: multitaper at **8** bins alone
gives AUC 0.818 vs 0.798 at 16 — the top-ranked "coarse-bin" lever pushed further.
On MERGED coarseness explains the precision gain and the short window adds only
ranking.

**Spectral variability features are a settled negative** — per-bin IQR, spectral
flux and peak wander are the weakest arms at every window on both cohorts
(iqr at 2.56 s is AUC 0.494, chance) and dilute the median when appended.

**Refuted along the way**: "a median over frames is robust to transients" — a
**mean** does as well or better, so the estimator is irrelevant.

## The spectrum is a near-sufficient statistic at this n

Two learned time-domain models were built and both fail (`time_domain_deep.md`):
a TCN on the band-passed waveform (macroP −0.034 *) and a TCN on the analytic
channels (macroP −0.076 *, precET −0.192 *). Votes with the reported model are
neutral-to-negative.

**Two causal explanations for that were offered and BOTH were tested and
refuted**: demodulation cost (the analytic stream is worse, not better, than the
raw waveform) and median-centring removing absolute frequency (restoring it gives
macroP +0.006 [−0.018, +0.029] against a pre-registered prediction of ~+0.042).
At this n, a representation's *content* predicts performance far less well than
whether the model must estimate anything from these patients.

**The informative contrast is against catch22.** Both read the same temporal
structure from the same band. catch22 *ties* the spectral descriptors; a learned
TCN on the same information is significantly worse. The difference is that
**catch22 does no learning on the time axis** — its formulas were fixed offline on
93 unrelated datasets, while a TCN must estimate temporal filters from 404
patients with 49 ET.

> Time-domain information is only reachable here through estimators that do not
> have to be learned from this cohort.

That also explains why the reported model's IF-trajectory stream works (+0.056
precET): it is a closed-form 64-point summary, not a learned representation of a
384-point sequence.

**Every time-domain representation is lower-variance and lower-accuracy** than the
spectral one — sd(macroP) 0.044 and 0.051 against 0.068, with means 0.584 and
0.626 against 0.660.

Processing rules if this is revisited: use the **principal-axis projection, never
the magnitude** (‖ω‖ has fundamental 2f for a linear oscillation — verified,
11.91 vs 6.05 Hz); pick a crop length that **pads nothing**, since padding amount
is a cohort signature; and do **not** standardise waveform inputs per feature.

## Where this sits against the literature

* **PADS published baseline: 72.42 % balanced accuracy for PD vs DD**, 91.16 % for
  PD vs HC (Varghese, npj Park Dis 2024). This repo's PADS PD-vs-ET AUC 0.794 is
  the same regime — the difficulty is not an artifact of this pipeline.
* **Häring (Mov Disord 2025) is the credible target: 81.8 % accuracy for PD vs ET**
  on 414 patients from massive time-series feature extraction, vs 70.4 % for TSI.
  Their TSI baseline matches ours (AUC 0.757 on PADS), so the gap is real.
* **A 2026 arXiv preprint claiming 87.04 % on PADS PD-vs-DD uses class-dependent
  window overlap** (70 % HC / 0 % PD / 65 % DD). Preprocessing that reads the
  label is not a like-for-like comparator; do not quote it as a target.

## catch22: a temporal family that ties the spectral one

`signal_processing/catch22_features.py`. Every other family here comes from the
power spectrum or the IF trajectory; catch22 reads the **waveform**. Six features
encode Häring's mechanism (discrete stable oscillator states in PD vs one
pacemaker in ET) and were fixed a priori from that paper.

* **PADS: state subset (6 features) AUC 0.798 vs descriptors 0.794** — a tie on
  60 % of the dimensions and **half the fold variance** (sd 0.012 vs 0.023).
  `SB_MotifThree_quantile_hh` is the best single catch22 feature (Cohen's d 1.235).
* **Rank-avg hybrid gives AUC +0.014 [+0.008, +0.019] *** — the combination rule
  firing exactly as predicted (comparable strength, different kind, score level).
* **But precET is significantly WORSE everywhere** (−0.028 * hybrid, −0.034 *
  state alone). AUC integrates the whole ranking; precision at 9 % prevalence
  reads only its top, and rank-averaging dilutes descriptors' most confident ET
  calls. **Descriptors remain the best model by ET precision.**
* Concatenation diluted twice more (15 instances now).
* Use the **principal-axis projection**, never the magnitude: for a linear
  oscillation ‖ω‖ has fundamental 2f (verified — 11.91 Hz vs 6.05 Hz on a 6 Hz
  synthetic).

## The rest-tremor axis is missing from these recordings

The strongest clinical PD-vs-ET sign — rest tremor present in PD, absent in ET —
**is not measurable in this data**. The within-patient postural/rest band-power
ratio is *positive* for PD (+0.837, 74 % of patients), the opposite of the
textbook direction, and gives PD-vs-ET AUC 0.579. Every contrast feature built on
it makes the model worse (`rest_postural_contrast.md`).

Two independent attempts have now failed: this, and `reemergent_tremor.md`
(onset latency ≈ 0.000 s for every class). Both target rest-tremor phenomena.

Likely cause: none of the three protocols uses a **distraction-based** rest
condition (limb supported, patient counting backwards), and PD cohorts are
typically recorded ON medication. This is the single most specific data-collection
recommendation the project can make, and it may be cheaper than recruiting more
ET patients.

Practical consequence: **merge and evaluate on the postural task only.** Every
model here separates PD from ET using the *shape* of the postural tremor alone,
which is why the axis is hard.

## Ceilings

**Macro precision > 0.90 on three classes is not reachable at any coverage** —
abstaining on 80 % of patients still gives 0.73. ET precision gets *worse* under
abstention (0.630 → 0.462): confidence does not track correctness for the
minority class, so no thresholding scheme rescues ET.

The levers that would actually raise it, in order:
1. more ET patients;
2. the PADS non-motor questionnaire (the published 91 % baseline is multimodal;
   this pipeline is IMU-only by the user's choice);
3. PADS's 8 unextracted tasks — worth extracting to **test** whether the kinetic
   ones separate ET, which is an open hypothesis and **not** the measured fact
   this entry used to assert (`kinetic_task_audit.md`).

   The old wording — "the kinetic ones are where ET separates best (NewData DRINK
   AUC 0.812)" — reproduces exactly (DRINK 0.804, REST/OUT 0.307/0.225) but does
   not survive its own null. At 6 ET the PD-vs-ET permutation null reaches
   **0.819**, so DRINK is p = 0.050 and FINGER_NOSE p = 0.060 among **seven**
   tasks tested; nothing survives multiplicity. On N-vs-Tremor, where 29–33
   positives make the axis measurable, the **postural OUT task is the best**
   (0.840) and the kinetic tasks are comparable, not superior.

   The pattern is still suggestive — DRINK and FINGER_NOSE at 0.804/0.826 against
   REST/OUT at 0.307/0.225 is a coherent split between task types — which is why
   extracting PADS (28 ET) is still the right next step. It is a test, not an
   exploitation.

   **No PD-vs-ET result from NewData alone can be evidence of anything**: the null
   there reaches 0.819. Apply this check to any other single-cohort NewData claim.

   Blocked here: `physionet.org` is denied by the environment's network policy
   (403 at the egress proxy). Needs a policy change or an uploaded archive.

Continued architecture work is not the route: five rounds moved macro precision
0.583 → 0.675, and the remaining gap to 0.90 is larger than everything that
bought.

## Operational

* `torch.set_num_threads(1)` — thread sync dominates on these tiny tensors (15×
  speedup).
* Full-batch training is the default and beat minibatch at every size.
* The container resets and reverts the working tree to an old commit. **Commit
  and push after every result**, and keep runnable experiments as tracked
  modules, not `scratch/` scripts — `scratch/` has not survived.
