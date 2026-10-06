---
name: tremor-classification
description: Work inside the TremorClassification2 repo — classifying wearable IMU recordings as Normal (N), Parkinson's (PD) or Essential Tremor (ET) across the 2015 / NewData / PADS cohorts. Use whenever the user loads tremor recordings, computes time-frequency features (STFT / multitaper / CWT / HHT / SST / wavelet-packet), works on tremor frequency or peak sharpness, builds or trains the CNN / TCN / two-stream models, merges cohorts, runs patient-level splits, audits preprocessing, proposes a way to raise ET precision, or asks why the model plateaus. Trigger on any mention of quaternion data, angular velocity, spectrograms, tremor stability, PADS, the `experiments` / `signal_processing` / `frequency` / `common` / `metrics` packages, or "improve the model" — the repo has closed most obvious ideas with matched controls and keeps a register of failed predictions, so consult this before proposing anything.
---

# Tremor classification (TremorClassification2)

Three-class N / PD / ET from wearable IMU, three cohorts, **404 merged patients
of whom 49 are ET**. Optimise **per-class precision, ET precision above all** —
never accuracy. The project is at its measured ceiling; the value now is in
knowing precisely what has been closed and why. Read this file, then the
reference file that matches the task:

| task | read |
|---|---|
| any proposal to improve the model | `references/closed_families.md` — what is closed, with the numbers |
| designing an experiment or a control | `references/method_rules.md` — the traps that produced wrong conclusions here |
| the ceiling, preprocessing, transfer, literature | `references/ceiling_and_preprocessing.md` |

**Pruned 2026-09-27:** most reports and closed experiments cited below were
removed from the working tree; read them with
`git show archive/pre-tidy-2026-09-27:reports/<name>.md`. Current scope is
2015 only, one action per model (OUT first), never combined.
`reports/` held ~90 findings; `reports/failed_predictions.md` is the register of
<<<<<<< HEAD
predictions made before the run (37 failed, 29 held); `experiments/INDEX.md` maps
=======
predictions made before the run (49 failed, 40 held); `experiments/INDEX.md` maps
>>>>>>> 0f9f0bdce2fea031faec4e8b03ce5e4a9141563e
every study to the reports that cite it.

## Layout and entry points

```
signal_processing/  transforms (METHODS: welch, stft512, multitaper,
                    wavelet_packet -- 8 benchmark estimators removed, commit
                    57fd8a72), tfd, quaternion, stability (TSI, IF trajectory),
                    tremor_physics
frequency/          characteristics (biomarker table, notebook 01), descriptors
                    (10), tables
common/             loaders per cohort (quaternion_data skips non-unit files),
                    cohorts (merge, logbin), protocol (train loop -- GPU via
                    TREMOR_DEVICE=cuda, ft_head_only; tune_offsets), extract_pads
models/             architectures.py -- every network
experiments/        final_model.py (merged 6-member model, build(); notebook 02)
                    own_data_10et.build (2015 / NewData / PADS feature blocks)
                    transfer_2015.py (2015 OUT model: members, zfit, fit)
                    transfer_2015_explore.py (variants of the transfer model)
                    multisegment.py (hand/lower/upper coupling features) and
                    segments_2015.py (coupling in the 2015 OUT model)
                    _inhouse_transfer_diagnostic.py (PADS -> in-house pre-check)
                    verify_data.py / verify_preprocessing.py (exit code =
                    failures; the one expected failure each is documented)
```

**Pruned twice (2026-09-27/28).** Scripts cited below that no longer exist --
`headline_audit`, `diverse_ensemble`, `pooling_rules.fit_members`,
`estimator_smoothing.load_cohorts / spec_for`, `pcen_hpss`, `inhouse_rest_deep`
and ~80 others -- are in git history: the tag `archive/pre-tidy-2026-09-27`
holds the first set, commit `6ea8ba09` the second. `git checkout <ref> --
<path>` restores one. The merged 9-member figures (precET 0.694 / macroP 0.667)
came from `diverse_ensemble`; `python -m experiments.final_model` still
reproduces the 6-member model. For the current scope (2015, one action) build
new experiments on `own_data_10et.build` and `transfer_2015.fit` / `members`,
and compare against the `scratch` and `ft` arms on the same partitions.

## Non-negotiable invariants

1. **Patient-level splits only.** In the merged tables each row is a patient, so
   `StratifiedShuffleSplit` on the row index is already patient-disjoint.
2. **Split first, then normalise; augmentation and resampling on the train fold only.**
3. **Report per-class precision with the test set's prevalence.** Precision is
   not comparable across differently-composed test sets.
4. **Paired bootstrap CIs for every comparison**, 20 splits minimum. 20 splits
   resolves ~0.04, 40 resolves ~0.025. Three differences of ~0.03 have flipped
   sign on doubling the splits this project, most recently the axis fix (precET
   −0.031 at 20 splits, +0.006 at 40).
5. **Print the split-level win rate beside every paired mean.** A positive mean
   with a sub-0.5 win rate is a few favourable folds, not a method that helps —
   logit adjustment read precET +0.034 with a win rate of 0.42
   (`logit_adjustment.md`).
6. **One-split smoke tests are not evidence.** Four this session inverted or
   evaporated at 20 splits, including a Spearman of +1.000 that became an
   inverted U. Use them to catch crashes, not to read direction.
7. **Every comparison needs a matched control that isolates one thing.** The
   control decides attribution; the plain baseline decides adoption. They are
   different questions — see `method_rules.md` for the time they were confused.
8. **A permutation null for any single-model claim.** In-house PD-vs-ET null
   spans [0.298, 0.655] at 21 ET; nothing below AUC 0.66 there is
   distinguishable from chance.
9. **Record the prediction in the docstring before launching**, then append the
   outcome to `reports/failed_predictions.md`. Measurement-derived predictions
   held twenty-one of twenty-one until the in-house REST and 2015 transfer work;
   several have failed since (#29-#36), mostly extrapolations beyond the measured quantity -- a binary
   ranking AUC read as 3-class precision (#31, #32), or an endpoint (head-only)
   read as the limit of a family (#34, #35). Mechanism stories have failed 28
   times. Measure the quantity you predict, at the setting you predict.
10. **Run the cheap diagnostic before the fits** when a proposal changes the
    representation. Two statistics of the 16-bin spectrum, taken in minutes,
    called PCEN's failure that a reasoned prediction got backwards
    (`_pcen_alpha_diagnostic.py`); a 6-feature AUC against a permutation null
    closed Euclidean Alignment without fitting anything
    (`_euclidean_alignment_diagnostic.py`). It can rule a method out; it cannot
    promise a gain — see the asymmetry in `method_rules.md`.
11. **Put a shuffled control on every appended feature block**, permuted within
    cohort and redrawn per split. Six tangent-space columns gave +0.003 macroP
    and +0.016 precPD; six *permuted* columns of the same block gave +0.003 and
    beat it on precET (`riemann_axes.md`). Appending columns moves numbers by
    dimensionality alone — six random ones cut ET predictions by −1.55/split
    [−3.35, −0.15] \*.
12. **Ask whether a feature is redundant, not only whether it is informative.**
    The pre-run diagnostic correctly showed the tremor's orientation separates
    PD from ET (AUC 0.702 / 0.713 vs permutation nulls) and the method still
    failed, because on that contrast it duplicated the ten descriptors the model
    already had (PADS union 0.797 vs 0.795). *Existing* is not *being new* — the
    cheap union-vs-best-member test is what distinguishes them.
13. **An adaptive normaliser is safe only when the unit it normalises over
    contains every class.** Three methods have died on this: PCEN (a band
    divided by its own time-average — erases *which band* has energy),
    patient-level Euclidean Alignment (a patient divided by their own
    covariance — erases *which direction* their tremor points), and per-cohort
    priors. A patient here carries exactly one label, which is what breaks
    every per-subject normaliser imported from BCI, where subjects supply all
    classes. Check this before importing; the check is one diagnostic.

## Cohorts and merging (settled)

| cohort | patients | N / PD / ET | recording | frames averaged |
|---|---|---|---|---|
| 2015 | 151 | 61 / 75 / 15 | 10.5–30.3 s, median 15.5 | 21 |
| NewData | 56 | 27 / 23 / 6 | 10 s epoch selected by in-band fraction | 12 |
| PADS | 383 (capped 90/class) | 79 / 276 / 28 | 10.24 s fixed, both wrists | 13 |

All three are gyroscope angular velocity at 100 Hz; 2015/NewData use the
`lower_arm` sensor as wrist-equivalent. **PADS labels are exact-matched from the
manifest** — `raw_label` takes exactly three values, so no atypical-parkinsonism
contamination remains (an earlier version of this file claimed otherwise; it was
wrong). NewData at 6 ET is a training cohort, never an evaluation one.
**In-house PD vs ET at OUT is not measurable from spectral features, and not
only because of sample size** (`inhouse_pd_vs_et.md`): at OUT, in-house ET match
in-house PD on tremor amplitude, peak sharpness and frequency, while PADS ET
have about 2x the amplitude and twice the sharpness. Pooling, transfer from PADS
and switching sensors all stay inside the chance range. The signal chain was
audited against independent SciPy code and is correct; the notebook's 0.31 was
one unlucky CV partition (0.32-0.52 across CV seeds) -- **report small-cohort CV
AUCs averaged over repeated partitions** (`classify(n_repeats=10)`). **At REST,
2015 does separate PD from ET** (AUC ~0.65-0.70, above chance on all three
sensors, 16 ET) with PD slower, the textbook direction; **PADS ET are slower
than PD** at both tasks, so the two sources teach opposite frequency rules.
**2015 REST as its own model is closed for ET** (`rest_2015.md`, 40 repeats):
precET 0.124 at prevalence 0.106 from scratch, and PADS Relaxed transfer
reverses ET-vs-PD ranking (AUC 0.484, −0.105 \*, below its permuted-label
control) -- PADS's opposite rest-frequency rule, inside the deep model. The
2015 REST PD-vs-ET linear signal is borderline (0.61-0.65, p 0.03-0.12).
Earlier: **In-house REST in the deep model is now tested and closed for ET**
(`inhouse_rest_deep.md`, now in git history): REST alone, OUT+REST score fusion and the hand
sensor all leave in-house precET at ~0.16 (prevalence 0.105); the only
baseline-significant gain is precPD +0.059 \* (hand, OUT+REST, 20 repeats).
**2015 OUT transfer learning works** (`transfer_2015.md`, 40 repeats): PADS
pretrain → 2015 fine-tune lifts precET 0.249 → 0.333 (+0.084 \*), macroP
+0.023 \*, precPD −0.040 \*; pooling PADS does not, and N-vs-tremor-only
pretraining collapses ET. `cohort_strategies`' "worst thing tried" was scored
on PADS test patients and does not apply here.
Explored (`transfer_2015_explore`, 40 repeats): the recipe is a local optimum --
longer fine-tune, head-only fine-tune, uncapped PADS, NewData in pretraining
and ensembling with scratch are null or worse; 6 seeds is +0.021 precET n.s.
The gain is in ranking (PD-vs-ET AUC +0.055 \*, win 0.93).
Rounds 2-3 (L2-SP, pretrain length, per-recording pretrain): L2-SP λ 1
raises ranking on fresh partitions (AUC +0.012 \*) but not precision
(precET −0.007, precPD −0.017 \*) -- its selection-partition precET gain
was selection bias. Plain `ft` stays adopted.
<<<<<<< HEAD
**Online pretrained model: MOMENT-1-small transfers** (`pretrained_moment.md`):
frozen embeddings beat the spectrum features (PD-vs-ET AUC 0.701 vs 0.621) and
weight-permuted MOMENT by 0.08-0.17; ft + MOMENT raises PD-vs-ET AUC +0.063 \*
and precN / precPD on fresh partitions but NOT precET (−0.058 \*), macroP flat.
Runs in a separate conda env `moment` (momentfm pins numpy 1.25). ImageNet ViT
stays closed; time-series pretraining is the family that transfers.
Fine-tuning MOMENT (LP-FT) is null vs frozen (AUC −0.010); keep it frozen.
**Trap:** `momentfm` freezes encoder + embedder by default -- pass
`freeze_encoder=False, freeze_embedder=False` or fine-tuning trains only the head.
Pooled over 100 partitions, ft + frozen MOMENT: AUC +0.064 \*, macroP +0.012 \*,
precET −0.023 (n.s.; swings −0.058 to +0.020 by partition set).
=======
**Multi-segment coupling** (`multisegment_2015.md`, 40 repeats): hand/lower/upper
coherence, |cos phase| and power gradients. On 2015-only models a real,
attributable gain (macroP +0.016 \*, macroF1 +0.013 \* over scratch, same vs a
shuffled-row control; precET +0.036 borderline); on top of `ft` nothing
significant on precision (precET +0.018), AUC +0.015 \* within what extra
columns give. Not adopted on `ft`. Feature level: hand-forearm coherence lower
in ET on 2015 OUT (p = 0.045), not on NewData.
**What limits `ft`** (`stack_2015.md`): 14 of its 27 consistent errors are patients with
no measurable tremor in OUT, REST or WING (PD->N 11, ET->N 3) -- unreachable by
tremor features. ET is found when its tremor is sharp (peak sharpness 12 vs 4).
A second stage on `ft`'s probabilities + coupling is real vs a shuffled control
but nets ~0 vs `ft` (refitting the decision on 151 patients costs macroP
−0.035 \*); REST amplitude as late fusion gives precN / precPD +0.02 \*.
**OUT + REST late fusion** (`fusion_2015.md`, 40 repeats, one action per model):
precPD +0.036 \*, precN +0.019 \*, AUC +0.037 \* but precET −0.062 \* and top-3 ET
precision 0.533 -> 0.358 \*; macro unchanged. A class trade: keep `ft` for ET.
**40 % of 2015 PD and ET show no OUT tremor above the control range**
(`tremor_present_2015.md`); `ft` precPD 0.826 where tremor is present.
**80 % precision** (`high_precision_2015.md`, nested thresholds): N 0.849 and PD
0.873 held-out with the REST-weight-0.5 fused model asking for 0.90 (weight 0.25:
0.846 / 0.872) (~40 % recall, rest referred);
tremor vs none 0.894 at 0.78 recall. **ET cannot reach 0.80** -- nested 0.1-0.3 on
<1 flag per repeat; any non-nested ET >= 0.8 claim is fitting noise.
**PD vs ET** (`pdet_2015.md`, reproduced on fresh partitions 100-139 of the same
patients; not patient-bootstrapped): OUT+REST fusion -> second stage + 8
coupling features, within-fold AUC 0.633 -> 0.709 (+0.076 \*); its top-1 ET pick is worse -- use `ft`/`ft_seg` to flag a few.
Score second stages WITHIN folds: pooled out-of-fold scores from differently
calibrated fold models cost even a monotone refit 0.023 AUC. Re-assigning the
base's ET labels by that ranking (fixed count) LOWERS precET (−0.03): AUC gain is
mid-ranking, not in the top calls.
**Default 2015 model: `ft` + 0.25 x REST scratch** (`fusion_2015.md`): vs the
extra-seeds control precN +0.016 \*, AUC +0.023-0.029 \* on both partition sets,
ET unchanged; vs plain `ft` mostly n.s.; none significant at patient level.
**+ 0.25 WING** (`fusion3_2015.md`; holds under patient resampling): precPD
+0.032 \* / +0.027 \*, ET unchanged -- use it for the standard 3-class decision;
for the high-confidence >= 80 % rule keep OUT + 0.25 REST (WING lowers precN there).
**Patient-level bootstrap** (`patient_bootstrap_2015.md`): only transfer's precET
(+0.092 [+0.012, +0.211]), WING's precPD (+0.03) and NewData's AUC (+0.044) survive
patient resampling; REST fusion / coupling / transfer AUC do not. **2015 OUT is 150
patients** after the PD 12 id fix in `quaternion_data` (saved runs used 151 rows).
**On the corrected 150-patient table** (`verify_transfer_150.md`): `ft` − scratch precET
+0.079 \* (repeat level) but patient-level [−0.028, +0.202] -- consistent, not established
for new patients. Early pretraining stops and in-fold epoch choices are closed
(`pretrain_stop_150.md`, `infold_epoch_150.md`).
**Best 2015 system for precision (corrected 150-patient table, `combined_150.md`):
`ft` + 0.25 x REST + 0.25 x WING** -- precN 0.750 / precPD 0.777 / precET 0.342 /
macroP 0.623; precPD +0.067 vs `ft` holds at patient level; high-confidence PD 0.87
at recall 0.37 (ft: 0.21). NewData pooling inside it costs ET precision (−0.060 \*,
patient level) while raising AUC -- use it for ranking only.
Closed this round (150 patients): ranking-aware ET losses (AUC / pAUC hurt),
Co-Tuning (hurts), Batch Spectral Shrinkage, supervised contrastive pretraining.
**Fine-tuning methods closed** (`training_methods_2015.md`, 40 repeats, `ft` arm
bit-exact): LP-FT +0.005 AUC only; WiSE-FT harmful (top-5 ET 0.50 -> 0.23 \*: the
PADS boundary is wrong for 2015); SWA worse than the best-val checkpoint (precET
−0.049 \*); SAM, label smoothing null. The `ft` recipe is at its optimum.
Checkpoint prediction ensembling and gradient-disparity stopping also closed
(`ckpt_ensemble_150.md`).
**NewData pooled into 2015 fine-tuning** (`newdata_2015.md`): PD-vs-ET AUC +0.044 \*
(+0.037 \* over shuffled labels), precision flat. Ranking gains stack: NewData +
REST + coupling -> within-fold AUC 0.719 vs `ft` 0.625 (selection). **Fresh
partitions: NewData pooling +0.043 \* (also at patient level), no significant
precision cost; rankers `fuse_nd` (NewData-pooled `ft` + REST, 0.5/0.5) 0.710 ~=
fuse + coupling 0.709; `fuse_nd` + coupling 0.718 (+0.008 \*).**
**Report verification** (2026-10-05): 58 claims re-computed from saved runs; 40
matched, 10 mismatches and 7 overstatements were corrected in the reports.
>>>>>>> 0f9f0bdce2fea031faec4e8b03ce5e4a9141563e
`TREMOR_DEVICE=cuda` trains on the GPU (1.6-2.4x faster; CPU stays the
default so existing results reproduce bit-for-bit).
In-house REST as a separate descriptor block in the 9-member model (`inhouse_rest.md`; its scripts are in git history at 969633e0): inAUC +0.022 \* but only +0.012 (null) over a shuffled control, in-house ET recall −0.062 \*; not adopted. **The 2015 REST signal does not replicate** (`rest_replication.md`): pooled with NewData 0.573, inside its null; frozen 2015 rule on PADS Relaxed 0.280, reversed at p < 0.001. Claim cohort-dependence, not in-house separability. `N 2` REST/WING files hold
accelerometer data (|q| ~ 9.8; `verify_data` check 13 fails on purpose).

Merge: cap PADS at 90/class, pool, one global set of validation-tuned priors,
postural task only. Dropping PADS is catastrophic (precET 0.519 → 0.065);
per-cohort priors, sample weights, PADS pretrain/finetune and distribution
alignment are all significantly worse. Cohort-ID as an input buys precN +0.024
over a valid control and nothing on ET.

## The reported model and its headline (corrected axis, 40 splits)

Two-stream: `Spectrum1DCNN` on the 16-log-bin **multitaper** spectrum (nw 2.5,
K 4, nperseg 256, 3–15 Hz) + `TrajectoryEncoder` on the IF trajectory, soft-voted
with `ResidualTCN`; 3 seeds each; per-class logit offsets tuned on validation.

| | precN | precPD | precET | macroP |
|---|---|---|---|---|
| welch baseline | 0.640 | 0.635 | 0.550 | 0.608 |
| 6-member (was reported) | 0.648 | 0.654 | 0.654 | 0.652 |
| **9-member, adopted** | 0.650 | 0.658 | **0.694** | **0.667** |

Paired **+0.044 [+0.020, +0.068] macroP**, **+0.104 [+0.041, +0.169] precET**,
winning 72 % of splits. Transform alone +0.078 [+0.022, +0.132] precET \*;
**the trajectory stream is no longer significant** (+0.026 [−0.009, +0.068])
once its transient end points were removed — treat it as plausible, not
verified. **Quote the 9-member precET 0.694 / macroP 0.667** (`diverse_ensemble`,
40 splits: precET +0.040 \*, macroP +0.016 \*, macroF1 +0.016 \* vs the 6-member
model; against a matched seed control only macroF1 +0.017 \* survives, precET
+0.032 has lower bound 0 and win 0.45 — so macroF1 is the established part).
The 6-member 0.654 / 0.652 is on the fixed axis,
fixed Q-factor and guarded trajectory; anything quoting 0.663 / 0.669 / 0.685
predates one of those fixes.

Every component of that recipe has been swept, and each sits at a setting
nothing beats — but **"interior optimum" was too strong for the estimator** and
is withdrawn. That sweep ran on a duplicated copy of the frequency-axis bug;
re-measured at 40 splits it is a **plateau** (welch / nw2.5 / nw4 / nw6 span
0.007 against a 0.025 resolution), with only the sharpest arm, ar16,
distinguishable. Keep nw 2.5 as the incumbent, not as a peak
(`estimator_smoothing.md`). The swept list:

## The ceiling, in four numbers

The six ensemble members agree on **59.5 %** of patients and are **68.8 %**
correct there; on the contested **40.5 %** they score **0.443 balanced accuracy
against a 0.465 constant baseline**, with a top-2 margin four times narrower.
Members are genuinely diverse (r = 0.859, 20.5 % disagreement), so this is a
boundary, not a redundant ensemble. It predicts, correctly, that every method
which only reshuffles the contested set will tie — and ten have.

Contested rate is cohort-dependent with class mix controlled (2015 0.307, PADS
0.432, NewData 0.573) and rises monotonically as tremor frequency falls
(0.515 / 0.416 / 0.253 by tercile) — but the frequency effect is class confusion
through a monotone class ordering, not a physical mechanism; the claim that it
was non-circular is **retracted**.

Clinical PD-vs-ET diagnostic accuracy is 74–80 % and ET has no gold standard,
even post-mortem. **That test has now been run** (`self_consistency_gate.md`):
on patients the model gets wrong it still gives both of their recordings the
same answer **73 %** of the time against a **55 %** same-class control, but that
sits **0.149 below** its rate on correctly classified patients. Scaled between
the guessing floor and the working ceiling, misclassified patients are **~54 %
of the way to fully self-consistent**, and a same-arm retest and a two-limb
comparison agree to within 3 points.

**Revised by `readjudication_list.md`:** a model-free amplitude test on the
actual consistently-wrong patients finds **43 of 62 are N↔PD phenotype** (PD
with no tremor, postural *or* rest; controls with PD-level tremor), leaving only
**~19 patients (~5 %)** as label-error candidates. The ceiling is mostly
phenotype–task mismatch. The original reading follows for the record.
**So both accounts hold, in almost equal measure.** Crucially, consistently
wrong = mislabelled ∪ genuinely atypical, so **~55 % is an upper bound on the
label-noise share of errors, not an estimate** — never quote it as "half the
labels are wrong". The operational payoff is that it names a **targeted
re-adjudication list** (recordings agreeing with each other, disagreeing with
the label; ~50–60 patients cohort-wide), and reviewing that list is what would
turn the bound into a measurement.

## Preprocessing: what is verified, what was fixed

* **Frequency axis bug, fixed.** `m_multitaper` / `m_sst` rebuilt their axis as
  `linspace(0, 15, n)` after cropping true rfft bins — a 1.05 % stretch, 0.156 Hz
  at the top, 14 % of the N-vs-ET gap, and the reported model's only input on a
  different scale from its own descriptors. Performance effect null; headline
  re-derived. `_kept_rfftfreq` now asserts axis length equals spectrum length.
  **Relative safeguards cannot see a defect every arm shares** — only an absolute
  check against ground truth does.
* **Two more defects, found by `verify_preprocessing.py`, fixed, null.**
  `describe()`'s Q-factor spanned every supra-half-max bin instead of the peak
  (a tone with a 0.8-amplitude harmonic read Q 0.94, not 15); under the correct
  definition PADS shows **no ET-vs-PD Q gap** (ET 21.0 / PD 22.1 / N 22.8, ratio
  1.90 → 0.95), so that descriptor's class contrast was definitional — the
  headline `peak_sharp` is a different, sound quantity and stands. And the IF
  trajectory's points 0 and 63 were band-pass transients (2.7 Hz of noise on a
  0.5 Hz signal); a 0.25 s guard removes them. Paired against the reconstructed
  pre-fix model: Q fix +0.012 [−0.003, +0.034] macroP, guard −0.004 [−0.027,
  +0.021], both together −0.006 [−0.026, +0.019]. Headline intact; the
  trajectory stream's own significance did not survive (see above).
* **Verified fine:** unit/modality consistency across cohorts; 3 Hz low edge
  (sub-3 Hz carries nothing usable); DC in the multitaper path (5×10⁻⁵ effect);
  noise-dominated recordings (≤ 8 % of any class); wrist averaging (aligning
  peaks before the mean is null; the model sits below the misalignment knee —
  doubling jitter costs −0.071. The "33 % of the sharpness gap" that motivated
  it was measured with the pre-fix Q-factor and is withdrawn).
* **The time-average is doing more than averaging.** Explicit
  harmonic–percussive separation confirms the physics — harmonic 0.660 >
  dense-hop control 0.639 > percussive 0.523 precET, the percussive arm
  significantly worse than its own control (−0.117 \*), so **class information
  sits in the sustained component** — yet adopting HPSS is null (+0.018 n.s.).
  `P.mean(0)` already divides a transient by the frame count while a sustained
  oscillation contributes to every frame, so it is a weak separator already.
  Same shape as the PADS onset: a real, class-ordered artifact that changed
  nothing after averaging. **PCEN is the opposite and must not be used**
  (precET −0.233 \*, macroP −0.101 \*): dividing each band by a smoothed copy of
  itself destroys *which band* has energy, which is the entire signal here.
  A dense 0.16 s hop, needed for anything on the time axis, is free.
* **Known and open:** cohort-dependent frame averaging (21 / 13 / 12 frames);
  NewData resamples quaternions *before* the sign-continuity fix (latent — no
  flips exist in the raw streams); **PADS carries an untrimmed arm-raising onset
  that is class-ordered** (first-1.5 s in-band RMS ratio N 1.39, PD 1.33, ET
  1.06; absent elsewhere). Trimming removes it (→ 1.10 / 1.04 / 0.96) but
  changes nothing: headline macroP −0.006 [−0.031, +0.022], and PADS→in-house
  transfer sits below the chance floor with or without it. Leave
  `--trim-start` at 0; do not trim the *end* (precN −0.037 \*).

## Before proposing an improvement

Check `references/closed_families.md`. Closed with matched controls: ensemble
pooling rule, ensemble size, balanced bagging, one-vs-rest (harmful, −0.162
precET), gating on disagreement, three subject-pruning criteria (all null — the
hard-drop harm is **retracted**, see `prune_training.md`), band edge,
estimator sharpness, peak-aligned averaging, cohort-ID input, feature unions,
learned pooling over recordings, SSL, time-domain networks, mixup, prior
objective, MiniRocket/ROCKET, logit adjustment, PCEN, HPSS, cropped training,
Euclidean Alignment, Riemannian tangent space, and everything in the older
table. **Descriptor-level gains do not compose to the model** — three
separate instances now, including a 33 % sharpness recovery that produced
+0.007 precET. **Descriptor-level damage does compose**, which is why the
label-free diagnostic is worth running first.

What remains genuinely open: feature-level cohort harmonisation (ComBat-style,
fitted on train only), physiology-preserving ET augmentation with a
label-preservation audit, severity-stratified reporting, and re-running `STAB`
(TSI) at 40 splits — it was null at 20 but halved precET variance. **The data
side is now the live work** — see `reports/data_plan.md`, whose §1 target is
**in-house ET (21 today)**, not merged ET, because the merged precET is
substantially PADS predicting PADS (in-house it is 0.193).

## Operational

* `torch.set_num_threads(1)`; full-batch training; 6 fits per split ≈ the unit
  of cost.
* The container resets and kills background runs. Write logs to the scratchpad
  outside the repo, commit and push after every result, and arm a monitor that
  detects a *dead* process, not just a crash line.
* `pkill -f <pattern>` matches its own shell. `A=… && nohup B & nohup C` puts the
  assignment in a subshell — `C` sees `$A` empty.
