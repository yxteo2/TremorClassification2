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

`reports/` holds ~70 findings; `reports/failed_predictions.md` is the register of
predictions made before the run (20 failed, 10 held); `experiments/INDEX.md` maps
every study to the reports that cite it.

## Layout and entry points

```
signal_processing/  transforms (12 estimators, METHODS dict), tfd, quaternion,
                    stability (TSI, IF trajectory), tremor_physics, reemergence
frequency/          characteristics, descriptors (10), tables
common/             loaders per cohort, cohorts (merge, logbin), protocol (splits,
                    train loop, tune_offsets), extract_pads
models/             architectures.py — every network
experiments/        final_model.py (reported model, build()), headline_audit.py
                    (40-split re-check), verify_preprocessing.py (every stage
                    vs synthetic ground truth; exit code = failures),
                    pooling_rules.fit_members (reusable 6-member trainer),
                    estimator_smoothing.load_cohorts / spec_for (rebuild
                    spectra from raw recordings)
```

`python -m experiments.final_model` reproduces the reported model. New
experiments should reuse `fit_members` and `load_cohorts` + `spec_for`, and
assert bit-exactness against `build()["SPEC"]["multitaper"]` for the unchanged
arm — several results here were only valid because that assert passed.

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
   have held here (eleven of eleven); mechanism stories have failed 21 times.
10. **Run the cheap diagnostic before the fits** when a proposal changes the
    representation. Two statistics of the 16-bin spectrum, taken in minutes,
    called PCEN's failure that a reasoned prediction got backwards
    (`_pcen_alpha_diagnostic.py`); a 6-feature AUC against a permutation null
    closed Euclidean Alignment without fitting anything
    (`_euclidean_alignment_diagnostic.py`). It can rule a method out; it cannot
    promise a gain — see the asymmetry in `method_rules.md`.
11. **An adaptive normaliser is safe only when the unit it normalises over
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
| **reported** | 0.648 | 0.654 | **0.654** | **0.652** |

Paired **+0.044 [+0.020, +0.068] macroP**, **+0.104 [+0.041, +0.169] precET**,
winning 72 % of splits. Transform alone +0.078 [+0.022, +0.132] precET \*;
**the trajectory stream is no longer significant** (+0.026 [−0.009, +0.068])
once its transient end points were removed — treat it as plausible, not
verified. **Quote precET 0.654 / macroP 0.652.** These are on the fixed axis,
fixed Q-factor and guarded trajectory; anything quoting 0.663 / 0.669 / 0.685
predates one of those fixes.

Every component of that recipe has now been swept and sits at an interior
optimum: the 3 Hz low edge, nw 2.5 (a 23× sharpness sweep from ar16 to nw 6
peaks exactly there), 16 bins, 3 seeds, the arithmetic mean over members, the
plain mean over a patient's recordings, and `tune_offsets` on macro F1. None
of these is an untested default any more.

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
even post-mortem. The measured ceiling coincides with that. It is a hypothesis
until tested: the test is whether the model agrees with *itself* across a
patient's own recordings far more than with the label. Untried.

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
precET), gating on disagreement, three subject-pruning criteria, band edge,
estimator sharpness, peak-aligned averaging, cohort-ID input, feature unions,
learned pooling over recordings, SSL, time-domain networks, mixup, prior
objective, MiniRocket/ROCKET, logit adjustment, PCEN, HPSS, and everything in
the older table. **Descriptor-level gains do not compose to the model** — three
separate instances now, including a 33 % sharpness recovery that produced
+0.007 precET. **Descriptor-level damage does compose**, which is why the
label-free diagnostic is worth running first.

What remains genuinely open: feature-level cohort harmonisation (ComBat-style,
fitted on train only), physiology-preserving ET augmentation with a
label-preservation audit, severity-stratified reporting, the within-patient
agreement test of the label-noise account, and re-running `STAB` (TSI) at 40
splits — it was null at 20 but halved precET variance.

## Operational

* `torch.set_num_threads(1)`; full-batch training; 6 fits per split ≈ the unit
  of cost.
* The container resets and kills background runs. Write logs to the scratchpad
  outside the repo, commit and push after every result, and arm a monitor that
  detects a *dead* process, not just a crash line.
* `pkill -f <pattern>` matches its own shell. `A=… && nohup B & nohup C` puts the
  assignment in a subshell — `C` sees `$A` empty.
