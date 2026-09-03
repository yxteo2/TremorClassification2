# Tremor classification — N / PD / ET from wearable IMU

Classifying Normal, Parkinson's and Essential Tremor from wrist/arm inertial
recordings, across three cohorts.

Two lines of work:

1. **Time-frequency processing** — characterise the tremor and classify from
   frequency and oscillation-shape quantities.
2. **Deep learning** — classify each patient's time-frequency signal.

## Start here

| notebook | what it does |
|---|---|
| `01_tremor_characteristics.ipynb` | frequency characteristics per class; classification from mean/max frequency, features added one at a time |
| `02_deep_model.ipynb` | the final two-stream deep model and each component's contribution |

```bash
python -m frequency.characteristics     # characteristics + frequency classification
python -m experiments.final_model       # the merged deep model, paired vs baseline
python -m experiments.verify_preprocessing  # every stage vs synthetic ground truth (exit = failures)
python -m experiments.own_data_10et     # in-house patients, 10 ET per test set
python -m experiments.inhouse_axes      # rotation-invariant axis features in-house
```

## Headline results

**Read the in-house and merged numbers separately.** They differ sharply, and
the merged figures do not describe in-house patients
(`reports/own_data_reality_check.md`).

### In-house (2015 + NewData, 10 ET per test set, prevalence 0.101)

| model | precN | precPD | precET | macro P |
|---|---|---|---|---|
| base | 0.652 | **0.769** | 0.193 | 0.538 |
| + rotation-invariant axis features | 0.681 | 0.729 | 0.245 | 0.552 |
| + PADS in training | 0.685 | 0.687 | 0.196 | 0.523 |

> **Read these against the detection floor.** At 21 in-house ET patients the
> permutation null for PD-vs-ET AUC reaches 0.655, and no in-house feature family
> clears it (`reports/permutation_null.md`). Differences between these rows are
> paired and can be real; the individual rows are not evidence that any family
> separates PD from ET in-house.

**PD precision 0.769 is the strongest in-house figure**, and adding PADS
significantly *degrades* it (−0.082). ET precision 0.193–0.245 is a ~2× lift over
prevalence; the axis gain is not significant at 21 ET patients.

### Merged cohort (2015 + NewData + PADS, n=404)

| model | precN | precPD | precET | macro P |
|---|---|---|---|---|
| welch baseline | 0.640 | 0.635 | 0.550 | 0.608 |
| **multitaper + IF trajectory** | 0.648 | 0.654 | **0.654** | **0.652** |

Paired **+0.044 [+0.020, +0.068] macro precision** and **+0.104 [+0.041, +0.169]
ET precision** over the welch baseline, at **40 splits**, winning on 72 % of them
(`reports/headline_audit.md`).

The decomposition changed once the descriptor and trajectory defects were fixed
(`reports/descriptor_trajectory_fix.md`): the transform alone is now precET
+0.078 [+0.022, +0.132] * / macroP +0.031 [+0.011, +0.053] *, while the
**trajectory stream is no longer significant** — precET +0.026 [−0.009, +0.068],
macroP +0.012 [−0.000, +0.027]. The two still sum to the whole. Do not describe
the trajectory as a verified component; it is a plausible one whose earlier
significance did not survive removing its transient end points.

> **Re-derived twice**: on the corrected multitaper frequency axis
> (`reports/axis_fix_audit.md`), and again after the Q-factor and IF-trajectory
> fixes (`reports/descriptor_trajectory_fix.md`), which moved the baseline too
> because both models consume the descriptors. On the first re-derivation the old
> axis was stretched 1.05 %; fixing it
> left the headline essentially unchanged and marginally stronger (macroP +0.043
> → +0.046, precET +0.097 → +0.103), and the welch baseline row is bit-identical
> because welch was never affected. A 20-split A/B of the fix alone read precET
> −0.031 [−0.087, +0.015]; at 40 splits the fixed model is +0.006 above the old
> one — the 20-split reading was noise, exactly as its interval said.
>
> Quoted at 40 rather than the original 20 splits. The claim was re-audited
> because 20 splits resolves only ~0.04 and a paired +0.021 had already evaporated
> on doubling. It held and its interval **tightened** (from [+0.014, +0.067]).
> After all three fixes the reported model reads precET **0.654** / macroP
> **0.652** at 40 splits; **those are the figures to quote**. sd(precET) is 0.19,
> so every figure between 0.65 and 0.69 quoted earlier in this project is the
> same number under noise.

### N vs Tremor — the one place >0.90 is reached

Six frequency characteristics and a logistic regression: **precision 0.910
(2015) / 0.924 (PADS)**.

### Tremor characteristics (PADS, 383 patients)

| | N | PD | ET |
|---|---|---|---|
| max frequency (Hz) | 7.20 | 7.07 | 6.16 |
| bandwidth (Hz) | 2.94 | 2.48 | 2.04 |
| peak sharpness | 4.08 | 5.80 | **12.19** |
| linearity (in-house) | — | higher | lower |

ET is the sharper, more tonal peak; PD's oscillation is more confined to a
single axis.

## Data

| cohort | patients | N / PD / ET | source |
|---|---|---|---|
| 2015 | 151 | 61 / 75 / 15 | `Data/` — quaternion, 3 sensors, OUT/REST/WING |
| NewData | 56 | 27 / 23 / 6 | `NewData/` — 2025 Moveo, both limbs, 7 tasks |
| PADS | 383 | 79 / 276 / 28 | `pads_stretchhold/`, `pads_relaxed/` — both wrists |

PADS is extracted with `python -m common.extract_pads`. Labels are re-derived
from the manifest by **exact** diagnosis match — a substring match once put 13
non-ET records in the ET class (`reports/pads_label_bug.md`).

## Package layout

```
models/              architectures.py — every model: CNN / TCN / BiLSTM /
                     two-stream / transformer / cross-attention

signal_processing/   transforms.py       12 TF methods, all power-scaled
                     tfd.py              multitaper / SST / CWT
                     quaternion.py       quaternion -> angular velocity
                     preprocessing.py    band-pass, framing, STFT
                     spectral.py         log compression, SpecAugment
                     stability.py        Tremor Stability Index, IF trajectories
                     tremor_physics.py   harmonics, rotation-invariant axes,
                                         modulation spectrum, amplitude
                     reemergence.py      envelope timing from recording start

frequency/           characteristics.py  6 characteristics + classification
                     descriptors.py      10 spectral descriptors
                     tables.py           per-patient spectra, asymmetry features
                     report.py           frequency comparison across cohorts

common/              data.py             the Recording type
                     quaternion_data.py  2015 loader
                     load_2025.py        NewData loader
                     extract_pads.py     PADS extraction (exact-match labels)
                     loaders.py          PADS loader
                     cohorts.py          merged assembly, capping, missing modality
                     training.py         training loops
                     protocol.py         splits + validation-tuned priors
                     cache.py            on-disk caching

metrics/             stats.py            subject-clustered bootstrap CIs
                     selective.py        precision at reduced coverage
                     benchmark.py        method ranking, BH + Bonferroni
                     merged.py           balanced accuracy, cohort probe

experiments/         70 runnable studies. The ones that carry a result:

                     final_model.py            the reported merged model
                     headline_audit.py         that model re-checked at 40 splits
                     own_data_10et.py          in-house, 10 ET per test set
                     pd_vs_et.py               the binary axis, per cohort
                     permutation_null.py *     detection floors  (see reports/)
                     catch22_family.py         waveform features vs spectral ones
                     tf_variability_screen.py  window-length sweep
                     tf_window_paired.py       short-window spectrum, paired
                     oneclass_paired.py        one-class PD + logreg hybrid
                     binning.py                band coverage vs estimator
                     loco_pd_et.py             cross-cohort transfer
                     kinetic_task_audit.py     auditing lever #3
                     ensemble_diversity.py     the ceiling's shape: 60/40 split
                     contested_specialists.py  is the contested 40 % readable?

                     The rest are recorded negatives — SSL, attention, MIL,
                     time-domain TCNs, fusion points, rest/postural contrasts.
                     None improved the reported model. Most have a report under
                     a different name; a handful were never written up.
                     experiments/INDEX.md maps every study to the reports that
                     cite it, and names the unreported ones honestly.

reports/             74 findings, including every retraction and a register
                     of predictions made before the run (failed_predictions.md)
```

The ViT checkpoint is stored split; rebuild with `cat vit_chunk_0* > vit_fp16.pt`.

## Conventions that matter

* **Patient-level splits only.** Never split one patient's recordings.
* **Every safeguard above is relative — keep one absolute check.** Patient-level
  splits, paired bootstraps and permutation nulls all compare arms, so a defect
  shared by every arm is invisible to all of them. Three survived that way: a
  1 % frequency-axis stretch, a Q-factor that spanned every supra-half-max bin
  instead of the peak, and IF-trajectory end points that were filter transients.
  `experiments/verify_preprocessing.py` pushes synthetic signals with known
  answers through every stage; run it after touching `signal_processing/`,
  `frequency/` or `common/cohorts.py` (`reports/descriptor_trajectory_fix.md`).
* **A rank correlation is the wrong instrument for a non-monotone effect.**
  Estimator smoothing has an interior optimum; Spearman read −0.600 and a canned
  line called it "smoother is better" when the table showed a peak at the current
  setting (). Read the table.
* **Report per-class precision with the test set's prevalence.** Precision is
  not comparable across differently-composed test sets — a cap sweep once
  produced a clean monotone trend that was entirely a prevalence artifact.
* **Paired bootstrap CIs for every comparison.** Unpaired differences of ~0.04
  sit inside the per-config sd here and have repeatedly evaporated when paired.
* **Two protocols, two questions.** Mixed-cohort answers "how well at sites we
  trained on"; leave-one-cohort-out answers "will this transfer". Only LOCO
  supports a generalisation claim.
* **Prefer replacing a feature family over appending one.** Eight feature unions
  have underperformed their best member. The two that work — `axes + stability`
  and `logreg + one-class` — combine things that differ *in kind*, and both do it
  at the **score** level, not the feature level (`reports/oneclass_hybrid.md`).
* **A frozen treatment needs a frozen control.** Changing two things at once cost
  this project a headline result (`reports/ssl_retraction.md`).
* **Never use a fixed random draw as a null control — re-draw it per split.**
  A frozen draw's chance association with the label is *constant* across splits,
  so the paired bootstrap cannot see it; pairing protects against split-to-split
  noise, not against a quietly informative fixed feature. One 5th-percentile
  draw manufactured precET +0.090 [+0.019, +0.168] from three meaningless
  columns (`reports/cohort_id_input.md`).
* **Never judge a second model on a subset the first model's uncertainty
  defined.** Conditioning on where model A is unsure selects patients A does
  badly on and applies no such selection to model B, so B looks good by
  construction. Only comparisons to a selection-independent baseline (chance)
  are valid there. Stated as a caveat in `reports/contested_specialists.md` and
  confirmed by `reports/contested_gating.md`, which found the apparent advantage
  was not complementary signal at all.
* **Do not select architectures on the validation split.** It is reliable for
  2-parameter offsets and not for choosing between models: asked to weight a
  one-vs-rest decomposition against the softmax, it picked the arm that is 0.162
  worse on ET precision in **20 of 20 splits** (`reports/one_vs_rest.md`).
* **Check what a reshape keeps.** `logbin` was exact on 64-column multitaper
  input and silently dropped 21 % of the band on 61-column welch input
  (`reports/band_truncation.md`).

## Known limits

* **Macro precision >0.90 on three classes is unreachable at any coverage**
  (`reports/precision_ceiling.md`). ET precision gets *worse* under abstention,
  so no confidence threshold rescues it.
* **The ceiling has a shape, and it is two populations.** The six ensemble
  members agree on **59.5 %** of patients and are **68.8 %** correct there; on
  the contested **40.5 %** they fall to **0.443 balanced accuracy** with a top-2
  margin four times narrower (`reports/ensemble_diversity.md`). The members are
  not near-copies — they disagree on 20.5 % of patient pairs — so the contested
  set is a real boundary, not an artifact of a redundant ensemble. Contested
  rate is **not** uniform: 0.573 for NewData against 0.307 for 2015, with class
  composition controlled and near-identical across cohorts.
* **Better uses of the current representation are exhausted.** Five families
  have now been tested against matched controls and every one is null or
  harmful: combination rule (7 pooling rules, `reports/pooling_rules.md`),
  ensemble size and member diversity (`reports/balanced_bagging.md`), class
  decomposition (`reports/one_vs_rest.md`, precET −0.162 *), training-set
  pruning (`reports/prune_training.md`, `reports/influence_prune.md`), and
  conditional routing on ensemble disagreement (`reports/contested_gating.md`,
  +0.001 against its fusion control). The contested region *does* retain
  structure — six of eight feature blocks clear chance there
  (`reports/contested_specialists.md`) — but nothing tried sees structure the
  deep model does not already see. What remains is a representation that
  separates those patients, not a better rule over this one.
* **PADS does not transfer to in-house patients** — it adds nothing to ET
  (+0.003) and significantly hurts PD (−0.082).
* **NewData has 6 ET** — a training cohort, not an evaluation one.
* **Pretrained vision backbones do not help.** Frozen ImageNet ViT-B/16 with a
  linear head reaches macro precision 0.501, below logistic regression on ten
  spectral descriptors (`reports/frozen_vit.md`).
* **Self-supervised pretraining does not transfer.** Masked-spectrum SSL on 3,081
  unlabelled recordings gives nothing once the evaluated patients are removed
  from the pretraining corpus — PADS precET −0.014, per-fold-excluded −0.029
  [−0.036, −0.014]. The gain first measured (+0.161) was a *frozen* treatment
  against a *fine-tuned* control; against a frozen control it is zero
  (`reports/ssl_retraction.md`).
* **Ten hand-computed spectral descriptors remain the best PD-vs-ET model on
  PADS** (precET 0.464 / macroP 0.705), ahead of every encoder tried.
* **21 in-house ET patients is the binding constraint, and it is now quantified.**
  The permutation null for in-house PD-vs-ET AUC spans **[0.298, 0.655]**, so a
  model must reach **AUC ≈ 0.66 before it can be told from chance at all**. The
  best measured is 0.629 (`axes`, p = 0.085). **No in-house single-model claim of
  the form "family X separates PD from ET" is supported** — including the axis
  result reported above, which should be read as underpowered, not established
  (`reports/permutation_null.md`). On PADS, five of six families clear the same
  test at p ≤ 0.010.
