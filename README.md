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
| welch baseline | 0.639 | 0.636 | 0.583 | 0.619 |
| **multitaper + IF trajectory** | 0.639 | 0.655 | **0.685** | **0.660** |

Paired +0.041 [+0.014, +0.067] macro precision over 20 splits.

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

experiments/         final_model.py               the merged model
                     oneclass_paired.py           one-class PD + logreg hybrid
                     binning.py                   band coverage vs estimator
                     binning_deep.py              the same, paired, 3-class deep
                     masked_pretrain.py           masked-spectrum SSL
                     ssl_leakage.py               is the SSL gain transductive?
                     ssl_matched.py               SSL with a matched pipeline
                     own_data_10et.py             in-house, 10 ET in test
                     inhouse_axes.py              axis features in-house
                     frozen_backbone.py           frozen pretrained ViT
                     attention_test.py            small attention models
                     trajectory_tuning.py         trajectory sweep
                     window_training.py           window-level training
                     selection_and_calibration.py selection, calibration, seeds
                     audio_techniques.py          freq-aware conv, PCEN, SpecAugment

reports/             51 findings, including every retraction
```

The ViT checkpoint is stored split; rebuild with `cat vit_chunk_0* > vit_fp16.pt`.

## Conventions that matter

* **Patient-level splits only.** Never split one patient's recordings.
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
* **Check what a reshape keeps.** `logbin` was exact on 64-column multitaper
  input and silently dropped 21 % of the band on 61-column welch input
  (`reports/band_truncation.md`).

## Known limits

* **Macro precision >0.90 on three classes is unreachable at any coverage**
  (`reports/precision_ceiling.md`). ET precision gets *worse* under abstention,
  so no confidence threshold rescues it.
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
