# Tremor classification — N / PD / ET from wearable IMU

Classifying Normal, Parkinson's and Essential Tremor from wrist/arm inertial
recordings, across three cohorts.

Two lines of work:

1. **Time-frequency processing** — characterise the tremor and classify from
   frequency quantities (mean, max, bandwidth, peak sharpness).
2. **Deep learning** — classify each patient's time-frequency signal.

## Start here

| notebook | what it does |
|---|---|
| `01_tremor_characteristics.ipynb` | frequency characteristics per class; classification from mean/max frequency, features added one at a time |
| `02_deep_model.ipynb` | the final two-stream deep model and each component's contribution |

Or from the command line:

```bash
python -m frequency.characteristics    # goal 1: characteristics + frequency classification
python -m experiments.final_model        # goal 2: the deep model, paired against baseline
python -m common.cohorts  # how the three cohorts should be combined
```

## Headline results

**Frequency characteristics (PADS, 383 patients)** — ET tremor is markedly more
sharply peaked and lower in frequency than PD:

| | N | PD | ET |
|---|---|---|---|
| max frequency (Hz) | 7.20 | 7.07 | 6.16 |
| mean frequency (Hz) | 8.02 | 7.67 | 6.61 |
| bandwidth (Hz) | 2.94 | 2.48 | 2.04 |
| peak sharpness | 4.08 | 5.80 | **12.19** |

**N vs Tremor** — six frequency numbers and a logistic regression reach
**precision 0.910 (2015) / 0.924 (PADS)**.

**PD vs ET, 3-class merged model** — two-stream deep model (multitaper spectrum
+ instantaneous-frequency trajectory):

| | precN | precPD | precET | macro precision |
|---|---|---|---|---|
| baseline | 0.639 | 0.636 | 0.583 | 0.619 |
| **final model** | 0.639 | 0.655 | **0.685** | **0.660** |

paired +0.041 [+0.014, +0.067] macro precision, +0.102 [+0.031, +0.175] ET
precision, over 20 splits.

## Data

| cohort | patients | N / PD / ET | source |
|---|---|---|---|
| 2015 | 151 | 61 / 75 / 15 | `Data/` — quaternion, 3 sensors, OUT/REST/WING |
| NewData | 56 | 27 / 23 / 6 | `NewData/` — 2025 Moveo, both limbs, 7 tasks |
| PADS | 383 | 79 / 276 / 28 | `pads_stretchhold/`, `pads_relaxed/` — both wrists |

PADS is extracted with `python -m common.extract_pads`. Class labels are
re-derived from the manifest by **exact** diagnosis match — an earlier substring
match put 13 non-ET records in the ET class (`reports/pads_label_bug.md`).

## Package layout

```
models/              deep learning architectures
  architectures.py     CNN / TCN / BiLSTM / two-stream / attention, all of them

signal_processing/   time-frequency transforms and signal-level methods
  transforms.py        12 TF methods on one interface, all power-scaled
  tfd.py               multitaper / SST / CWT implementations
  quaternion.py        quaternion -> angular velocity, log map, gravity
  preprocessing.py     band-pass, framing, STFT magnitude
  spectral.py          log compression, per-frequency normalisation, SpecAugment
  stability.py         Tremor Stability Index, instantaneous-frequency trajectories
  signal_features.py   Hjorth, sample entropy, amplitude modulation

frequency/           the mean / max frequency method
  characteristics.py   6 characteristics + cumulative classification   (goal 1)
  descriptors.py       10 spectral descriptors
  tables.py            per-patient spectra, bilateral asymmetry features
  biomarker.py         Welch PSD, band power
  report.py            frequency comparison across cohorts

common/              data loading, cohort assembly, training
  data.py              the Recording type
  quaternion_data.py   2015 cohort loader
  load_2025.py         NewData loader, task/side selection
  extract_pads.py      PADS extraction (exact-match labelling)
  loaders.py           PADS loader
  cohorts.py           merged-cohort assembly, capping, missing-modality asymmetry
  training.py          training loops (full-batch; minibatch measured worse)
  protocol.py          train/val/test protocol, validation-tuned priors
  datasets.py          torch Datasets
  cache.py             on-disk caching

metrics/             evaluation
  evaluate.py          classification report
  stats.py             subject-clustered bootstrap CIs, permutation tests
  selective.py         precision at reduced coverage (abstain option)
  benchmark.py         method ranking with BH + Bonferroni
  merged.py            balanced accuracy, cohort-identity probe

experiments/         runnable studies (library code lives above)
  final_model.py       the final model, paired against baseline   (goal 2)
  audio_techniques.py  frequency-aware conv, PCEN, SpecAugment

reports/             23 findings, including the retractions
```

## Conventions that matter

* **Patient-level splits only.** Never split recordings of the same patient.
* **Report per-class precision**, with the test set's class prevalence —
  precision is not comparable across differently-composed test sets.
* **Paired bootstrap CIs** for every comparison. Unpaired differences of ~0.04
  sit inside the per-config sd here and have repeatedly evaporated when paired.
* **Two protocols, different questions.** Mixed-cohort (all three sources in
  train/val/test) answers "how well at sites we trained on"; leave-one-cohort-out
  answers "will this transfer". Only LOCO supports a generalisation claim.

## Known limits

* **Macro precision >0.90 on three classes is not reachable** at any coverage
  (`reports/precision_ceiling.md`). It needs more ET patients or the PADS
  non-motor questionnaire, not more architecture.
* **NewData has 6 ET patients** — a training cohort, not an evaluation one.
* **Feature unions dilute.** Seven have underperformed their best member; at 404
  patients with 49 ET, dimensionality binds harder than information.
