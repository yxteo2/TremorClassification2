# Tremor Classification — N / PD / ET from wearable IMU

## Start here

```bash
jupyter notebook START_HERE.ipynb
```

One notebook, top to bottom: loads and verifies all three cohorts, reproduces
the frequency tables, the best PD-vs-ET model, and the externally validated
N-vs-Tremor result. Executed with outputs committed, so the numbers are
readable without running anything.

Deeper dives live in `tfbench/`:
`01_signal_processing_benchmark` (12 TF methods),
`02_deep_model_comparison` (architectures),
`03_cohort_comparison` (2015 vs NewData vs PADS).

**Read balanced accuracy and precision together.** Majority baselines are 0.833
(2015 PD-vs-ET) and 0.908 (PADS), so raw accuracy misleads.

Classifies Normal, Parkinson's and Essential Tremor from arm-worn IMU
quaternion recordings, via two diagnostic axes: **N vs Tremor** (screening) and
**PD vs ET** (differential).

## Headline results
Local cohort, forearm sensor, subject-grouped CV — see `reports/final_results.md`.

| axis | headline | positive class |
|---|---|---|
| **N vs Tremor** | acc 0.884 [0.832, 0.929], **AUC 0.936** | Tremor P 0.92 / R 0.88 / F1 0.90 |
| **PD vs ET** | **AUC 0.873**, balanced-acc 0.673 | ET P 0.60 / R 0.40 / F1 0.48 |

Note: on PD-vs-ET, raw accuracy is meaningless ("always PD" = 0.833). Report AUC
and balanced accuracy.

## Layout
    Data/               2015 cohort, quaternions @100 Hz (OUT / REST / WING)
    NewData/            2025 cohort, Moveo h5 @128 Hz (6 ET subjects)
    pads_stretchhold/   PADS StretchHold extract (validation cohort only)
    preprocessed/       PADS-derived intermediates
    tremor/             core pipeline: loaders, TFDs, models, metrics, stats
    pdetn/              condition-aware analysis, feature families, notebooks
    reports/            all findings (start with final_results.md)
    docs/               handoff notes, licence
    notebooks/          exploratory notebooks
    scripts/            PADS reference scripts (from the dataset authors)

## Key findings
* **Forearm sensor alone beats all three** on both axes.
* **Frequency alone cannot separate PD from ET** — identical 6.64 Hz medians,
  50-70% distribution overlap, replicated in two independent cohorts. The signal
  lives in the full spectral profile (AUC 0.88) not in summary statistics (0.61).
* **8 time-frequency methods** compared (STFT/CWT/HHT/wavelet/multitaper/SST/
  VMD/S-transform) — STFT-256 best, all within CI.
* **PADS cannot be used as training data** (device domain shift, identity AUC
  0.999); it serves as an independent validation cohort.
* Simple models beat deep ones here — the cohort (15 ET) is the binding limit.

## Reproduce
    pip install -r requirements.txt
    python -m pdetn.run --data-root Data            # interpretable pipeline
    python -m tremor.cv_benchmark --data-root Data --action OUT ...   # deep
Notebooks in `pdetn/` cover the two-stage comparison, decomposition study and
cross-dataset work.

Next steps: `docs/NEXT_SESSION.md`.
