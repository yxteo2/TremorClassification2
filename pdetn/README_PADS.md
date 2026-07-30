# Cross-dataset PD/N/ET with PADS — local run guide

The dev sandbox cannot download PADS (PhysioNet/Kaggle are network-blocked), so
run this on your own machine. Goal: add PADS's ~28 ET subjects to your ~16
(→ ~44) and test whether more ET data yields a better model — plus cross-dataset
generalisation, the result top venues want.

## 1. Get PADS
- PhysioNet: DOI **10.13026/m0w9-zx22**
  (https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/), or the
  Kaggle mirror. License CC BY-NC-SA 4.0 (academic use OK).
- Unzip to a folder, e.g. `PADS/` containing `movement/timeseries/*.txt` and
  `patients/patient_*.json`.

## 2. Confirm the adapter (5 minutes)
Open `tremor/pads_data.py` and check the four `VERIFY:` constants against the
real files:
- `GYRO_COLS` — the 3 gyroscope columns in `movement/timeseries/<id>_<Task>_<Wrist>.txt`.
- `LABEL_FIELD` + `LABEL_TO_LETTER` — the diagnosis field/strings in `patients/patient_<id>.json`
  (map Healthy→N, Parkinson→P, Essential Tremor→E).
- `TASK_TO_CONDITION` — which task maps to which condition (StretchHold→OUT,
  Relaxed→REST, an action task→WING).

Quick check:
```python
from tremor.pads_data import load_pads_recordings
recs = load_pads_recordings("PADS", conditions=["OUT"])
print(len(recs), "recordings", {r.y for r in recs})   # expect labels {0,1,2}
```

## 3. Run
```bash
# dry run first (local only, no PADS) — proves the pipeline works
python -m pdetn.pads_experiment --data-root Data

# full cross-dataset
python -m pdetn.pads_experiment --data-root Data --pads-root PADS \
    --action OUT --pads-condition OUT
```

## What it reports
- **P1 — generalisation:** train on LOCAL → test on PADS, and reverse. External
  ET-F1 + CI. The reviewer-gold number.
- **P2 — pooled LOSO (44 ET):** leave-one-subject-out over both cohorts. Tighter
  ET-F1 CI — the direct answer to "n=16".
- **Dataset-identity probe:** AUC that a classifier can tell the datasets apart.
  >0.85 ⇒ strong domain shift ⇒ pooled results are confounded (add domain
  alignment before trusting P2).

## Design notes
- **Single-sensor features only** (hand for local, wrist for PADS): STFT-256
  spectral profile + biomarker + regularity. The 3-sensor **spatial** features
  (your best in-house lever) are excluded — PADS has one wrist, no arm chain.
- Both datasets are 100 Hz gyroscope/angular-velocity, so the feature columns
  line up. Per-recording `per_recording` normalisation absorbs device-scale
  differences.
- If the dataset-identity AUC is high, the honest next step is a simple domain
  adapter (feature-distribution matching / CORAL) before pooling.
