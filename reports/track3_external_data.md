# Track 3 — External data & cross-dataset validation protocol

**Purpose.** The binding limit on this project is **16 ET subjects**; no model
change moves it (we proved that — every architecture/fusion sits inside the
same CI). The only lever that answers a reviewer's "n=16, underpowered" is
**external ET data + cross-dataset validation.** This document assesses public
options and specifies a concrete protocol.

## Dataset landscape (assessed)

| dataset | modality | ET / PD / HC | fs | tasks | fit | access |
|---|---|---|---|---|---|---|
| **PADS** (Varghese 2024) | wrist IMU, **accel + gyro** | **28 / 291 / 79** | **100 Hz** | 11 incl. StretchHold, Relaxed | **excellent** | open, CC BY-NC-SA 4.0 |
| TIM-Tremor | video + wrist **accel only** | mixed (ET+PD), 55 total | — | 21 | weak (no gyro) | open, 4TU |
| WearGait-PD | body IMU (gait) | PD + controls; **ET not labelled** | — | gait | poor (gait, no ET label) | open |
| jNER 2020 (PD vs ET) | motion sensors | PD + ET | — | gait/balance | n/a | **not public** |

**Winner: PADS.** It is a near-exact modality/protocol match to our data and it
roughly **triples the ET cohort**.

## Why PADS fits our pipeline almost 1:1

Our data: 3 arm sensors → quaternion → **angular velocity @ 100 Hz**, tasks
OUT/REST/WING, classes N/PD/ET, **16 ET**.

- **Same sampling rate (100 Hz)** and **same core modality** — PADS records a
  3-axis **gyroscope (angular velocity)**, exactly our feature. (PADS also has
  accelerometer; we ignore it to match our angular-velocity-only representation.)
- **Task overlap maps cleanly to our conditions:**
  - `StretchHold` (arms outstretched, postural) → **OUT**
  - `Relaxed` → **REST**
  - an action task (`DrinkGlas` / `TouchNose` / `LiftHold`) → **WING**-like
- **Label mapping:** PADS `HC → N`, `PD → PD`, `ET → ET`. The PADS
  differential-diagnosis group also contains non-ET disorders (atypical/
  secondary parkinsonism, MS) — **filter to ET only**.
- **Cohort impact:** ET **16 → 44** subjects when pooled; PD and N/HC also grow
  by an order of magnitude.

### Known file layout (to verify on download)
Timeseries: `movement/timeseries/<id>_<Task>_<Wrist>.txt` (e.g.
`155_StretchHold_RightWrist.txt`), 6-axis rows at 100 Hz. Labels/demographics:
`patients/patient_<id>.json`. Enums in `scripts/utils/constants.py`.
PhysioNet blocks automated fetch (403), so the exact JSON field names and column
order must be confirmed against the downloaded files — see the adapter skeleton
`tremor/pads_data.py`.

### Domain shifts we must control
1. **Sensor placement:** ours = 3 arm sensors; PADS = wrist only. Harmonise by
   using **our `hand` sensor only** (3 angular-velocity channels) ↔ PADS
   **wrist gyroscope** (3 channels). The skill already notes "hand carries the
   most tremor," so this costs little.
2. **Device/units:** Apple-Watch gyro (rad/s) vs our quaternion-derived rate —
   absorb with **`per_recording` z-score** (already our default) so absolute
   scale doesn't leak dataset identity.
3. **Wrist choice:** PADS has both wrists; use the more-affected (or the one
   with higher tremor-band power), or average — decide empirically.
4. **Task semantics:** confirm StretchHold posture matches OUT; start with the
   single cleanest pair before adding tasks.

## The protocol — three tiers of increasing ambition

All tiers keep the non-negotiables: **subject-level** splits, train-fold-only
normalisation/augmentation, val-loss model selection, one test scoring, and the
subject bootstrap CI + permutation p-value we already have.

- **P1 — External generalisation (the reviewer-gold result).**
  Train on ours (OUT ≈ StretchHold), test on PADS (StretchHold), and the reverse.
  Report ET-F1 + CI **on the external cohort**. Directly answers "does it
  transfer?" This is the single most valuable number for a Transactions bar.

- **P2 — Pooled cross-dataset LOSO (the n-fix).**
  Combine both cohorts → **44 ET subjects**, subject-level LOSO over the union,
  with `dataset` as a tracked covariate. Tighter CI, the direct answer to "n=16."
  **Guard:** a *dataset-identity probe* — train a classifier to predict which
  dataset a recording came from; if it succeeds trivially, pooled disease
  results are confounded and need domain alignment before they can be trusted.

- **P3 — Leave-one-dataset-out (the strongest claim).**
  Train on the pooled data minus one dataset, test on the held-out dataset.
  The cleanest generalisation statement; report both directions.

### Domain-shift controls (report alongside)
- Dataset-identity probe (above) as a confound check on P2.
- `per_recording` normalisation + hand/wrist harmonisation as the baseline
  alignment; add a simple domain adapter (feature-distribution matching, e.g.
  CORAL) only if the probe shows strong dataset separability.
- Keep the in-CV ET threshold selection (Track 1) — re-fit the threshold on the
  *training* cohort's validation folds, never on the external test set.

## Expected payoff
- **P1/P3** give cross-dataset generalisation — the result top venues reward and
  the one thing 16 in-house subjects can never provide.
- **P2** takes ET from 16 → 44 subjects, roughly halving the ET-F1 CI width and
  neutralising the "underpowered" objection.

## Next actions
1. **Download PADS** — PhysioNet DOI `10.13026/m0w9-zx22` (or the Kaggle mirror).
   License **CC BY-NC-SA 4.0**: academic use is fine, but it is **non-commercial**
   and **share-alike** — note this if any part of the pipeline is meant to ship
   commercially (it constrains the ONNX/C++ deployment path).
2. **Confirm the schema** of `patient_<id>.json` and the timeseries column order
   against `tremor/pads_data.py` (skeleton provided, marked with `VERIFY:`).
3. **Run P1** first (cheapest, highest-value) — train on our OUT, test on PADS
   StretchHold, ET/PD/HC, report external ET-F1 + CI.
4. Then **P2** (pooled LOSO, 44 ET) with the dataset-identity probe.

## Sources
- PADS: Varghese et al., *Machine Learning in the Parkinson's disease smartwatch
  (PADS) dataset*, npj Parkinson's Disease 10:9 (2024) —
  https://www.nature.com/articles/s41531-023-00625-7 ;
  PhysioNet v1.0.0 https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
  (DOI 10.13026/m0w9-zx22).
- TIM-Tremor: https://doi.org/10.4121/uuid:522d14ed-3019-4206-b49e-a4e674b6440a
- WearGait-PD: https://www.nature.com/articles/s41597-026-06806-2
- PD-vs-ET wearable (not public): https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-020-00756-5
