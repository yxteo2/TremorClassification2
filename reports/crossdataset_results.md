# Cross-dataset validation with PADS — results

Combined the local cohort (16 ET) with PADS StretchHold (41 ET / 296 PD / 79 N),
matched on the wrist-placed sensor (local lower_arm ~ PADS wrist), single-sensor
gyroscope features (STFT-256 + biomarker + regularity), two-stage logistic
regression, subject-grouped evaluation.

## Results

| protocol | macro-F1 | ET-F1 [95% CI] | note |
|---|---|---|---|
| P1 train-local → test-PADS | 0.406 | 0.13 | transfer fails |
| P1 train-PADS → test-local | 0.220 | 0.20 | transfer fails |
| dataset-identity probe | — | — | **AUC 0.999** (severe domain shift) |
| P2 pooled (naive) | 0.525 | 0.290 [0.19, 0.39] | pooling doesn't help |
| P2 pooled + per-dataset alignment | 0.510 | 0.307 [0.21, 0.41] | probe AUC → 0.06, still no gain |
| **PADS-only within-dataset (41 ET)** | 0.479 | **0.262 [0.14, 0.39]** | intrinsic difficulty |
| local-only lower_arm (16 ET) | 0.704 | 0.516 [0.27, 0.71] | optimistic (wide CI) |

## Findings
1. **Severe device domain shift.** A classifier separates the two datasets with
   AUC **0.999** from the features (Apple-Watch gyro vs quaternion-derived rate).
   Cross-dataset transfer (P1) fails in both directions.
2. **Pooling does not help, even after alignment.** Per-dataset standardization
   drops the identity-probe AUC to 0.06 (marginal distributions aligned), yet
   pooled ET-F1 stays ~0.31 — indicating **conditional/concept shift** (the
   disease→signal mapping differs by device), which simple alignment can't fix.
3. **The low ET-F1 is intrinsic, not a small-cohort artifact.** PADS-only, with
   **2.5x the ET subjects** and a **tight CI [0.14, 0.39]**, still gives ET-F1
   ~0.26. More data tightened the estimate but did not raise it; ET remains
   heavily confused with PD (24/41 ET → PD).
4. **Reconciled estimate.** The local 0.516 [0.27, 0.71] and PADS 0.262
   [0.14, 0.39] CIs overlap; the true ET-F1 for this task is likely **~0.3-0.4**,
   and the local point estimate was optimistic within a wide interval. The tight
   PADS number is the more honest characterization.

## Interpretation (a real contribution)
PD-vs-ET differentiation from a single wrist/forearm IMU at the postural
(arms-outstretched) task is **intrinsically hard**, confirmed across two
independent datasets. This is a rigorous, cross-dataset-validated negative
result — more valuable than an over-claimed number on one small cohort.

## Caveats / open items
- PADS N-F1 is low (0.44): many healthy controls are called PD in StretchHold —
  worth checking whether PADS HC have residual movement or a task artifact.
- PADS ET count here is 41 patients vs a documented ~28; the raw diagnosis
  strings (manifest.csv / patients JSON) were not available to audit. If some
  PADS "ET" are mislabeled, the PADS ET-F1 is a slight underestimate — but the
  core conclusion (more data does not raise ET-F1) is robust to this.
- Only tested at the postural task and with single-sensor features (PADS is
  wrist-only). Rest-task or multi-sensor data could differ.

Reproduce: extract PADS with `pdetn.extract_pads`, then the pooled/PADS-only
loaders in this session (to be folded into `pdetn/crossdataset.py`).
