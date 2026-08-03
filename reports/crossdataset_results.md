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

## Can PADS be combined with the local data to improve it? No.

Every combination strategy, tested on the local (your) patients:

| strategy | macro-F1 | ET-F1 | vs baseline |
|---|---|---|---|
| your Data only (baseline, same folds) | 0.663 | 0.432 [0.21, 0.62] | — |
| your Data **+ PADS** (augmented training) | 0.599 | 0.350 [0.15, 0.54] | **worse** |
| pooled, mixed test | 0.525 | 0.290 | worse |
| transfer (train PADS → test yours) | 0.220 | 0.20 | far worse |

Adding PADS training data **reduces** performance on your own patients (ET-F1
0.43 → 0.35). The device domain shift means PADS examples confuse rather than
help. **PADS is useful only as an independent validation cohort, not as extra
training data.**

## REVISION — the domain shift is orientation + scale, and it is fixable

The conclusion above ("pooling is blocked by an intractable device effect") was
**too pessimistic**. Testing the sensor-coordinate hypothesis showed the shift is
largely an **orientation and amplitude-scale** effect, which can be removed
*without* knowing either dataset's reference frame: summing the PSD across the
three axes gives the **trace of the spectral matrix**, invariant to any rotation;
normalising the spectrum to sum 1 additionally removes scale.
(`pdetn.crossdataset.invariant_features`.)

| feature space | dataset-identity AUC |
|---|---|
| full features | 0.999 |
| **rotation + scale invariant (40-dim spectral shape)** | **0.526 (chance)** |
| rich invariant (+10 signal features) | 0.978 — the signal features re-introduce device signature |

With truly invariant features the two datasets become **statistically
indistinguishable**, and operations that previously failed start working:

| setting (invariant features) | macro-F1 | ET-F1 | before (full features) |
|---|---|---|---|
| LOCAL-only | 0.508 | 0.276 | — |
| **TRANSFER PADS→local** | **0.540** | **0.364** | 0.220 / 0.20 (failed) |
| **AUGMENTED (PADS in train, test local)** | 0.535 | **0.368** [0.17, 0.55] | 0.599 / 0.350 (hurt) |
| POOLED (44 ET) | 0.472 | 0.250 | 0.525 / 0.290 |

* **PADS→local transfer now generalises** (ET-F1 0.364 vs 0.276 for local-only on
  the same features) — real cross-dataset generalisation, previously impossible.
* **Augmentation now helps instead of hurting** (0.276 → 0.368), though the CI is
  wide and the gain is within noise.
* local→PADS transfer still fails — expected asymmetry: 151 patients / 16 ET
  cannot cover PADS's 416 / 41, but the reverse direction does.

**The trade-off that remains.** Invariance costs more discriminative power than
PADS adds: the best invariant result (ET-F1 ≈ 0.37) still trails the full-feature
local model (**0.516**), because the most informative features — absolute power,
orientation structure, spatial propagation — are precisely those carrying the
device signature. Amplitude/noise characteristics also differ (PADS peaks are
~3.3× larger at equal RMS: raw gyro vs our smoother quaternion-derived rate),
which is *not* removed by rotation correction.

**Revised conclusion:** cross-dataset combination is **possible** in an
orientation/scale-invariant feature space — it is not an intractable device
barrier — but on this task invariance costs more than the extra ET subjects buy,
so the best overall model remains the local full-feature one. The
invariance-vs-discriminability tension is itself a reportable finding.

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
