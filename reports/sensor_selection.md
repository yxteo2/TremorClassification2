# Sensor selection — the forearm (lower_arm) sensor is best

Motivated by matching PADS (a wrist smartwatch), we compared single-sensor
separation. The result was decisive and surprising.

## Single-sensor two-stage (OUT, LOO, tuned ET threshold)

| sensor (position) | macro-F1 | ET-F1 [95% CI] | PD-vs-ET acc |
|---|---|---|---|
| hand (distal) | 0.601 | 0.389 [0.17, 0.58] | 0.81 |
| **lower_arm (≈ wrist)** | **0.704** | **0.516 [0.27, 0.71]** | **0.84** |
| upper_arm (proximal) | 0.579 | 0.250 [0.07, 0.43] | 0.65 |

## Finding
**The lower_arm (forearm) sensor alone is the best in the whole study** —
ET-F1 0.516, macro-F1 0.704 — beating the deep model (~0.47), TF+spatial@OUT
(0.421), and every all-sensor / multi-feature combination.

Interpretation: the hand performs fine voluntary motion that contaminates the
tremor band, and the upper arm barely moves; the **forearm is the sweet spot**
where tremor is expressed cleanly. Using all 9 channels *diluted* the signal
with noisier channels — restricting to the right single sensor beat using more.

## Consequences
- **Best interpretable model: lower_arm single-sensor, ET-F1 0.516** (features:
  STFT-256 profile + biomarker + regularity, no spatial).
- The PADS cross-dataset match uses lower_arm (wrist-equivalent) — so it uses the
  *strongest* single sensor, not a compromise. `load_local_sensor` defaults to
  lower_arm.

## lower_arm across conditions (confirmed)

| condition | macro-F1 | ET-F1 | PD-vs-ET acc | p |
|---|---|---|---|---|
| **OUT** | **0.704** | **0.516 [0.27, 0.71]** | 0.84 | 0.0005 |
| WING | 0.607 | 0.333 [0.09, 0.56] | 0.78 | 0.0005 |
| REST | 0.501 | 0.143 [0.00, 0.29] | 0.69 | 0.0005 |

**lower_arm + OUT is the confirmed best config** — OUT decisively beats WING/REST
(REST weakest: quiet rest tremor). This is the session's headline interpretable
model: **macro-F1 0.704, ET-F1 0.516, p=0.0005.**

Honest caveat: ET-F1 CI is wide ([0.27, 0.71], 16 ET subjects) — a real point
improvement that still overlaps prior results at the tail. The result is
deterministic (logreg + LOO), so it is not a lucky seed. Reproduce:
`load_local_sensor(..., sensor="lower_arm", action="OUT")` + `build_features` +
`protocol_p2_pooled_loso`.
