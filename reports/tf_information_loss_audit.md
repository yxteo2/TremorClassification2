# Audit: what the time-frequency pipeline does to magnitude and orientation

Asked because both are physically meaningful — tremor **amplitude** is clinical,
and **which axis** the limb oscillates along distinguishes pronation-supination
from flexion-extension. Below is what each stage actually preserves, verified in
code and measured, not assumed.

## Summary

| stage | absolute magnitude | inter-channel ratio | orientation (per-axis) | spectral contrast |
|---|---|---|---|---|
| quaternion → angular velocity | **faithful** (matches raw gyro) | kept | **kept** | faithful |
| `tfbench` transforms | kept in `total_power` | **lost** (channels averaged) | **lost** | kept |
| `tfbench` descriptors | only `total_power` | n/a | n/a | kept (`q_factor`) |
| `TremorDataset` (deep path) | **removed** by z-score | kept | **kept** (stacked rows) | largely kept |

## 1. A claim I nearly made and had to retract

`log_compress(eps=1e-8)` followed by `per_recording_zscore` computes
`(log S − mean log S) / sd(log S)`. On a **toy** 1×20 single-peak spectrum this
looked catastrophic: a 100:1 peak and a 3:1 peak both normalised to **exactly
4.59 sd units** — apparently a total collapse of peak-sharpness information.

**On real spectrograms it is not true.** Measured over 60 recordings:

| | raw `log10(peak/median)` | after log+z-score |
|---|---|---|
| range | [0.44, 1.61] | [1.14, 2.76] |
| correlation raw vs normalised | — | **r = +0.890** |
| relative spread retained | — | 0.71× |

Contrast **largely survives**. The toy case is pathological: with one peak over a
flat floor the sd is set entirely by that peak, so dividing by it cancels the
contrast exactly. A real spectrogram has ~30 frequency bins × many frames × 9
channels, so the sd reflects the whole distribution and peak sharpness comes
through. The `per_recording_zscore` docstring is essentially accurate.

Recorded because the toy result was dramatic and wrong, and this project has a
track record of reporting exactly that kind of thing before checking it.

## 2. What IS lost — absolute amplitude (deep path)

`per_recording_zscore` subtracts the per-recording mean, so **absolute tremor
amplitude is removed** from every deep model (`tfbench/deep.py`,
`pdetn/deep_eval.py`, `pdetn/deep_crossdataset.py`, `tremor.cv_benchmark`).

This is deliberate — it absorbs sensor calibration and per-patient gain, and it
is what keeps the dataset-identity probe from trivially reading device scale.
But it is a real trade: amplitude is clinically informative.

Measured cost of restoring it explicitly (patient-level LOSO, OUT):

| descriptor set | n_feat | PD-vs-ET bal-acc | N-vs-Tremor bal-acc |
|---|---|---|---|
| channel-averaged (current `tfbench`) | 10 | 0.640 | 0.793 |
| per-channel (orientation kept) | 90 | 0.600 | 0.818 |
| **absolute amplitude + log-contrast** | 36 | **0.647** | 0.810 |
| per-channel + absolute amplitude | 126 | 0.620 | **0.834** |

**These are point estimates with no paired CIs — they are not claims.** The
differences are small and the project's own history says such gaps do not
survive a paired test. What they do show is that nothing here is being lost at a
scale that would change conclusions.

## 3. What IS lost — orientation (classical path only)

`signal_processing.transforms._per_freq_mean` averages the per-channel spectra, so **every
`tfbench` descriptor is orientation-blind by construction**. This was a
deliberate choice (rotation invariance, and it keeps descriptors comparable
across the 3-sensor local data and single-sensor PADS) but it should be explicit.

Keeping orientation (per-channel descriptors, table above) **helps N-vs-Tremor**
(0.793 → 0.818) and **hurts PD-vs-ET** (0.640 → 0.600) — 9× the features against
15 ET subjects is straightforward overfitting. So orientation carries usable
information on the easy axis and cannot be exploited on the hard one at this n.

**The deep path does not have this problem.** `TremorDataset` stacks channels as
separate rows of the `(n_ch·n_freq, T)` image, so networks see per-axis spectra
and orientation is fully available to them.

## 4. The differentiation tilt — RETRACTED, it does not exist in practice

**What I wrote here first:** that `ω = 2 q̇ q⁻¹` multiplies the amplitude
spectrum by ω, tilting energy upward, and that angular velocity and the log map
are "both biased, in opposite directions." The evidence was that the same 40
recordings give median dominant frequency 6.45 Hz via angular velocity and
4.69 Hz via the log map.

**That inference was wrong.** The 2025 `.h5` files store a raw `Gyroscope`
stream alongside `Processed/Orientation`, which lets the conversion be checked
directly rather than argued from theory. Same sensor, same recording,
normalised spectra:

| | peak | 1–3 Hz | 3–6 Hz | 6–9 Hz | 9–15 Hz | >18 Hz |
|---|---|---|---|---|---|---|
| quaternion-derived ω | 5.47 Hz | 0.095 | 0.781 | 0.119 | 0.004 | 0.000 |
| **raw gyroscope** | **5.47 Hz** | 0.092 | 0.782 | 0.120 | 0.005 | 0.000 |

**Identical to three decimal places.** The quaternion → angular-velocity
conversion is faithful. It has to be: the orientation stream is itself produced
by the sensor's fusion filter *from* the gyroscope, so differentiating it
recovers the gyroscope almost exactly. There is no tilt to correct.

The 6.45 vs 4.69 Hz gap is therefore **not** symmetric bias. Angular velocity is
correct; **the log map is the biased one**, because it retains low-frequency
pose drift inside the 3–15 Hz band and that drags its peak down. The earlier
recommendation to "document the ω tilt wherever a frequency is quoted" is
withdrawn — frequencies measured from angular velocity need no such caveat.

Two further consequences:

* A de-tilting experiment (dividing LOCAL PSDs by ω²) made the dataset-identity
  probe **much worse**, 0.629 → **1.000**. Correcting a distortion that is not
  there simply manufactures a new dataset-specific signature.
* The 2.4–2.6× more relative power above 18 Hz in LOCAL vs PADS is therefore
  **not** a differentiation artifact. It is a genuine device difference between
  the Moveo/2015 units and the Apple Watch.

**Unused data worth knowing about:** `Sensors/<id>/` in the 2025 files also holds
`Accelerometer`, `Magnetometer`, `Barometer` and `Temperature`, none of which the
pipeline touches. The accelerometer in particular is a genuinely independent
modality and is what PADS pairs with its gyro.

## 5. Already-known issue, confirmed still handled

Vector-magnitude reduction `sqrt(Σ ch²)` rectifies the signal and doubles the
apparent frequency (`reports/signal_processing_summary.md`). **`tfbench` does not
do this** — `transforms.py` computes each channel's spectrum separately and
averages the spectra, which is linear and introduces no rectification. Verified
by the synthetic check: all 12 methods recover a 6 Hz tone at 6 Hz, not 12 Hz.

## Recommendations

1. **Leave the deep-path z-score as is.** It is the reason cross-dataset pooling
   is possible at all; removing it would let models read device gain.
2. **Add absolute amplitude as explicit features** rather than trying to recover
   it from the normalised spectrogram — cheap, and the only way to have both.
3. ~~Document the ω tilt wherever a frequency is quoted.~~ **Withdrawn** — the
   conversion is faithful to the raw gyroscope (§4). No caveat needed.
4. **Leave `tfbench` channel-averaged** for cross-dataset comparability, but use
   per-channel descriptors for N-vs-Tremor, where they help.

## 6. Second audit round — remaining checks

| check | result | verdict |
|---|---|---|
| truncation to 25th-pct length | 5.9 % of all samples discarded, median 97 % of each recording kept | fine |
| 3–15 Hz band crop | **33 % of power lies below 3 Hz** (max 67 %) | see below |
| quaternion resampling 128→100 Hz | `resample_poly` leaves ‖q‖ off unit by up to **11 %** | corrected downstream by `_normalize_quaternions` |
| LOCAL vs PADS amplitude | median ratio 0.5×, but **p99 ratio ~10×** | genuine device difference |

### The band crop is the one worth acting on

A third of the signal power sits below the 3 Hz floor. Widening the analysis
band (Welch descriptors, patient LOSO, lower_arm):

| band | PD-vs-ET bal-acc | N-vs-Tremor bal-acc |
|---|---|---|
| 3–15 Hz (current) | 0.513 | 0.843 |
| 2–15 Hz | 0.520 | 0.799 |
| **1–15 Hz** | **0.553** | 0.854 |
| 3–20 Hz | 0.500 | 0.837 |
| **2–25 Hz** | **0.553** | **0.857** |
| 1–30 Hz | 0.553 | 0.812 |

Including sub-3 Hz content lifts PD-vs-ET from 0.513 to 0.553 consistently
across three different wide bands. **Point estimates, no paired CIs — not a
claim**, but the direction is consistent and the band floor has never been swept
before. Worth a proper test with `tfbench`.

### The identity probe depends enormously on the feature set

| feature set | LOCAL-ET vs PADS-ET identity AUC |
|---|---|
| STFT-702 profile | 1.000 |
| orbit geometry (66) | 0.959 |
| **tfbench descriptors (10)** | **0.629** |
| gravity-chirality (15) | 0.567 (vs NewData) |

The compact descriptor set is far less device-revealing than the raw spectral
profile — the identity signal lives mostly in the high-dimensional spectrum
shape, not in the summary statistics. That makes `tfbench` descriptors a
candidate for legitimate cross-dataset pooling.

**But pooling still does not rescue the local model:**

| model (tfbench descriptors) | PD | ET | AUC | bal-acc | ET-F1 |
|---|---|---|---|---|---|
| LOCAL only | 75 | 15 | 0.524 | 0.507 | 0.240 |
| **PADS only** | 276 | 28 | **0.775** | **0.736** | **0.387** |
| LOCAL + PADS (scored on LOCAL) | 75 | 15 | 0.492 | 0.527 | 0.255 |

### The observation I cannot explain

On **identical features and an identical protocol**, PADS reaches bal-acc 0.736
while the local cohort sits at 0.507 — chance. That is a very large gap and it is
not a cohort-size artifact (the power curve in `docs/IMPLEMENTATION_PLAN.md`
shows PD-vs-ET plateauing by n=15). Candidate explanations, none tested:

1. the Apple Watch wrist gyro is a cleaner tremor measurement than the
   Moveo-derived rate (the p99 amplitude gap and the >18 Hz difference are
   consistent with different anti-alias/fusion filtering);
2. PADS's PD cohort (n=276) is more severe or more homogeneous;
3. `StretchHold` elicits tremor more reliably than the local `OUT` protocol.

**This is the most important open question the audit surfaced** — if it is (1),
the local measurement chain is the bottleneck, and that is fixable in a way that
more subjects is not.
