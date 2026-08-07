# Audit: what the time-frequency pipeline does to magnitude and orientation

Asked because both are physically meaningful — tremor **amplitude** is clinical,
and **which axis** the limb oscillates along distinguishes pronation-supination
from flexion-extension. Below is what each stage actually preserves, verified in
code and measured, not assumed.

## Summary

| stage | absolute magnitude | inter-channel ratio | orientation (per-axis) | spectral contrast |
|---|---|---|---|---|
| quaternion → angular velocity | rescaled (ω tilt) | kept | **kept** | tilted |
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

`tfbench.transforms._per_freq_mean` averages the per-channel spectra, so **every
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

## 4. The differentiation tilt — a real distortion, unavoidable

`ω = 2 q̇ q⁻¹` is a derivative, which multiplies the amplitude spectrum by ω.
Energy is tilted toward high frequencies, so the *apparent* dominant frequency
shifts up relative to an orientation (displacement-like) representation. Measured
on the same 40 recordings:

| representation | median dominant frequency |
|---|---|
| `angular_velocity` (derivative) | **6.45 Hz** |
| `log_map` (pose) | **4.69 Hz** |

Both are biased, in opposite directions: angular velocity is tilted **up** by
differentiation; the log map retains low-frequency **pose drift** inside the
3–15 Hz band, pulling its peak **down**. Neither is "the" tremor frequency.

Practical consequence: any absolute frequency quoted in the paper must state the
representation it was measured in. The biomarker frequencies in
`reports/biomarker.md` come from per-channel Welch PSD of angular velocity and
therefore carry the ω tilt.

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
3. **Document the ω tilt** wherever a frequency is quoted.
4. **Leave `tfbench` channel-averaged** for cross-dataset comparability, but use
   per-channel descriptors for N-vs-Tremor, where they help.
