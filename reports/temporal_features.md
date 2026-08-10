# Temporal descriptors — tested, and they make things worse

Every descriptor in `tfbench.descriptors` is computed from a spectrum already
averaged over time, so tremor **dynamics** were never measured: bursting vs
continuous, frequency wander, modulation rhythm. PD rest tremor is classically
intermittent and re-emergent while ET is more continuous, so this looked like a
real gap.

`tfbench/temporal.py` adds 13 descriptors along the time axis of the STFT:
amplitude CV, burst fraction, longest burst, bursts/s, frequency wander (std and
IQR), spectral flux and its CV, envelope autocorrelation at 1 s, envelope rhythm
frequency and strength, band-power slope, stationarity.

Validated on synthetic dynamics: continuous → amp_cv 0.000, freq_wander 0.000;
bursting → amp_cv 0.598; frequency-wandering → freq_wander 4.81, IQR 10.16.
(`burst_fraction` initially thresholded at the median, which returns 0.5 by
construction; it now thresholds at the mean so it carries the skew of the power
distribution — continuous 0.535, bursting 0.500, rare spikes 0.419.)

## Result: no help, and significant harm at REST

2015, lower_arm, PD-vs-ET, patient-level LOSO, paired bootstrap vs spectral-only:

| condition | features | n | bal-acc | AUC | precision | paired vs spectral |
|---|---|---|---|---|---|---|
| **REST** | spectral only | 10 | **0.730** | 0.729 | **0.393** | — |
| REST | temporal only | 13 | 0.505 | 0.472 | 0.179 | **−0.224 [−0.40, −0.04]** |
| REST | spectral + temporal | 23 | 0.547 | 0.605 | 0.222 | **−0.183 [−0.32, −0.06]** |
| OUT | spectral only | 10 | 0.460 | 0.496 | 0.139 | — |
| OUT | temporal only | 13 | 0.527 | 0.599 | 0.188 | +0.067 [−0.07, +0.20] |
| OUT | spectral + temporal | 23 | 0.480 | 0.545 | 0.152 | +0.020 [−0.13, +0.17] |

**At REST both temporal variants are significantly worse** — the paired CIs
exclude zero. This is stronger than the usual "no effect": adding 13 features to
the best configuration actively degrades it, from 0.730 to 0.547.

The mechanism is the same one that killed the multi-sensor experiment: 23
features against **16 ET subjects** overfits. Two independent feature families
have now failed in exactly this way, at exactly this n.

## Univariate screen — nothing survives correction

| condition | feature | effect | raw p | **BH q** |
|---|---|---|---|---|
| REST | env_rhythm_freq | −0.366 | 0.022 | 0.291 |
| OUT | env_autocorr_1s | +0.372 | 0.024 | 0.307 |
| OUT | longest_burst_s | +0.322 | 0.050 | 0.326 |
| OUT | n_bursts_per_s | −0.310 | 0.060 | 0.258 |

Zero of 13 survive BH q<0.05 in either condition.

## The one thing worth noting

At **OUT**, where the spectral descriptors sit at chance (0.460, AUC 0.496),
temporal-only reaches 0.527 with **AUC 0.599** — the higher AUC says the
temporal block carries *some* independent ranking information that the spectral
block does not. It is not enough to be useful, and it cannot be combined without
overfitting, but it is not pure noise either.

If ET ever reaches n≈30, this is worth retesting: the features are implemented
and validated, and the failure mode here is sample size rather than the idea.

## Standing best, unchanged

**2015 REST, lower_arm, stft512, spectral descriptors only, threshold 0.5 —
bal-acc 0.730, AUC 0.729, ET precision 0.393, ET-F1 0.500 [0.31, 0.67].**
