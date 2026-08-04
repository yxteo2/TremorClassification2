# Temporal-spatial features — exploiting the 3-sensor arm geometry

Adds the spatial axis (hand / lower_arm / upper_arm propagation) on top of the
time-frequency transform. Features in `pdetn/spatial_features.py`: per-sensor
tremor power, distal→proximal power gradients, distal concentration, pairwise
cross-sensor coherence & phase, cross-sensor dominant-frequency consistency.

## Univariate PD-vs-ET discrimination (Mann-Whitney)
Strongest at REST — including a genuine *propagation* feature:

| condition | feature | effect | p |
|---|---|---|---|
| REST | logpow_hand | +0.37 | 0.002 |
| REST | **absphase_lower_upper** (cross-sensor phase) | −0.34 | 0.004 |
| OUT | logpow_hand | +0.26 | 0.032 |
| OUT | absphase_hand_lower | +0.25 | 0.035 |

The cross-sensor **phase** is new signal beyond spectral power — PD and ET
propagate tremor up the arm differently.

## Two-stage: TF vs spatial vs TF+spatial (per condition, LOO, tuned ET threshold)

| condition | TF (STFT-256) | spatial | **TF + spatial** |
|---|---|---|---|
| OUT | 0.651 / ET 0.378 | 0.627 / 0.409 | **0.662 / ET 0.421** |
| REST | 0.477 / ET 0.047 | 0.502 / **0.250** | 0.481 / 0.048 |
| WING | 0.669 / ET 0.400 | 0.333 / 0.164 | **0.674** / 0.400 (PD-vs-ET 0.84) |

(values are macro-F1 / ET-F1)

## Findings
1. **TF+spatial on OUT gives ET-F1 0.421** — the best interpretable ET-F1 so far
   (biomarker 0.324, STFT-256 alone 0.378). The spatial axis adds ~0.04 ET-F1.
2. **WING is the strongest condition for the headline** (TF+spatial macro-F1
   0.674, PD-vs-ET accuracy 0.84) — surprising: WING, not OUT, tops macro-F1.
3. **At REST, spatial rescues TF** — spatial-only ET-F1 0.250 vs TF's collapsed
   0.047 (rest tremor too quiet for a good spectrogram, but propagation carries
   signal).

## Recommended config for the interpretable arm
- **Headline 3-class:** TF+spatial on **WING** (macro-F1 0.674).
- **Best ET-F1:** TF+spatial on **OUT** (ET-F1 0.421).
- REST is TF-poor; use spatial there.

Honest note: ET-F1 CIs remain wide (~16 ET subjects) — the gains are real point
improvements but sit within overlapping CIs. Reproduce: the spatial section of
`pdetn/two_stage_comparison.ipynb`.
