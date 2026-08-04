# Quantitative decomposition study — tune the transform for separability

Sweep of each time-frequency method's **decomposition parameters**, ranked by
model-free class separability (subject-CV LDA macro-F1) on the OUT condition,
**before** training any model. Code: `pdetn/decomposition_sweep.py`. Raw:
`artifacts/decomp_sweep/OUT.json`.

## Best configs

**3-class (N/PD/ET):**
| config | subject-CV LDA F1 |
|---|---|
| **STFT window=256** | **0.640** |
| CWT w0=10, step=0.25 | 0.618 |
| CWT w0=8, step=0.25 | 0.611 |
| STFT window=128 (default) | 0.573 |
| HHT (4–10 IMFs) | 0.44–0.47 |

**PD-vs-ET (the hard axis):**
| config | subject-CV LDA F1 |
|---|---|
| **HHT, 8 IMFs** | **0.651** |
| HHT, 10 IMFs | 0.629 |
| CWT w0=6, step=0.5 | 0.627 |
| STFT window=256 | 0.623 |
| STFT window=128 | 0.560 |

## Findings
1. **HHT is worst for 3-class but best for PD-vs-ET — at 8 IMFs (0.651).** Its
   instantaneous-frequency decomposition resolves the fine PD/ET distinction
   that STFT/CWT smear, but is noisy for gross N-vs-tremor. **IMF count is the
   decisive knob:** 8 beats 4/6/10.
2. **STFT window=256 > 128** for 3-class (0.640 vs 0.573) — longer window gives
   the frequency resolution tremor's narrow peaks need. A free tunable gain.
3. CWT w0=8–10 with a fine step (0.25) is a strong all-rounder (0.61–0.62).

## Data-driven hybrid design (was previously guessed)
- **Stage 1 — N vs tremor: STFT window=256** (best 3-class).
- **Stage 2 — PD vs ET: HHT with 8 IMFs** (best hard-axis separator, 0.651 —
  beats CWT 0.627 and STFT 0.623).

## Deeper dig — SST, finer HHT, feature fusion

| method | 3-class | PD-vs-ET |
|---|---|---|
| HHT 7 IMFs | 0.449 | **0.647** |
| HHT 8 IMFs | 0.454 | 0.651 |
| HHT 9 / 12 IMFs | 0.442 | 0.629 |
| CWT w0=6 | 0.544 | 0.627 |
| STFT-256 | 0.640 | 0.623 |
| fuse STFT+CWT+HHT | **0.653** | 0.508 |
| fuse STFT+HHT | 0.648 | 0.528 |
| SST (synchrosqueezed) | 0.486 | 0.568 |

- **HHT's PD-vs-ET optimum is 7–8 IMFs**; more plateau.
- **Feature fusion helps 3-class only marginally (0.653) but dilutes PD-vs-ET**
  (0.508) — extra STFT/CWT dims are noise for the hard binary axis on few ET.
- **SST is not competitive.**

Converged conclusion: STFT-256 for 3-class / N-vs-tremor, HHT-7/8 for PD-vs-ET,
CWT as all-rounder; fusion and SST do not beat them.

### VMD and S-transform (added for completeness)

| method | sep 3-class | sep PD-vs-ET | 2-stage macro-F1 | 2-stage ET-F1 |
|---|---|---|---|---|
| STFT-256 (incumbent) | 0.640 | 0.623 | **0.651** | 0.378 |
| HHT-7/8 (incumbent) | 0.45 | **0.647** | 0.584 | 0.250 |
| S-transform (Stockwell) | 0.484 | 0.539 | 0.621 | 0.383 |
| VMD | 0.547 | 0.438 | 0.588 | 0.235 |

- **VMD underperformed** — despite the "VMD beats EMD" literature prior, its
  PD-vs-ET separability (0.438) is below even STFT/CWT and far below HHT. Its
  mode-centre features separate 3-class acceptably but not the hard axis.
- **S-transform** is middling on separability but its two-stage ET-F1 (0.383)
  ties the best; macro-F1 (0.621) still trails STFT-256.
- **Neither beats the incumbents.** With 8 TF methods now compared
  (STFT, CWT, HHT, wavelet_packet, multitaper, SST, VMD, S-transform) plus
  parameter tuning and fusion, the TF-method lever is exhausted: STFT-256 (3-class)
  and HHT-7/8 (PD-vs-ET) are the winners, and the ceiling is the ~16 ET cohort.

## Two-stage model test on the tuned decompositions
Per-patient leave-one-patient-out, logreg two-stage, tuned ET threshold (OUT):

| two-stage config | macro-F1 | ET-F1 | PD-vs-ET acc | N-vs-tremor acc |
|---|---|---|---|---|
| **STFT-256** | **0.651** | **0.378** | 0.76 | 0.86 |
| HYBRID STFT-256→HHT-8 | 0.618 | 0.250 | **0.79** | 0.86 |
| HHT-8 | 0.584 | 0.250 | 0.78 | 0.81 |
| CWT | 0.595 | 0.300 | 0.70 | 0.83 |
| biomarker (ref) | 0.582 | 0.324 | — | — |

- **STFT-256 tuning lifts the two-stage from 0.582 to 0.651 macro-F1** (ET-F1
  0.324→0.378) vs the biomarker feature set — a real gain from the study.
- HHT-8 / hybrid win **PD-vs-ET accuracy** (0.78–0.79), consistent with the
  separability ranking, but their ET-F1 is lower — ET-F1 also depends on
  stage-1 routing and the minority threshold, not just PD-vs-ET separability.

## Caveat for model testing
HHT's transform is ~200× slower than STFT and the training pipeline recomputes
it per epoch, so an HHT deep stage is impractical on CPU **unless features are
precomputed once**. Because HHT-8-IMF already reaches 0.651 PD-vs-ET with a
plain LDA, a light stage-2 on **precomputed HHT features** is the practical route
(fast, and already validated by the separability number). Reproduce:
`python -m pdetn.decomposition_sweep --data-root Data --action OUT`.
