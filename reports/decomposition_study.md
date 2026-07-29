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

## Caveat for model testing
HHT's transform is ~200× slower than STFT and the training pipeline recomputes
it per epoch, so an HHT deep stage is impractical on CPU **unless features are
precomputed once**. Because HHT-8-IMF already reaches 0.651 PD-vs-ET with a
plain LDA, a light stage-2 on **precomputed HHT features** is the practical route
(fast, and already validated by the separability number). Reproduce:
`python -m pdetn.decomposition_sweep --data-root Data --action OUT`.
