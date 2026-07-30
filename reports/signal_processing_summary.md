# Signal-processing exploration — consolidated summary

An exhaustive signal-processing study for N/PD/ET separation. All evaluations are
honest (subject/patient-level LOSO, tuned ET threshold, subject bootstrap CIs).

## Classifier progression (ET-F1, interpretable two-stage)

| feature set | macro-F1 | ET-F1 |
|---|---|---|
| biomarker spectral | 0.582 | 0.324 |
| STFT-256 (tuned decomposition) | 0.651 | 0.378 |
| **TF + spatial @ OUT** | **0.662** | **0.421** |
| TF + spatial + nonlinear | 0.623 | 0.324 (overfit) |
| curated compact (12 feat, multi-cond) | 0.549 | 0.263 |
| deep STFT (different eval, ref) | ~0.63 | ~0.47 |

**Best: TF + spatial @ OUT, ET-F1 0.421** — up from 0.324, driven mainly by the
3-sensor spatial/propagation features. WING gives the best macro-F1 (0.674).

## What helped vs what didn't
- **Helped:** tuning the STFT window (128→256); **spatial features** (per-sensor
  power gradients + cross-sensor coherence/phase) — complementary and low-dim.
- **Did not help the classifier (overfit 16 ET):** nonlinear dynamics, higher-
  order spectra, AR, feature fusion, curated compact sets, multi-condition concat.
- **TF methods:** STFT-256 best 3-class, HHT-7/8 best PD-vs-ET separability;
  VMD/SST/S-transform/wavelet did not beat them.

## Biomarkers discovered (great for interpretation, not classifier inputs)
| biomarker | condition | PD-vs-ET | note |
|---|---|---|---|
| Higuchi fractal dimension | REST | −0.52, p<0.001 | PD tremor less complex than ET |
| dominant frequency | REST | −0.41, p<0.001 | PD slower (5.5 vs 6.2 Hz) |
| AR pole frequency | REST | −0.41, p=0.001 | parametric resonance |
| rest-vs-action power contrast | — | p=0.057 | ET action-dominant |
| cross-sensor phase (lower–upper) | REST | −0.34, p=0.004 | tremor propagation differs |
| sample entropy | OUT | −0.33, p=0.006 | PD more regular |

## The binding conclusion
Across 8 TF methods, spatial, nonlinear-dynamics, higher-order, and parametric
features — plus parameter sweeps, fusion, selection, and curated sets — the
classifier ceiling is set by the **~16 ET subjects**, not the signal processing.
The exploration yielded a real +0.10 ET-F1 (0.324→0.421) and a rich set of
interpretable biomarkers. Further feature engineering returns within-CI results;
the remaining lever that moves the ceiling is **more ET data** (e.g. PADS,
16→44 subjects — `reports/track3_external_data.md`).
