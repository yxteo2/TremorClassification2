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

## Rectification: the magnitude reduction, and why we keep it

Checking the frequency **distributions** exposed that the vector-magnitude
reduction `sqrt(gx²+gy²+gz²)` **rectifies** the signal — squaring a 6 Hz
oscillation puts energy at 12 Hz. Measured on the local data:

| class | peak via magnitude | peak via per-channel PSD | ratio |
|---|---|---|---|
| PD | 7.81 Hz | 6.64 Hz | 1.18 |
| ET | 11.72 Hz | 6.64 Hz | 1.76 |

We tested the principled fix (`mode='pc1'`: project onto the first principal
component of the bandpassed axes, preserving the signed oscillation):

| features | macro-F1 | ET-F1 | PD-vs-ET |
|---|---|---|---|
| spatial magnitude | **0.627** | **0.409** | 0.73 |
| spatial pc1 | 0.512 | 0.254 | 0.47 |
| signal magnitude | **0.529** | **0.294** | 0.74 |
| signal pc1 | 0.384 | 0.200 | 0.28 |
| TF+spatial magnitude | **0.662** | **0.421** | 0.77 |
| TF+spatial pc1 | 0.651 | 0.378 | 0.76 |

**The magnitude wins everywhere.** Two reasons: (1) it is **rotation-invariant**,
whereas PC1 depends on the dominant oscillation axis, which varies with sensor
placement across subjects; (2) it retains **amplitude-modulation** structure
(tremor waxing/bursting) that PC1 discards — visible in the spatial coherence
collapse (PD-vs-ET 0.73 → 0.47), since envelope co-fluctuation across sensors is
a robust propagation measure while PC1 coherence needs the limb segments'
rotation axes to align.

**Consequence for the write-up (naming, not performance):** features derived
from the magnitude in `pdetn/signal_features.py` and `spatial_features.py`
describe the tremor **envelope**, not the raw oscillation — so "dominant
frequency"/"Q-factor" there should be described as envelope measures. The
biomarker frequencies reported in `reports/biomarker.md` are **unaffected**:
`tremor/biomarker.py` computes per-channel PSDs and averages them (no
rectification), so PD ≈ 7 Hz / ET ≈ 6 Hz are correct tremor frequencies.
`mode='pc1'` remains available but the default stays `magnitude`.

## Key distribution finding: the classes overlap in frequency

Plain frequency measures (max / mean / median), PD-vs-ET **distribution overlap**
(1.0 = identical):

| measure | LOCAL | PADS |
|---|---|---|
| max (peak) frequency | 0.67 | 0.66 |
| mean frequency | 0.53 | 0.73 |
| median frequency | 0.62 | 0.73 |

Two-thirds of the PD and ET distributions coincide, **in both datasets
independently**, while N separates strongly (Kruskal-Wallis p = 1e-10 to 1e-16).
Computed without rectification, PD and ET even share the same median dominant
frequency on the lower_arm sensor (6.64 Hz vs 6.64 Hz).

**This explains why no time-frequency method helped:** the classes are not
separated along the frequency axis, so finer frequency *resolution* has nothing
to resolve. It also explains why the spatial/propagation features were the only
ones that improved ET-F1 — they carry information orthogonal to frequency.

## The binding conclusion
Across 8 TF methods, spatial, nonlinear-dynamics, higher-order, and parametric
features — plus parameter sweeps, fusion, selection, and curated sets — the
classifier ceiling is set by the **~16 ET subjects**, not the signal processing.
The exploration yielded a real +0.10 ET-F1 (0.324→0.421) and a rich set of
interpretable biomarkers. Further feature engineering returns within-CI results;
the remaining lever that moves the ceiling is **more ET data** (e.g. PADS,
16→44 subjects — `reports/track3_external_data.md`).
