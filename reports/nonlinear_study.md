# Nonlinear-dynamics / higher-order signal features

New signal-processing axes (`pdetn/nonlinear_features.py`): DFA, RQA
determinism, Poincaré SD1/SD2, Higuchi fractal dimension, harmonic bicoherence,
AR pole frequency/bandwidth — on the 3-15 Hz bandpassed hand magnitude.

## Finding 1 — a strong new biomarker (univariate PD-vs-ET)

| condition | feature | effect | p |
|---|---|---|---|
| REST | **Higuchi fractal dimension** | **−0.52** | **<0.001** |
| REST | AR pole frequency | −0.41 | 0.001 |
| REST | Poincaré SD2 | +0.37 | 0.002 |
| REST | Poincaré ratio | −0.37 | 0.002 |

**Higuchi FD at REST is the single strongest PD-vs-ET discriminator found in the
whole study** (|effect| 0.52, p<0.001) — stronger than dominant frequency
(−0.41). Interpretation: **PD tremor is less complex / more regular than ET at
rest.** A clean, novel, interpretable biomarker.

## Finding 2 — but it does not improve the classifier

Two-stage (logreg, tuned ET threshold), LOO, per condition:

| condition | TF+spatial (best) | TF+spatial+nonlinear |
|---|---|---|
| OUT | 0.662 / **ET 0.421** | 0.623 / ET 0.324 |
| REST | 0.481 / ET 0.048 | 0.493 / ET 0.000 |
| WING | 0.674 / ET 0.400 | 0.674 / ET 0.400 |

Adding 11 nonlinear features to the ~375-dim TF+spatial set **overfits the
16-ET cohort** (ET-F1 drops on OUT, unchanged elsewhere) — the same
dimensionality effect seen with the earlier advanced features.

## Conclusion
The nonlinear-dynamics features are **excellent biomarkers** (Higuchi FD,
AR pole, Poincaré — all p≤0.002 at REST, quantifying PD's greater tremor
regularity) but **do not raise the classification ceiling**. Best classifier
remains **TF+spatial @ OUT, ET-F1 0.421**. Consistent with every other
feature-addition: on 16 ET subjects, more features overfit; the cohort is the
binding constraint. These features belong in the paper's biomarker/interpretation
section, not the classifier's default input.
