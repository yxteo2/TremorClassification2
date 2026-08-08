> **SUPERSEDED — see `reports/multicohort_method_ranking.md`.** These numbers were
> computed before the power fix (four transforms returned amplitude, not power, so
> every power-weighted descriptor used the wrong weights) and on 2015 OUT only —
> the cohort/condition where PD-vs-ET sits at chance. Re-run, `hht_imf2plus` falls
> to bal-acc 0.613 with CI [-0.030, +0.247], p=0.067: no method beats Welch there.

# Stage 1 results — which signal-processing method discriminates tremor best?

12 methods, same descriptors, same patients (151: N=61, PD=75, ET=15), OUT
condition, patient-level LOSO. Reproduce: `tfbench/01_signal_processing_benchmark.ipynb`.

## N vs Tremor — the estimator does not matter

| method | bal-acc | paired diff vs welch | 95% CI |
|---|---|---|---|
| welch | **0.843** | — | — |
| ar16 | 0.843 | +0.000 | [−0.042, +0.042] |
| cwt | 0.843 | +0.001 | [−0.039, +0.041] |
| stft512 | 0.840 | −0.002 | [−0.044, +0.039] |
| sst / hht / multitaper / stransform / vmd | 0.826–0.834 | −0.009 … −0.018 | all span 0 |
| stft256 | 0.824 | −0.018 | [−0.056, +0.014] |
| wavelet_packet | 0.823 | −0.020 | [−0.072, +0.031] |
| hht_imf2plus | 0.793 | −0.051 | [−0.118, +0.009] |

**Every CI spans zero.** Plain Welch PSD — the simplest estimator available —
is as good as anything. For this axis the information is in the spectrum, not
in how you estimate it. That is a clean, quotable negative.

## PD vs ET — one method separates from the reference

| method | bal-acc | paired diff vs welch | 95% CI | p(≤0) |
|---|---|---|---|---|
| **hht_imf2plus** | **0.640** | **+0.126** | **[+0.020, +0.250]** | **0.0083** |
| wavelet_packet | 0.620 | +0.107 | [−0.020, +0.239] | 0.0517 |
| vmd | 0.560 | +0.047 | [−0.113, +0.207] | 0.288 |
| hht | 0.553 | +0.040 | [−0.138, +0.215] | 0.330 |
| cwt | 0.533 | +0.019 | [−0.101, +0.143] | 0.390 |
| welch | 0.513 | — | — | — |
| stft256 / stransform | 0.513 | ±0.000 | span 0 | — |
| ar16 | 0.500 | −0.013 | [−0.100, +0.050] | 0.618 |

`hht_imf2plus` — the Hilbert-Huang transform **with IMF1 discarded** — is the
only method whose paired CI against Welch excludes zero.

### But it does not survive correction for having compared 12 methods

| | |
|---|---|
| p(≤0) at 20 000 bootstrap draws | **0.0083** |
| Bonferroni threshold, 11 comparisons | **0.00455** |
| verdict | **fails** |

Worth recording *how* this was nearly missed: at 1 000 bootstrap draws the same
comparison read **p = 0.0040**, which would have passed. The bootstrap
resolution was too coarse and the small p was an artifact of it. Any p being
compared against a corrected threshold needs enough draws to resolve it.

### The univariate screen agrees on the method and finds nothing significant

118 (method × descriptor) tests, BH-corrected:

| method | descriptor | AUC | effect | raw p | **BH q** |
|---|---|---|---|---|---|
| hht_imf2plus | **max_freq** | 0.675 | −0.349 | 0.034 | 1.000 |
| hht_imf2plus | **median_freq** | 0.664 | −0.328 | 0.046 | 1.000 |
| hht_imf2plus | **mean_freq** | 0.643 | −0.285 | 0.083 | 1.000 |

**0 of 118 survive BH q<0.05.** The top three are all `hht_imf2plus` frequency
descriptors — exactly the max/mean/median frequency measures this study is about
— and the multivariate ranking independently picks the same method. Two lines of
evidence agreeing on *which* method is best, and neither reaching significance
on *whether* it is genuinely better.

## Reading

* **N-vs-Tremor is solved well and equally by everything** (~0.84). Use Welch.
* **PD-vs-ET: `hht_imf2plus` is the best method found** (0.640 vs Welch's 0.513,
  which is chance). The effect is real enough to carry into Stage 2 but is
  **not established** — it fails multiplicity correction, and the cohort is 15 ET.
* Plain `hht` (0.553) is much worse than `hht_imf2plus` (0.640). EMD puts
  broadband noise into IMF1, which dominates the marginal spectrum; on synthetic
  data a clean 6 Hz tone is recovered exactly but at noise sd 0.3 the peak jumps
  to the band edge. **Dropping IMF1 is what makes HHT work here** — the single
  most actionable preprocessing finding in this benchmark.
* This is consistent with the project's prior: 9 feature families already landed
  within CI, and a power curve suggests PD-vs-ET plateaus near 0.68 balanced
  accuracy (`docs/IMPLEMENTATION_PLAN.md`). `hht_imf2plus` at 0.640 sits just
  under that plateau.

## Carried to Stage 2

`hht_imf2plus`, `wavelet_packet`, `hht`, `cwt`.

Caveat on the first: `TremorDataset` has no IMF-selection option, so a deep model
trained on "hht" sees the **full** Hilbert spectrum including IMF1. Since
dropping IMF1 is precisely what made the method work, the Stage-2 row is
"HHT-family", not a like-for-like carry-over. Implementing IMF-selected input is
the obvious first extension if Stage 2 shows promise.
