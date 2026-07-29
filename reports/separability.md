# Time-frequency separability — choose the representation before the model

Model-free ranking of time-frequency conversions by how well they separate
N/PD/ET, computed **before** training any deep model (`pdetn/separability.py`).
Each recording's TF image is reduced to mean+std over time; scored by Fisher
trace ratio, silhouette, and subject-CV LDA macro-F1. Transforms run once per
recording, so HHT is feasible here.

## OUT condition, all methods

| method | Fisher | silhouette | subject-CV LDA F1 |
|---|---|---|---|
| **STFT** | 0.083 | 0.035 | **0.573** |
| **CWT** | **0.146** | **0.056** | 0.544 |
| multitaper | 0.080 | 0.032 | 0.516 |
| HHT | 0.063 | 0.027 | 0.454 |
| wavelet_packet | 0.064 | 0.024 | 0.450 |

STFT wins the generalization proxy; CWT wins raw separability (~2× Fisher).
HHT/wavelet_packet separate worst. **This matches the trained deep-model TFD
sweep** (STFT best, CWT second, HHT/wavelet worst) — the model-free ranking and
the trained ranking agree, validating the separability-first approach.

## Condition × method (3-class and PD-vs-ET)

| cond | method | 3-class LDA F1 | PD-vs-ET LDA F1 | silhouette |
|---|---|---|---|---|
| OUT | **CWT** | 0.544 | **0.627** | 0.056 |
| OUT | STFT | **0.573** | 0.560 | 0.035 |
| OUT | multitaper | 0.516 | 0.571 | 0.032 |
| WING | STFT | 0.498 | 0.579 | 0.031 |
| WING | CWT | 0.484 | 0.542 | 0.054 |
| REST | STFT | 0.474 | 0.481 | 0.001 |
| REST | CWT | 0.432 | 0.458 | 0.003 |

## Findings
1. **OUT is the best condition**, not REST. Reconciles the biomarker result: REST
   separates PD-vs-ET on the *single* dominant-frequency feature, but on the
   *full* representation REST is worst (silhouette ~0 — rest tremor is too quiet
   in N/ET). Postural (OUT) elicits tremor broadly → richer, more separable.
2. **CWT best separates the hard PD-vs-ET axis** (0.627 vs STFT 0.560), while
   **STFT wins 3-class** (0.573). CWT is the representation to prefer when the
   goal is PD-vs-ET specifically.
3. Model-free ranking agrees with the trained-model ranking.

## Implication for model testing
- 3-class headline: **STFT @ OUT** (already the deep model's config; macro-F1
  ~0.63, ET-F1 ~0.47).
- Hard PD-vs-ET axis: **CWT @ OUT** is the best separator; deep CWT reached
  ET-F1 0.545 (thresh) in the TFD sweep — competitive with STFT (0.571).
- Natural next model: a two-stage where stage-2 (PD-vs-ET) uses **CWT**, stage-1
  (N-vs-tremor) uses **STFT**.

Reproduce: `python -c "from pdetn.separability import rank_methods; ..."` or the
saved `artifacts/separability_*.txt`.
