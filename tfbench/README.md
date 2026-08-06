# tfbench — signal-processing benchmark + deep architecture comparison

Self-contained. Stage 1 picks the best signal-processing method; Stage 2 tests
whether deep models exploit it better.

## Run

```
jupyter notebook tfbench/01_signal_processing_benchmark.ipynb   # Stage 1
jupyter notebook tfbench/02_deep_model_comparison.ipynb         # Stage 2
```

Stage 1 writes `artifacts/tfbench_top_methods.json`; Stage 2 reads it.

## Files

| file | role |
|---|---|
| `transforms.py` | 12 methods, each -> (freqs, power). Add new ones to `METHODS`. |
| `descriptors.py` | max/mean/median frequency, spread, entropy, Q, ... |
| `benchmark.py` | Stage 1: univariate screen (BH-corrected) + LOSO ranking (paired CI) |
| `deep.py` | Stage 2: patient-grouped CV over method x architecture x seed |

## Methods

`welch`, `stft256`, `stft512`, `multitaper`, `cwt`, `hht`, `hht_imf2plus`,
`sst`, `wavelet_packet`, `stransform`, `vmd`, `ar16`

All verified to recover a synthetic 6 Hz tone (notebook section 2). Note
**plain `hht` is noise-dominated** — EMD puts broadband noise in IMF1, which
takes over the marginal spectrum (peak jumps to the band edge at noise sd 0.3).
`hht_imf2plus` drops IMF1 and recovers 6.00 Hz. Both are kept so the benchmark
shows the difference.

## Architectures (Stage 2)

From `tremor.models.MODELS`: `tremor_bilstm`, `bilstm`, `gru`, `lstm`,
`restcn`, `resbilstm`, `resnet18`, `ast`.

Use `resnet18` or `restcn` if you later want GradCAM — it needs conv feature
maps, which the BiLSTM does not have.

## Reading the output — three rules

1. **PD-vs-ET is reported as balanced accuracy.** The majority baseline is
   0.833 locally (75 PD / 15 ET). Raw accuracy on that axis is meaningless.
2. **Use the BH q column, not raw p.** The univariate grid is ~120 tests; about
   6 raw p<0.05 are expected by chance.
3. **A point estimate without a paired CI is not a comparison.** `rank_methods`
   marks with `*` only methods whose paired bootstrap CI against the reference
   excludes zero.

These are not generic caution — each corresponds to a specific error made
earlier in this project (see `reports/handedness_does_not_survive.md` and
`reports/quaternion_session_verdict.md`).

## Prior expectation

Nine feature families have already been compared on this cohort and all landed
within CI on PD-vs-ET (`reports/signal_processing_summary.md`,
`reports/quaternion_session_verdict.md`). An empirical power curve
(`docs/IMPLEMENTATION_PLAN.md`) suggests PD-vs-ET plateaus near 0.68 balanced
accuracy regardless of cohort size. If Stage 1 finds nothing beating `welch`,
that is the expected outcome and is itself reportable: the information is in the
spectrum, not in the estimator.
