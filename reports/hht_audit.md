# HHT audit: the removed estimator had a binning bug; fixed, it still does not beat multitaper

Run: `python -m experiments.hht_audit` (synthetic + 2015 OUT, ~30 min CPU).

## Why

HHT was removed on 2026-09-28 as "worst of the estimators"
(`closed_families.md`), while an earlier summary had called HHT the best PD-vs-ET
separator. A ranking is only as good as the implementation, so the removed code
was audited. It is **executed from git history** (`signal_processing/tfd.py` at
`57fd8a72^`, one dead import stubbed), not retyped.

## The defect

`apply_hht` binned instantaneous frequency with outer edges of **-inf and +inf**:

```python
freq_edges = np.concatenate([[-np.inf], midpoints, [np.inf]])
valid = (bin_idx >= 0) & (bin_idx < n_f)     # always true
```

So nothing was ever out of band. Energy above 15 Hz -- sensor noise, which EMD
puts in its first IMF -- piled into the 15 Hz bin; energy below 3 Hz
(voluntary movement, drift) into the 3 Hz bin. The old `hht_imf2plus` docstring
had seen the symptom ("at noise sd=0.3 the peak jumps to the top of the band")
and blamed noise.

## Synthetic (median of 10 draws; 3 channels, 15 s)

| signal (true peak) | old HHT | HHT, edges fixed | HHT, full fix | multitaper |
|---|---|---|---|---|
| clean 6 Hz | 6.00 (sharp 49.0) | 6.00 (49.0) | 6.00 (47.3) | 6.25 (7.3) |
| 6 Hz + noise | **15.00** (22.8) | 6.00 (3.8) | 6.00 (4.2) | 6.25 (5.1) |
| jittered + AM 6 Hz | **15.00** (22.2) | 5.75 (3.5) | 5.75 (3.9) | 5.86 (5.0) |
| 6 + 12 Hz harmonic | **15.00** (22.4) | 6.00 (3.4) | 6.00 (3.9) | 6.25 (4.8) |
| 4.5 + 7.5 Hz | **15.00** (18.4) | 4.25 (3.1) | 4.25 (3.0) | 4.49 (4.0) |
| noise only | **15.00** (30.0) | flat (1.4) | flat (1.4) | flat (1.2) |

"Full fix" = finite edges + 1 s mirror padding before the Hilbert transform +
Savitzky-Golay instantaneous frequency. **The edge fix alone recovers every
frequency**; HHT is then more precise than multitaper on a clean tone (6.00 vs
6.25). Under noise multitaper keeps the sharper peak (4.8-5.1 vs 3.9-4.2): EMD
spreads noise-IMF energy across the band.

## 2015 OUT, lower arm (4 descriptors, LR 5-fold x 20 repeats, 200-perm null)

| estimator | PD vs ET | N vs tremor |
|---|---|---|
| old HHT (bugged) | 0.387, p = 0.82 | 0.875, p = 0.005 |
| HHT, edges fixed | 0.399, p = 0.82 | 0.868, p = 0.005 |
| HHT, full fix | 0.386, p = 0.82 | 0.868, p = 0.005 |
| **multitaper** (pipeline) | 0.572, p = 0.23 | **0.891**, p = 0.005 |

* **PD vs ET: no estimator separates them on 2015 OUT** -- all inside the null,
  as every spectral feature there.
* **N vs tremor: the bug barely mattered** (0.875 vs 0.868). The edge spike's
  size relative to the in-band energy is itself a tremor-vs-noise measure: old
  HHT's "peak_sharp" reads AUC 0.25 (inverted -- it measured the 15 Hz noise
  spike), and after the fix the descriptors behave like multitaper's
  (bandwidth 0.11-0.13, peak_sharp 0.72-0.78).
* **Fixed, HHT is still slightly behind multitaper** (0.868 vs 0.891).

## Verdict

The closure stands, for a corrected reason: HHT is not worse because it is
HHT -- the removed implementation was wrong (edge binning) -- but **corrected,
it does not beat multitaper** on 2015 OUT, and it is ~100x slower (EMD per
channel). Its one real advantage, exact frequency on a clean tone, does not
turn into separability here. Do not restore it to the pipeline.

## Prediction (in the docstring) -- failed

It expected the old HHT to have the *bluntest* peak (from the noisy
instantaneous frequency) and the fix to recover it. The actual defect was the
opposite: a *sharp spurious* peak at 15 Hz from edge binning; the
instantaneous-frequency smoothing was worth only +0.4 sharpness. The real-data
half ("no method separates PD from ET") held. Registered as #38.
