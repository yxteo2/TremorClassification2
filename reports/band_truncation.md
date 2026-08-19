# The merged path was discarding 21 % of the frequency band

**What was wrong.** `common.cohorts.logbin` reduced a spectrum to `nb` bins by
reshaping:

```python
n = X.shape[1] // nb * nb
return L[:, :n].reshape(len(L), nb, -1).mean(2)
```

Whether that loses anything depends entirely on the width of what it is fed, and
this repo feeds it two different widths:

| producer | columns | `nb=16` keeps | band actually seen |
|---|---|---|---|
| `frequency.tables.spectrum_table` (welch, band-masked) | **61** | 48 | 3.12–12.30 Hz |
| `experiments.final_model.method_table` (interp onto `GRID`) | **64** | 64 | 3.00–15.00 Hz |

64 // 16 * 16 = 64, so **the multitaper path was always exact** and no result
built on it is affected. 61 // 16 * 16 = 48, so the welch path silently dropped
**12.50–14.84 Hz — 21 % of the band**. That path feeds `common.cohorts.load_all`
(the merged table) and the unlabelled corpus in `experiments.masked_pretrain`.

Tremor fundamentals are 4–12 Hz and were always inside the retained range. What
sat in the discarded region is the **second harmonic of a 6.3–7.4 Hz tremor**,
and harmonic structure is the strongest of the four physics families on PADS
(0.736, `four_families.md`). So the region was not empty.

**The fix** slices on rounded edges so every input column lands in some bin. It
is bit-identical to the old behaviour whenever `nb` divides the width (verified),
so nothing on the multitaper path moves.

Run: `python -m experiments.binning`. Logistic regression, 20 repeats, threshold
at the prevalence quantile so precision = recall per class.

## What the coverage is worth

`welch interp` vs `welch reshape` — **identical dimensionality (16), identical
classifier, identical folds**; the only difference is whether the top of the band
is represented.

| cohort / axis | precET | macroP | AUC |
|---|---|---|---|
| PADS PD-vs-ET | **+0.032** [+0.014, +0.050] * | **+0.018** [+0.008, +0.028] * | −0.016 * |
| PADS N-vs-Tremor | **+0.027** [+0.023, +0.031] * | **+0.067** [+0.057, +0.076] * | +0.053 * |
| in-house PD-vs-ET | +0.024 [−0.002, +0.052] | +0.014 [−0.001, +0.032] | +0.036 * |
| in-house N-vs-Tremor | **+0.029** [+0.024, +0.035] * | **+0.034** [+0.028, +0.041] * | +0.032 * |
| MERGED PD-vs-ET | **+0.035** [+0.021, +0.048] * | **+0.020** [+0.012, +0.027] * | −0.012 * |
| MERGED N-vs-Tremor | **+0.025** [+0.023, +0.027] * | **+0.044** [+0.041, +0.048] * | +0.030 * |

**Positive in all six cells, significant in five.** The one that misses is
in-house PD-vs-ET, where 21 ET patients is the binding constraint as usual.

On PD-vs-ET the AUC moves slightly the *wrong* way while precision moves the
right way. That is not a contradiction: AUC is threshold-free and averages over
operating points nobody would use at 9 % prevalence, while precET here is read at
the prevalence quantile. The clinically relevant end of the curve improves.

## A correction to "multitaper beats welch"

The repo ranked transform choice as the 4th-largest contributor, multitaper over
welch. Those two arms were binned from 64 and 61 columns, so **band coverage was
confounded with the estimator**. Separating them:

| | PADS PD-vs-ET AUC | PADS N-vs-T AUC | in-house PD-vs-ET AUC |
|---|---|---|---|
| welch, truncated | 0.752 | 0.727 | 0.558 |
| welch, full band | 0.736 | 0.780 | **0.594** |
| multitaper | **0.792** | **0.797** | 0.537 |

* **On N-vs-Tremor the gap was almost entirely band coverage.** welch full-band
  0.780 against multitaper 0.797 on PADS, 0.829 against 0.833 in-house,
  0.780 against 0.782 merged. The estimator is worth very little there.
* **On PD-vs-ET the estimator does matter, and it splits by cohort** — multitaper
  is clearly better on PADS (precET 0.391 vs 0.339) and significantly *worse*
  in-house (precET 0.169 vs 0.250, paired −0.057 [−0.086, −0.029]). This is the
  same cohort inversion already documented for feature families in
  `own_data_reality_check.md`, now showing up in the choice of spectral
  estimator.

So "multitaper over welch" should be stated as a PADS result, not a general one.

## Coarse binning replicates

`welch raw 61` — no reduction at all — is *worse* than 16 bins on 4 of 6 cells
(PADS PD-vs-ET precET −0.037 *, MERGED −0.037 *). This independently replicates
the standing finding that 61 bins at n=404 is 15 % of the sample count and every
model collapses there. Coarse binning is doing real work; the bug was only ever
about *which* frequencies the coarse bins covered.

## Scope

* Affects `common.cohorts.load_all` and `experiments.masked_pretrain`. Every
  merged-table number computed before this fix was produced on 3.12–12.30 Hz.
* Does **not** affect `experiments/final_model.py`, `pd_vs_et.py`,
  `own_data_10et.py`, `inhouse_axes.py` or `trajectory_tuning.py`, all of which
  bin `method_table` output at 64 columns where the old and new code agree
  exactly.
## The 3-class merged deep model does not convert it — but cannot resolve it either

`python -m experiments.binning_deep` runs both binnings through the merged
3-class model on **20 shared splits**, so split variance cancels:

| binning | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| truncating (3.12–12.30 Hz) | 0.661 | 0.633 | 0.587 | 0.627 | 0.585 |
| full band (3.12–14.84 Hz) | 0.667 | 0.622 | 0.611 | 0.633 | 0.572 |

paired: precET **+0.023 [−0.056, +0.103]**, macroP +0.006 [−0.026, +0.039].
Nothing significant.

The precET point estimate (+0.023) is the same direction and roughly the same
size as the logistic-regression gain (+0.035 on the same merged patients), so
this is **not evidence the deep model fails to use the band** — the interval is
equally consistent with the logreg-sized effect and with zero. It is a power
statement: precET has sd 0.154 across splits here, and the paired differences are
barely correlated between arms, so 20 splits cannot resolve a 0.03 effect. The
merged 3-class protocol is simply the wrong instrument for measuring a change
this size.

Where the fix *is* confirmable is the binary axes under logistic regression,
which is where it is reported above.

## Scope of the measurement

* Confirmed with logistic regression on six cohort × axis cells.
* Not confirmable, and not refuted, on the 3-class merged deep model.
* The earlier unpaired before/after of `python -m common.cohorts` (macroP 0.649 →
  0.642) should not be read as a regression: two unpaired 10-split runs at
  sd 0.064 cannot distinguish those.
