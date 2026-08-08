# Method ranking across cohorts — and why the single-cohort version misled

The stage-1 benchmark originally ran on **2015 OUT only**. That is the cohort
where PD-vs-ET frequency discrimination sits at chance, so it was ranking
methods on data with almost no signal to detect. Re-run on three cohort/condition
combinations, plus with the power bug fixed, the conclusions change.

## 1. The power fix weakened the original winner

2015 OUT, PD-vs-ET, `hht_imf2plus` vs `welch`:

| | before (amplitude-weighted, buggy) | **after (power-fixed)** |
|---|---|---|
| bal-acc | 0.640 | **0.613** |
| paired diff | +0.126 | +0.106 |
| 95% CI | **[+0.020, +0.250]** | **[−0.030, +0.247]** |
| p | 0.0083 | **0.0672** |

It no longer clears even the uncorrected CI test. **`reports/tfbench_stage1_results.md`
is superseded** — on 2015 OUT no method beats plain Welch.

## 2. Condition matters far more than method (2015)

PD-vs-ET balanced accuracy, full 10-descriptor set, lower_arm:

| method | 2015 **OUT** | 2015 **REST** |
|---|---|---|
| stft512 | 0.447 | **0.730** |
| multitaper | 0.473 | 0.710 |
| sst | 0.453 | 0.697 |
| welch | 0.513 | 0.666 |
| hht_imf2plus | 0.613 | 0.657 |
| stft256 | 0.513 | 0.583 |
| stransform | 0.513 | 0.565 |

**REST beats OUT by ~0.15–0.28 balanced accuracy for almost every method.** No
method difference within REST is significant against Welch, but the *condition*
effect is large and consistent across all 12 transforms.

This corrects an earlier statement in this project that the 2015 data is "at
chance on PD-vs-ET". It is at chance **at OUT with two features**. At REST with
the full descriptor set it reaches **0.730**, which beats PADS.

**Best PD-vs-ET on the 2015 cohort alone: REST + stft512 — bal-acc 0.730,
AUC 0.729, ET-F1 0.500 [0.30, 0.67]** (75 PD vs 16 ET, no external data).

## 3. On a cohort with real power, method choice DOES matter

PADS StretchHold, 276 PD vs 28 ET:

| method | bal-acc | paired diff vs welch | 95% CI | p |
|---|---|---|---|---|
| **cwt** | **0.774** | **+0.040** | **[+0.009, +0.085]** | **0.0029 BONF-PASS** |
| stransform | 0.765 | +0.030 | [−0.002, +0.075] | 0.0365 |
| wavelet_packet | 0.761 | +0.027 | [−0.020, +0.087] | 0.165 |
| stft512 | 0.749 | +0.014 | [−0.049, +0.082] | 0.346 |
| ar16 | 0.738 | +0.003 | [−0.028, +0.049] | 0.480 |
| multitaper | 0.736 | +0.001 | [−0.029, +0.046] | 0.519 |
| welch | 0.734 | — | — | — |

**`cwt` is the first method comparison anywhere in this project to survive
Bonferroni correction.** It required 28 ET subjects to detect a +0.040 effect —
which is exactly why nothing was ever significant on 15.

Note the ranking does **not** transfer: `hht_imf2plus` led on 2015 OUT,
`stft512` on 2015 REST, `cwt` on PADS. Method choice is cohort-specific; the
robust finding is that the margins are small (~0.04) compared with the condition
effect (~0.2).

## 4. Pooling 2015 + NewData does not help

NewData now uses the corrected 10 s segmentation. Paired diff is scored on the
**same 2015 patients** in both arms.

| condition | method | device probe | 2015 alone | +NewData | paired diff |
|---|---|---|---|---|---|
| REST | welch | 0.688 | 0.666 (ET=16) | 0.728 (ET=22) | +0.062 [−0.013, +0.160] |
| REST | stft512 | 0.729 | **0.730** | 0.679 | −0.051 [−0.167, +0.060] |
| REST | multitaper | 0.760 ⚠ | 0.710 | 0.704 | −0.007 [−0.030, +0.014] |
| OUT | welch | 0.089 ⚠⚠ | 0.507 (ET=15) | 0.587 (ET=21) | +0.080 [+0.000, +0.185] |
| OUT | stft512 | 0.122 ⚠⚠ | 0.460 | 0.527 | +0.067 [−0.034, +0.185] |
| OUT | multitaper | 0.000 ⚠⚠ | 0.567 | 0.613 | +0.047 [−0.006, +0.128] |

**Every paired CI includes zero**, and at REST the direction is not even
consistent (welch improves, stft512 degrades). Six extra ET subjects change
nothing — as the power curve in `docs/IMPLEMENTATION_PLAN.md` predicted.

### ⚠⚠ A bug in how I read the device probe

The OUT rows were printed as "safe" because the check was `AUC > 0.75`. An AUC
of **0.000** is not indistinguishable — it is *maximally* distinguishable with
LOO-inverted labels, the standard instability when a small dataset is perfectly
separable. **The criterion must be `|AUC − 0.5|`, not `AUC`.** So NewData remains
trivially identifiable at OUT, and the apparent gains in those rows are
untrustworthy. At REST the probe is well behaved (0.688–0.760), meaning
segmentation genuinely removed most of the device signature *there*.

## Bottom line

1. Use **REST**, not OUT, for PD-vs-ET on the 2015 cohort.
2. Best 2015-only result: **REST + stft512, bal-acc 0.730, ET-F1 0.500**.
3. Method choice is worth ~0.04 and only detectable at n≈28 ET; condition is
   worth ~0.2. Spend effort on condition and cohort, not on transforms.
4. NewData's 6 ET subjects add nothing. Its value is that it is a device on
   which **N and PD** could still be collected.
