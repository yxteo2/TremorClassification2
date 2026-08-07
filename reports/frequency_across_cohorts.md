# Mean and max frequency across the three cohorts

Recomputed after fixing the power bugs below — the fix matters here, because
`mean_freq` is a **power-weighted** average and half the transforms were
weighting by amplitude instead.

Method: Welch PSD (the only Parseval-exact transform), 3–15 Hz, per patient,
lower_arm / wrist sensor throughout.

## Power bugs found and fixed first

| bug | effect | status |
|---|---|---|
| `multitaper`, `cwt`, `hht`, `sst` returned **\|S\|** not **\|S\|²** | doubling the signal doubled, not quadrupled, reported power — so every power-weighted descriptor used amplitude weights | fixed (`square=True`) |
| `ar16` had no innovation-variance gain | AR spectrum was pure shape and **completely scale-invariant** (ratio 1.000) | fixed |
| descriptors summed `P` ignoring bin width | biased mean/median frequency on non-uniform grids (VMD centres, S-transform) | fixed (integrate with bin widths) |

Verified against analytic truth: for `sin(2π·6t) + 0.5·sin(2π·11t)` the
power-weighted mean is `(6 + 0.25·11)/1.25 = 7.00 Hz`. Welch, STFT-256 and
STFT-512 now all return **7.00** exactly.

Absolute calibration still differs per method (integrated power / true power
ranges 0.15–1.4e4), so `total_power` is comparable **within** a method, not
across methods. Shape descriptors are unaffected.

## The three ET cohorts

| cohort | n | max_freq median [IQR] | mean_freq median [IQR] |
|---|---|---|---|
| Data (2015) ET | 15 | 6.45 [5.76–8.30] | 7.25 [6.73–8.14] |
| **NewData (2025) ET** | 6 | **3.32 [3.12–5.27]** | 5.86 [5.56–6.35] |
| PADS ET | 28 | 5.86 [5.42–7.23] | 6.55 [5.93–7.26] |
| Data (2015) PD | 75 | 6.64 [5.37–8.69] | 7.35 [6.44–8.22] |
| PADS PD | 276 | 6.84 [5.66–8.01] | 7.71 [6.97–8.30] |

**The ET cohorts are not interchangeable.** Kruskal-Wallis on `mean_freq`:
H=8.32, **p=0.0156**.

| pair | effect | p |
|---|---|---|
| **Data-2015 ET vs NewData-2025 ET** | **+0.733** | **0.0084** |
| Data-2015 ET vs PADS ET | +0.424 | 0.0241 |
| NewData-2025 ET vs PADS ET | −0.298 | 0.276 |

Your **own two ET cohorts differ more from each other** (+0.733) than either
differs from PADS. Same disease, same group, different device and session.

**NewData needs a data-quality look before it is used further.** Its median
`max_freq` is **3.32 Hz with IQR [3.12–5.27]** — sitting on the 3 Hz band floor.
That is the signature of low-frequency drift or gross limb motion dominating the
band, not tremor. Combined with the earlier finding that NewData survives a
device-identity probe only on gravity-chirality, the 6 subjects should be
inspected individually before being pooled into anything.

## PD vs ET on frequency alone — the key contrast

Same measure, same method, same sensor position:

| cohort | measure | PD | ET | effect | p |
|---|---|---|---|---|---|
| Data 2015 | max_freq | 6.64 | 6.45 | −0.029 | 0.862 |
| Data 2015 | mean_freq | 7.35 | 7.25 | −0.070 | 0.673 |
| **PADS** | max_freq | 6.84 | 5.86 | +0.281 | **0.0143** |
| **PADS** | **mean_freq** | **7.71** | **6.55** | **+0.538** | **<0.0001** |

**In PADS, mean frequency alone separates PD from ET** (effect +0.538,
p<0.0001, 276 vs 28 patients). **In the 2015 data it does not separate at all**
(p=0.67).

This resolves the open question flagged in `reports/tf_information_loss_audit.md`
— why PADS reaches balanced accuracy 0.736 on identical features while the local
cohort sits at chance (0.507). It is not the classifier: the underlying
frequency contrast simply is not present in the 2015 recordings.

It also **narrows a long-standing project claim**. "PD and ET share the same
dominant frequency, confirmed in two independent cohorts" holds for the 2015
data (6.64 vs 6.45 Hz, p=0.86) but is **false for PADS** (6.84 vs 5.86 Hz,
p=0.014; mean 7.71 vs 6.55, p<0.0001). The overlap is a property of the local
cohort, not of the disease pair.

## What this implies

The most likely explanations, in order of how much they would change the plan:

1. **The 2015 recordings do not capture the discriminative frequency contrast**
   that PADS does — measurement chain, sensor, or protocol. If so, no amount of
   modelling or extra ET subjects recovers it, and fixing acquisition is the
   whole game.
2. The 2015 ET cohort is clinically different (milder, mixed, or differently
   adjudicated) from the PADS ET cohort.
3. `StretchHold` elicits tremor more reliably than the local `OUT` task.

**Recommended next check, cheap and decisive between (1) and (2/3):** run the
same PD-vs-ET frequency test on the 2015 `WING` and `REST` conditions. If the
contrast is absent in every local condition but present in PADS, that points at
acquisition rather than task.
