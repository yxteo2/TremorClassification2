# Signal-processing audit, 2015 cohort (2026-09-27)

Checked end to end for the current scope (2015, OUT, lower-arm sensor): raw
files → quaternion → angular velocity → multitaper spectrum / descriptors /
IF trajectory → biomarker table. Three defects fixed, one design flaw
documented, two dataset properties measured and found benign or weak.

## Fixed

| issue | effect | fix |
|---|---|---|
| `verify_data` check 13 crashed on Windows (`glob` returns `\`, the `'raw_quaternion/'` split failed) | the unit-norm check never ran here | normalise the path; now 13 checks, 1 expected failure |
| the four `N 2` REST/WING files (accelerometer data, median \|q\| 9.74-9.77) were converted like quaternions — `process_quaternion_data` normalises each row to \|q\| = 1, so they became plausible-looking angular velocity. `01_tremor_characteristics.ipynb` loads 2015 REST without excluding them | one N patient's REST biomarkers were built from accelerometer data. **OUT unaffected** | `load_quaternion_recordings` now skips any file whose median \|q\| is outside (0.95, 1.05), with a warning. Loads OUT 274 / REST 273 / WING 248 — exactly the four files dropped. Re-run notebook 01 to refresh its REST tables |
| `frequency.characteristics.patient_table` defaulted to `ch=slice(0, 3)` — the HAND sensor on 2015 | no current caller relied on it (all pass `ch`), but new code would silently analyse a different sensor from the model | default is now the lower arm, `slice(3, 6)` |

## Documented, not changed

**`hht_imf2plus` is invalid on strong tremor** — the one `verify_preprocessing`
failure (51 checks, 50 pass). It drops EMD's first IMF as noise, but IMF1 is the
highest-frequency oscillation, which *is* the tremor when the tremor dominates:
on a clean 6 Hz tone the peak lands at the 3 Hz edge. Benchmark-only, not used
by the model or the notebooks; the docstring now says so.

## Measured, benign

**Two export formats.** 2015 files store either 6 decimals or full precision,
and the mix is class-associated (OUT: 58 % of PD files are 6-dp vs 38 % N, 41 %
ET). Within class, 6-dp and full-precision files do not differ on in-band tremor
amplitude, peak frequency, 25-50 Hz noise, 0.1-3 Hz drift, duration or step size
(2 of 21 within-class tests at p < 0.05, the chance rate). The 6-dp rounding
noise after differentiation is ~2e-5 rad/s in band, ~750x below the median
tremor (0.015 rad/s). Formatting, not a second pipeline.

**Quaternion → angular velocity** remains correct (central difference with
normalisation and sign-continuity; previously matched an independent SciPy
implementation to 1e-4 relative error). Space- vs body-frame does not matter:
the pipeline averages power over the three axes, which is rotation-invariant.

## Measured, open

**Sensor order.** The 2015 files carry no sensor IDs. Against NewData (known IDs:
hand, lower arm, upper arm), 2015 shows the same adjacency pattern — sensors 0
and 1 agree on peak frequency most (0.66 vs NewData 0.64) and sensor 2 has the
least tremor — but sensors 0 and 1 have near-equal amplitude in 2015 (−1.42 /
−1.39) where NewData's hand is clearly larger (−1.27 / −1.48). Consistent with
hand / lower arm / upper arm, not proof. **Confirm with whoever exported the
2015 data.**

**Recordings per patient are class-confounded.** OUT recordings per patient:
PD 25 of 75 have only one (33 %), N 2 of 61 (3 %), ET 1 of 15 (7 %); total
seconds differ by class (Kruskal p = 0.004). Fewer averaged recordings means a
noisier per-patient spectrum. Within PD alone (no class signal), whether a
patient has one or two recordings is decodable from the model's features at
AUC 0.649 against a null of [0.359, 0.654], p = 0.059 — weak, not significant.
Not fixed. If it matters: equalise averaging (e.g. first recording only for
everyone) and compare in the model with paired repeats.
