# Learned time-domain models fail where fixed time-domain formulas succeed

**The gap that motivated this.** Every network in this project convolves along
**frequency** — `ResidualTCN` "over the frequency axis", `SpectrumBiLSTM` reading
the spectrum "as a sequence over FREQUENCY", `Spectrum1DCNN` likewise. The only
time-domain stream, `TrajectoryEncoder`, sees an instantaneous-frequency
trajectory already reduced to 64 points. **No model had read the waveform.**

`catch22_waveform_features.md` made that worth closing: six *fixed* temporal
statistics matched ten tuned spectral descriptors on PADS PD-vs-ET (AUC 0.798 vs
0.794) at half the fold variance. Temporal structure demonstrably carries
comparable information.

Two data-processing routes were built and tested on the merged 3-class protocol,
20 splits, everything else identical to the reported model. Arm A reproduces the
reported model exactly (0.639 / 0.655 / 0.685 / 0.660 / 0.593) in both runs.

## Result 1 — raw waveform

`signal_processing/waveform.py`: band-pass 3–15 Hz → principal-axis projection →
decimate 100→40 Hz → z-score → centre-crop 384. `WaveformTCN`, 5,451 parameters,
3.0 s receptive field (~18 cycles of a 6 Hz tremor).

| arm | precN | precPD | precET | macroP | sd(macroP) |
|---|---|---|---|---|---|
| A reported model | 0.639 | 0.655 | **0.685** | **0.660** | 0.068 |
| B waveform TCN alone | 0.631 | 0.630 | 0.617 | 0.626 | **0.051** |
| C soft vote A + B | 0.637 | 0.634 | 0.638 | 0.636 | 0.071 |

paired: B is **macroP −0.034 [−0.066, −0.004] \***; C is −0.024 [−0.056, +0.007],
negative on every column.

## Result 2 — analytic signal (envelope + IF stability)

The hypothesis was that a TCN on the raw waveform wastes capacity learning to
demodulate, so `analytic_channels` hands it the answer: log envelope, plus
instantaneous frequency centred on the patient's own median.

| arm | precN | precPD | precET | macroP | sd(macroP) |
|---|---|---|---|---|---|
| A reported model | 0.639 | 0.655 | **0.685** | **0.660** | 0.068 |
| B analytic TCN alone | 0.654 | 0.604 | 0.493 | 0.584 | **0.044** |
| C soft vote A + B | 0.644 | 0.621 | 0.676 | 0.647 | 0.077 |

paired: B is **precET −0.192 [−0.278, −0.108] \*** and **macroP −0.076
[−0.107, −0.047] \***; C is −0.013 macroP, not significant.

**The demodulation hypothesis is wrong** — the analytic stream is *worse* than the
raw waveform (0.584 vs 0.626), not better.

**A candidate explanation — tested, and REFUTED.** The IF channel was centred on
each patient's own median, to avoid duplicating the absolute frequency the
spectrum stream already carries. Since absolute tremor frequency is the single
most discriminative quantity available (max + mean frequency alone give AUC 0.786
on PADS), the stream looked like it had been deprived of the strongest signal by
construction.

`experiments/analytic_if_control.py` re-ran the identical stream with the IF left
absolute. Envelopes are bit-identical between arms, so frequency is the only
difference, and the absolute IF does carry class structure (N 8.28, PD 7.73,
ET 7.07 Hz). The prediction written down before the run was that restoring it
would recover roughly the 0.042 macroP separating the analytic stream from the
raw waveform.

| paired C (absolute IF) − B (centred IF) | |
|---|---|
| precET | +0.035 [−0.015, +0.081] |
| macroP | **+0.006 [−0.018, +0.029]** |

**The prediction fails.** precET moves in the predicted direction but spans zero,
and macro precision is essentially unchanged. Median-centring was not the cause.

So **both** causal stories offered for this experiment were wrong: demodulation
does not explain why the raw waveform underperforms, and centring does not explain
why the analytic stream underperforms further. What survives is the plain reading
— **the analytic representation is not learnable at this sample size**, and the
ordering among time-domain streams is not explained by what information they
carry.

Two wrong explanations in one experiment is itself the lesson: at this n, a
representation's *content* predicts its performance far less well than whether the
model has to estimate anything from these patients.

## The pattern across both, and what it means

Neither stream helps, but they fail the same way and it is not random:

| stream | macroP | sd(macroP) |
|---|---|---|
| reported (spectral) | **0.660** | 0.068 |
| waveform TCN | 0.626 | 0.051 |
| analytic TCN | 0.584 | **0.044** |
| catch22 state features (PADS, fixed formulas) | — | half of descriptors' |

**Time-domain representations are consistently lower-variance and lower-accuracy.**
They are more stable across folds and worse on average — every one of them,
including the catch22 family.

The sharp version of the finding is the contrast between catch22 and these TCNs.
Both read the same temporal structure from the same band. catch22 **ties** the
spectral descriptors; a learned TCN on the same information is significantly
worse. The difference is that **catch22 does no learning on the time axis** — its
22 formulas were fixed offline on 93 unrelated datasets. The TCN has to estimate
its temporal filters from 404 patients with 49 ET, and it cannot.

So the operative statement is not "time-domain information is absent". It is:

> **The spectrum is a near-sufficient statistic for this problem at this sample
> size, and time-domain information is only reachable through estimators that do
> not have to be learned from this cohort.**

That also explains, retrospectively, why the IF *trajectory* stream in the
reported model works (+0.056 precET) while these do not: it is a 64-point
summary computed in closed form, not a learned representation of a 384-point
sequence.

## One descriptive finding that stands

Before any model, the IF-stability channel orders the classes exactly as Häring's
mechanism predicts, on 404 patients:

| class | IF sd (Hz) |
|---|---|
| N — no coherent oscillator | 2.705 |
| PD — "several central oscillators" | 2.322 |
| **ET — "singular pacemaker"** | **1.946** |

This is descriptive, with no cross-validation, so it is an observation and not a
model claim. But it is the second independent confirmation of that mechanism this
session on a cohort its authors never saw, and the direction is the predicted one.

The synthetic validation is worth recording too: a stable 6 Hz oscillation gives
IF sd 0.381 Hz against 1.082 Hz for a signal alternating between 5 and 7 Hz
states, with indistinguishable envelopes (0.955 vs 0.973) — the channel measures
state switching and not amplitude.

## Processing details worth keeping

* **Principal-axis projection, never the magnitude.** ‖ω(t)‖ is rotation-invariant
  but for a linear oscillation has fundamental **2f** — verified, 11.91 Hz against
  6.05 Hz on a 6 Hz synthetic. The projection is rotation-invariant *and* keeps
  the waveform. Invariance checks come out exact.
* **Crop length chosen so nothing is padded.** 384 samples at 40 Hz fits inside
  every recording (shortest is 400). Padding amount would be a cohort signature —
  PADS is always 1024 samples at 100 Hz, NewData always 1000.
* **Do not standardise waveform inputs per feature.** They are already z-scored
  per recording; standardising per time index across patients is meaningless and
  corrupts the mask columns.

## Standing

Do not re-try learned time-domain models on these cohorts: raw waveform TCN,
analytic-channel TCN, or votes with either. Re-open only with substantially more
patients, or with a time-domain estimator that requires no fitting.
