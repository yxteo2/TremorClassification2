# Instantaneous-frequency stability: the feature family we had ruled out

**Result: a two-stream deep model reading the instantaneous-frequency
trajectory alongside the spectrum reaches ET precision 0.720 and macro
precision 0.675 -- the best measured in this repo.**

## The literature finding, and the belief it overturned

Di Biase et al., *Brain* 2017 ("Tremor stability index: a new tool for
differential diagnosis in tremor syndromes") report that PD and ET patients do
**not** differ in peak power frequency, median power frequency, power
dispersion, harmonic index or relative power contribution -- precisely the
static spectral quantities that every descriptor and every spectrum model in
this repo is built from. What separates them is the stability of the
**instantaneous** frequency: in ET it stays inside a narrow band cycle to cycle,
in Parkinsonian tremor it wanders more widely.

This repo had concluded the opposite of the relevant premise. `small_nets.py`
and several reports state that "tremor is quasi-stationary, so the time axis
carries nothing", on the evidence that a BiLSTM over spectrogram frames sat at
bal-acc 0.513.

**That evidence was answering a narrower question than it was taken to answer.**
A sequence model over spectrogram frames asks whether spectral *shape* evolves.
Whether *instantaneous frequency* is stable is a different measurement, and it
had never been computed here.

Sanity check on real data, computed without labels -- instantaneous-frequency
fluctuation, standard deviation in Hz:

| class | IF fluctuation sd |
|---|---|
| N | 0.528 |
| PD | **0.544** |
| ET | **0.447** |

ET most stable, PD least: the ordering the Tremor Stability Index predicts.

## Single-cohort screening, PD-vs-ET

| features | dim | PADS (28 ET) | 2015 (15 ET) |
|---|---|---|---|
| spectrum (log-binned) | 16 | 0.711 | 0.550 |
| 10 descriptors | 10 | **0.807** | 0.482 |
| **stability (6, new)** | 6 | 0.742 | **0.652** |
| descriptors + stability | 16 | 0.754 | 0.547 |

Univariate AUC on PADS: `if_std` 0.773, `if_iqr` 0.763, `tsi` 0.742,
`autocorr_decay` 0.705 -- single numbers rivalling the whole descriptor set.

**The published claim does not fully replicate.** On PADS the 10 descriptors --
which are exactly the quantities Di Biase et al. report as non-discriminating --
reach AUC 0.807, the best PD-vs-ET figure in this repo. The stability features
win on 2015, where descriptors are *below chance* (0.482). The two families are
each strongest on a different cohort.

## In the merged 3-class model

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| baseline (desc + asym) | 0.653 | 0.655 | 0.639 | 0.649 | 0.600 |
| + stability appended | 0.626 | 0.637 | 0.669 | 0.644 | 0.588 |
| stability **replaces** descriptors | 0.628 | 0.647 | 0.699 | **0.658** | 0.591 |
| **TWO-STREAM (IF trajectory)** | 0.642 | 0.664 | **0.720** | **0.675** | **0.605** |

Paired against baseline, 10 splits:

| config | precET | macroP |
|---|---|---|
| + stability appended | +0.030 [-0.052, +0.118] | -0.005 [-0.039, +0.030] |
| stability replaces desc | +0.060 [-0.033, +0.146] | +0.009 [-0.019, +0.038] |
| two-stream | **+0.081 [-0.007, +0.180]** | +0.026 [-0.008, +0.068] |

None is significant at 10 splits; the two-stream ET-precision interval misses by
0.007, and a 30-split run is under way to resolve it.

Two secondary observations:

* **Replace, do not append.** Appending stability to the descriptors
  significantly *hurts* precN (-0.027 [-0.055, -0.001]); replacing them helps.
  Sixth instance in this session of a feature union underperforming its best
  member at these sample sizes.
* **Variance drops when descriptors are replaced**: precET sd 0.185 -> 0.120,
  the tightest ET-precision estimate measured.

## The deep-learning form

The point of `TrajectoryEncoder` / `TwoStreamNet` is that the finding is fed to
the network rather than bolted on as a scalar. A dilated TCN reads the
``(2, T)`` trajectory of centred instantaneous frequency and relative envelope;
the spectrum branch is unchanged. The encoder uses **mean and std pooling**,
because the discriminative quantity is how much the trajectory varies, which
mean pooling alone would average away.

### A bug caught by a sanity test, not by a metric

`if_trajectory` initially z-normalised the frequency channel, which sets its
variance to 1 and **destroys the fluctuation magnitude -- the exact quantity TSI
measures**. A rock-steady and a wildly wandering 6 Hz tremor both came out with
std 1.000. No accuracy number would have looked obviously wrong. After the fix,
synthetic stable / mild / wandering signals give 0.053 / 0.091 / 0.446 Hz.

Reproduce: `tfbench/stability.py`, `scratch/twostream.py`.
