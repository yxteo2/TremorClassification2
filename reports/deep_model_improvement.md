# Improving CNN / TCN / BiLSTM on the merged cohort

**Result: LOCO macro F1 0.435 -> 0.505 and ET precision 0.226 -> 0.269. This is
the first verified case in this repo of deep models beating logistic regression
on held-out cohorts.**

The lever was **input representation**, not architecture, capacity, or training
tricks.

## Setup

Merged 2015 + NewData + PADS, postural task alignment, PADS capped at 90/class
(the `merge_design.md` optimum), n=404 with 49 ET. Class weights on. Scored both
pooled 5-fold and **leave-one-cohort-out**, over 5 independent capping draws.

## What worked

### 1. Log-scale the spectrum

A sum-normalised tremor spectrum is extremely peaked: one or two bins carry
almost all the mass. After standardisation the network spends its capacity
fitting the near-zero tail. `log(x + 1e-8)` compresses the dynamic range.

### 2. Coarse binning -- the big one

61 bins over 3-15 Hz is ~0.2 Hz resolution against a tremor peak 1-2 Hz wide,
and 61 input dimensions at n=404 is 15 % of the sample count.

LOCO macro F1 by bin count, no mixup:

| bins | 12 | 16 | 24 | 32 | 61 |
|---|---|---|---|---|---|
| CNN | 0.459 | 0.499 | 0.481 | **0.505** | 0.473 |
| TCN | 0.489 | **0.505** | 0.453 | 0.481 | 0.412 |
| BiLSTM | 0.401 | 0.452 | 0.465 | **0.484** | 0.389 |

Every model has an interior optimum and every model collapses at 61.

### 3. Cosine LR decay

The previous loop was 200 fixed full-batch steps with no schedule. Adding cosine
decay lifted the CNN's LOCO from 0.425 to 0.460 on identical input, before any
representation change.

## Verified table (5 capping draws)

| config | pooled F1 | pooled pET | **LOCO F1** | **LOCO pET** |
|---|---|---|---|---|
| logreg raw (61) | 0.472 +/- 0.022 | 0.221 +/- 0.020 | 0.446 +/- 0.018 | 0.194 +/- 0.019 |
| logreg 16-bin | 0.507 +/- 0.016 | 0.256 +/- 0.030 | 0.458 +/- 0.012 | 0.152 +/- 0.009 |
| MLPHead 16-bin | 0.549 +/- 0.021 | 0.337 +/- 0.044 | 0.454 +/- 0.008 | 0.167 +/- 0.011 |
| CNN 16-bin | 0.543 +/- 0.006 | 0.328 +/- 0.022 | 0.494 +/- 0.013 | 0.230 +/- 0.044 |
| **CNN 32-bin** | 0.560 +/- 0.016 | 0.352 +/- 0.033 | **0.505 +/- 0.009** | 0.238 +/- 0.029 |
| **TCN 16-bin** | 0.537 +/- 0.012 | 0.365 +/- 0.022 | 0.500 +/- 0.009 | **0.269 +/- 0.017** |
| TCN 32-bin | 0.550 +/- 0.008 | 0.367 +/- 0.010 | 0.490 +/- 0.007 | 0.249 +/- 0.015 |
| BiLSTM 32-bin | 0.543 +/- 0.013 | 0.339 +/- 0.032 | 0.490 +/- 0.014 | 0.219 +/- 0.043 |
| BiLSTM 24-bin | 0.496 +/- 0.026 | 0.289 +/- 0.037 | 0.436 +/- 0.022 | 0.154 +/- 0.018 |

CNN 32-bin beats logreg raw by 0.059 at sd 0.009/0.018 -- 3-4 sd, not a lucky
split. Best ET precision is TCN 16-bin at 0.269 +/- 0.017 against 0.194 +/- 0.019.

Before this work, under LOCO **no** deep model beat logistic regression
(`three_cohort_deep.md`: logreg 0.435, CNN 0.425, TCN 0.396, BiLSTM 0.366).

## What did not work

### mixup HURTS

Lower LOCO macro F1 in 7 of 9 architecture x bin combinations:

| config | mixup 0.0 | 0.2 | 0.4 |
|---|---|---|---|
| TCN 16-bin | **0.505** | 0.479 | 0.467 |
| CNN 32-bin | **0.505** | 0.482 | 0.473 |
| BiLSTM 32-bin | **0.484** | 0.471 | 0.473 |

This was predicted to help LOCO specifically, on the reasoning that the
cross-cohort gap was a decision-boundary sharpness problem. It helps neither
pooled nor held-out, so **that explanation is withdrawn**. Whatever separates
cohorts is not something convex interpolation smooths over.

## The diagnostic detail

Binning helps the networks but **hurts logreg's ET precision** (0.194 raw ->
0.152 at 16-bin), while helping its macro F1 only slightly.

If coarse bins were simply denoising the spectrum, a linear model would benefit
too. It does not. That supports the specific claim that the constraint was
**input dimensionality relative to sample count for models that must learn a
representation** -- a linear model with 61 coefficients was never the thing
straining. It also explains, in one mechanism, why 1 k-parameter models beat
35 k ones here and why large backbones fail outright.

## Caveats

* ET precision is still 0.27. This is a separability result, not a deployable
  classifier.
* All gains push the same direction: smaller input, smaller model. The remaining
  headroom in architecture work is therefore small. The two levers with real
  upside remain more ET patients and bilateral recording -- four asymmetry
  features reach AUC 0.730 on PD-vs-ET with no network at all
  (`limb_asymmetry_pd_vs_et.md`).

Reproduce: `scratch/improve.py`, `scratch/improve2.py`, `scratch/improve3.py`
(gitignored; models in `tfbench.small_nets`).

## Round 3: architecture fixes, fusion, ensembling

5 capping draws, 16 log-bins, cap 90/class.

| config | pooled F1 | LOCO F1 | LOCO pET |
|---|---|---|---|
| **FUSION cnn+desc** | 0.557 +/- 0.008 | **0.515 +/- 0.004** | 0.287 +/- 0.026 |
| **ResidualTCN** | 0.571 +/- 0.012 | 0.510 +/- 0.007 | **0.352 +/- 0.098** |
| ENSEMBLE CNN+TCN+BiLSTM | 0.545 +/- 0.015 | 0.509 +/- 0.009 | 0.259 +/- 0.027 |
| Spectrum1DCNN 32-bin | 0.553 +/- 0.012 | 0.506 +/- 0.008 | 0.227 +/- 0.023 |
| FUSION rtcn+desc | 0.570 +/- 0.020 | 0.501 +/- 0.007 | 0.249 +/- 0.019 |
| SpectrumTCN 16-bin | 0.537 +/- 0.012 | 0.500 +/- 0.009 | 0.269 +/- 0.017 |
| ENSEMBLE + logreg | 0.543 +/- 0.012 | 0.496 +/- 0.010 | 0.215 +/- 0.034 |
| FUSION bilstm+desc | 0.527 +/- 0.017 | 0.482 +/- 0.007 | 0.202 +/- 0.013 |
| logreg desc only | 0.555 +/- 0.017 | 0.480 +/- 0.008 | 0.170 +/- 0.005 |
| AttnPoolBiLSTM | 0.506 +/- 0.020 | 0.468 +/- 0.010 | 0.161 +/- 0.008 |
| SpectrumBiLSTM | 0.492 +/- 0.028 | 0.456 +/- 0.007 | 0.166 +/- 0.017 |

### Residual connections were a real omission

`SpectrumTCN` had no residual connections -- a dilated conv stack, not a TCN.
Adding them (`ResidualTCN`) took LOCO F1 0.500 -> 0.510 and ET precision
0.269 -> 0.352. The precision figure carries sd 0.098, far wider than anything
else in the table, so it is unstable and should not be quoted without more
draws.

### Descriptors still hold information the network does not extract

`DescriptorFusion(CNN)` reaches LOCO F1 0.515 +/- 0.004 -- the best measured,
with the tightest error bar -- against 0.494 for the CNN alone and 0.480 for
logistic regression on descriptors alone. At 404 patients the network is still
not recovering, from the raw spectrum, what peak location / bandwidth /
harmonic descriptors state explicitly.

### The two winners do not combine

`FUSION rtcn+desc` (0.501) is worse than ResidualTCN alone (0.510) and worse
than `FUSION cnn+desc` (0.515). Fusion helps the CNN (+0.021) and the BiLSTM
(+0.026) but hurts the ResidualTCN (-0.009).

Read together: the ResidualTCN's advantage **is** better feature extraction
rather than access to different information, so bolting descriptors onto it
adds parameters without adding signal. The plain CNN has extraction capacity to
spare and benefits.

### Attention pooling: small but real

`AttnPoolBiLSTM` 0.468 +/- 0.010 against `SpectrumBiLSTM` 0.456 +/- 0.007.
Mean-pooling over frequency weights the 3 Hz bin as heavily as the tremor peak;
letting the model choose is worth ~0.012. BiLSTM remains the weakest family.

## ResNet18 revisited on the merged cohort

The earlier verdict came from 58 patients with RANDOM weights. Re-tested from
scratch at n=404, 3 capping draws (ImageNet weights remain proxy-blocked, so
transfer learning is still untested):

| model | input | params | pooled F1 | LOCO F1 | LOCO pET |
|---|---|---|---|---|---|
| ResNet18 scratch | 2-D spectrogram | 11 172 k | 0.552 +/- 0.010 | 0.466 +/- 0.021 | 0.287 +/- 0.039 |
| Small2DCNN | 2-D spectrogram | 1 k | 0.575 +/- 0.011 | 0.432 +/- 0.009 | 0.147 +/- 0.009 |
| SpectrumTCN 16-bin | 1-D spectrum | 3 k | 0.527 +/- 0.015 | 0.472 +/- 0.006 | 0.229 +/- 0.011 |
| Spectrum1DCNN 32-bin | 1-D spectrum | 1 k | 0.553 +/- 0.012 | **0.506 +/- 0.008** | 0.227 +/- 0.023 |

**ResNet trades macro F1 for ET precision.** Its LOCO F1 (0.466) is clearly
below the 1 k 1-D CNN's (0.506), but its ET precision (0.287) is the higher of
the two. A first single-draw run put that precision at 0.341; over 3 draws it
regressed to 0.287 +/- 0.039, so part of the initial figure was draw luck.

`Small2DCNN` gets the identical 2-D input and collapses (0.432 / 0.147), so the
effect is not "2-D input is better" -- it is specific to the depth / residual /
BatchNorm stack with minibatch training. That is the one result in this session
that cuts against the smaller-is-better trend, and it is worth stating plainly
rather than smoothing over.

## Session summary

LOCO macro F1 **0.435 -> 0.515**, ET precision **0.226 -> 0.287** (or 0.352
unstable). Ranked by contribution:

1. coarse binning of the spectrum (61 -> 16/32)
2. descriptor fusion (+0.021 over the CNN alone)
3. residual connections in the TCN (+0.010, and +0.083 ET precision)
4. log-scaling
5. cosine LR decay
6. attention pooling for the BiLSTM (+0.012)

Refuted along the way: mixup, and the decision-boundary explanation of the
cross-cohort gap.

## Mixed-cohort protocol: all three sources in train, validation and test

A different question from LOCO, and an easier one. Splits are stratified jointly
on **cohort x class**, so 2015, NewData and PADS all appear in train, val and
test. LOCO answers "will this transfer to a new clinic?"; this answers "how well
does it do at sites it was trained on?" Both belong in a paper; only LOCO
supports a generalisation claim.

Also changed here: the validation split does real work. Training keeps the
parameters from the epoch of lowest validation loss instead of running a fixed
200 epochs, so these numbers are not directly comparable to earlier rounds even
setting the protocol aside.

Cohort x class composition (patients, PADS capped at 90/class):

| | N | PD | ET | total |
|---|---|---|---|---|
| 2015 | 61 | 75 | 15 | 151 |
| NewData | 27 | 23 | 6 | 56 |
| PADS | 79 | 90 | 28 | 197 |
| **TOTAL** | 167 | 188 | 49 | 404 |

### Test-set precision, 10 random splits

| model | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| logreg spectrum | 0.603 +/- 0.056 | 0.655 +/- 0.057 | 0.345 +/- 0.078 | 0.535 +/- 0.052 | 0.532 +/- 0.049 |
| logreg descriptors | 0.666 +/- 0.041 | 0.689 +/- 0.062 | 0.343 +/- 0.082 | 0.566 +/- 0.042 | 0.559 +/- 0.047 |
| Spectrum1DCNN 32 | 0.621 +/- 0.043 | 0.646 +/- 0.057 | 0.376 +/- 0.128 | 0.548 +/- 0.059 | 0.547 +/- 0.064 |
| SpectrumTCN 16 | 0.579 +/- 0.030 | 0.696 +/- 0.075 | 0.407 +/- 0.120 | 0.561 +/- 0.050 | 0.530 +/- 0.054 |
| ResidualTCN 16 | 0.579 +/- 0.024 | 0.686 +/- 0.066 | **0.496 +/- 0.154** | **0.587 +/- 0.070** | 0.561 +/- 0.064 |
| SpectrumBiLSTM 32 | 0.618 +/- 0.057 | 0.641 +/- 0.054 | 0.398 +/- 0.098 | 0.552 +/- 0.048 | 0.555 +/- 0.049 |
| FUSION cnn+desc | 0.620 +/- 0.044 | 0.627 +/- 0.053 | 0.426 +/- 0.105 | 0.558 +/- 0.045 | 0.559 +/- 0.044 |
| **ENS fusion+rtcn** | 0.606 +/- 0.048 | 0.667 +/- 0.064 | 0.475 +/- 0.113 | 0.583 +/- 0.046 | **0.576 +/- 0.041** |

`ResidualTCN` has the higher mean macro precision but carries sd 0.070 against
the ensemble's 0.046, and sd 0.154 on ET precision against 0.113. The two are
indistinguishable on the mean; the ensemble is the safer choice.

**Ensembling helps here but did not under LOCO** (0.576 against 0.559 / 0.561
for its members; under LOCO it matched but never beat them).

ET precision roughly doubles versus LOCO (0.287 -> ~0.48). That is the protocol,
not the model.

### Per-cohort precision within the mixed test set

| model | cohort | precN | precPD | precET | macroP |
|---|---|---|---|---|---|
| logreg spectrum | 2015 | 0.612 | 0.694 | 0.276 | 0.527 |
| logreg spectrum | NewData | 0.685 | 0.615 | *0.100* | 0.467 |
| logreg spectrum | PADS | 0.585 | 0.654 | 0.433 | 0.557 |
| FUSION cnn+desc | 2015 | 0.676 | 0.762 | 0.376 | 0.605 |
| FUSION cnn+desc | NewData | 0.627 | 0.610 | *0.100* | 0.446 |
| FUSION cnn+desc | PADS | 0.574 | 0.545 | 0.486 | 0.535 |
| ENS fusion+rtcn | 2015 | 0.650 | 0.743 | 0.447 | 0.613 |
| ENS fusion+rtcn | NewData | 0.639 | 0.662 | *0.100* | 0.467 |
| ENS fusion+rtcn | PADS | 0.573 | 0.610 | 0.524 | 0.569 |

**The NewData ET column is an artifact -- do not report it.** NewData has 6 ET
patients, so a 20 % test split holds ~1.2 of them. An identical 0.100 across
three structurally different models is the signature of one patient, not a model
property. The meaningful ET figures are 2015 (0.447) and PADS (0.524).

This is the same conclusion reached from the other direction in
`merge_design.md`: NewData should be a training cohort, not an evaluation one,
until it has more ET subjects.

## Round 4: per-class precision under LOCO, and two more refutations

16 log-bins, cap 90/class, 5 capping draws. LOCO figures are the mean over the
three held-out cohorts.

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| ResidualTCN ch=8 | 0.628 | 0.582 | 0.227 | 0.479 | 0.465 |
| **ResidualTCN ch=16** | 0.663 | 0.589 | **0.352** | **0.535** | 0.510 |
| ResidualTCN ch=32 | 0.660 | 0.587 | 0.193 | 0.480 | 0.476 |
| ResidualTCN ch=16 attn | 0.658 | 0.597 | 0.272 | 0.509 | 0.501 |
| **FUSION cnn+desc** | **0.716** | **0.601** | 0.287 | **0.535** | **0.515** |
| ENS fusion+rtcn | 0.686 | 0.586 | 0.305 | 0.526 | 0.510 |
| ENS fusion+rtcn+cnn | 0.697 | 0.595 | 0.286 | 0.526 | 0.513 |

Standard deviations, macroP / precET: ResidualTCN ch=16 **0.033 / 0.098**,
FUSION **0.010 / 0.026**. The two tie on mean macro precision; only one of them
is stable.

### Attention pooling does NOT transfer to the ResidualTCN

macroP 0.535 -> 0.509 and precET 0.352 -> 0.272 with `pool="attn"`. It was worth
+0.012 to the BiLSTM, so the transfer was expected and did not happen.

Read with the round-3 result that descriptor fusion also *hurts* the
ResidualTCN while helping the CNN, both point the same way: the residual TCN's
dilated stack already localises the tremor peak and already extracts what the
descriptors state. Additions that help weaker extractors are dead weight on it.

### Capacity optimum is sharp

ch=8 -> 0.479, ch=16 -> 0.535, ch=32 -> 0.480 macro precision. Halving or
doubling the width costs ~0.055. The ResidualTCN won its round-3 row at the one
setting that happens to be best, untuned.

### Ensembling helps within-site but not across-site

Under LOCO the ensemble reaches 0.526 macroP against fusion's 0.535 -- no gain.
Under the mixed-cohort protocol the same ensemble reaches macroF1 0.576 against
0.559/0.561 for its members -- a real gain. Ensembles buy within-site robustness,
not cross-site transfer.

### PD precision is what domain shift destroys

Under LOCO precN (0.63-0.72) exceeds precPD (0.58-0.60). Pooled and mixed, the
ordering flips (precPD 0.66-0.70 > precN 0.59-0.67). The middle class is the one
that does not survive being evaluated on an unseen cohort, which matches the
earlier observation that sequence models dissolve PD into N and ET.

## Round 5: the decision rule was the biggest single lever

Mixed-cohort protocol, 10 splits, `ENS fusion+rtcn` throughout.

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| baseline | 0.606 | 0.667 | 0.475 | 0.583 | 0.576 |
| **+ val-tuned priors** | 0.638 | 0.645 | **0.612** | **0.632** | 0.585 |
| + spectrum augment x2 | 0.605 | 0.656 | 0.439 | 0.567 | 0.562 |
| + rich descriptors (10 -> 34) | 0.599 | 0.680 | 0.473 | 0.584 | 0.573 |
| **+ priors + augment** | 0.663 | 0.639 | **0.640** | **0.647** | 0.592 |

### Tuning the decision rule beats every architecture change so far

ET precision 0.475 -> 0.612, macro precision 0.583 -> 0.632, from per-class
logit offsets fitted on the VALIDATION split and applied unchanged to test.
Every model here trains with balanced class weights -- which deliberately buys
ET recall at precision's expense -- and then predicts plain argmax. Nothing had
ever corrected that.

Two things to state honestly about it:

* macro F1 moves only 0.576 -> 0.585. This is largely **converting recall into
  precision**, not making the model better. That is the right trade when
  precision is the target metric, but it should not be described as a modelling
  gain.
* precET sd widens from 0.113 to 0.177 -- the tuned offset is itself variable
  across splits, because it is fitted on ~15 validation ET patients.

### Rich descriptors do nothing

10 -> 34 descriptors: macroP 0.583 -> 0.584. The extra biomarker and
regularity features carry nothing the original 10 do not, matching the dilution
seen when 122 spectral dimensions buried 6 asymmetry features.

### Augmentation only works with a tuned decision rule

Spectrum augmentation (+/-1 bin circular shift, ~0.37 Hz jitter, plus
multiplicative noise) **hurts alone** (0.583 -> 0.567) but **helps on top of
prior tuning** (0.632 -> 0.647). The reading is that augmentation distorts
probability calibration, which plain argmax cannot absorb and a tuned decision
rule can.

The unpaired gap (0.015) is well inside the per-config sd (~0.06), so this needs
the paired comparison on matched splits before it is quotable.

## Round 5b: two retractions

### The augmentation gain does not survive a paired test

The unpaired table showed `priors + augment` (macroP 0.647) above `priors`
alone (0.632). Paired on the same 10 splits:

| metric | paired diff | 95 % CI |
|---|---|---|
| precN | +0.025 | [+0.000, +0.049] * |
| precPD | -0.006 | [-0.040, +0.025] |
| precET | +0.028 | [-0.046, +0.094] |
| macroP | +0.016 | [-0.009, +0.039] |
| macroF1 | +0.007 | [-0.019, +0.036] |

**"priors + augment is the best configuration" is withdrawn.** The honest best
is prior tuning alone: macroP 0.632, precET 0.612. Spectrum augmentation does
nothing reliable in either direction, which also removes the "augmentation
distorts calibration and prior tuning absorbs it" story built on top of it.

### Two-stage does not fix PD dissolution

`TWO-STAGE fusion` (N-vs-Tremor, then a dedicated PD-vs-ET model on predicted
tremor) reaches macroP 0.568 against 0.583 for the flat 3-class head. Giving the
hard axis its own model does not help: PD is not suffering from shared capacity,
it is genuinely less separable.

## Round 6: the optimiser regime does not explain ResNet18

| config | precN | precPD | precET | macroP | macroF1 |
|---|---|---|---|---|---|
| ResidualTCN full-batch | 0.579 | 0.686 | 0.496 | **0.587** | 0.561 |
| ResidualTCN minibatch 16 | 0.593 | 0.653 | 0.455 | 0.567 | 0.559 |
| ResidualTCN minibatch 32 | 0.587 | 0.663 | 0.467 | 0.573 | 0.559 |
| ResidualTCN minibatch 64 | 0.596 | 0.672 | 0.460 | 0.576 | 0.566 |
| FUSION full-batch | 0.620 | 0.627 | 0.426 | 0.558 | 0.559 |
| FUSION minibatch 32 | 0.627 | 0.638 | 0.435 | **0.567** | 0.561 |

Minibatch training hurts the ResidualTCN at every batch size and helps FUSION
slightly. There is no general effect, so the earlier attribution of ResNet18's
ET precision to depth / residuals / BatchNorm **stands** -- the confound was
worth checking and turned out not to be one.
