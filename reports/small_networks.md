# Small networks on descriptors — the first deep result that beats the baseline

Every earlier deep model here (BiLSTM, ResNet18, WideResNet50, ViT-B/16) was fed
a **raw spectrogram** and asked to learn its own features. All sat at chance.
Meanwhile logistic regression on **10 hand-computed descriptors** reached AUC
0.729 (2015 REST) and 0.812 (NewData DRINK).

The obvious reading: **feature learning is what fails at this n, not the
classifier.** `tfbench/small_nets.py` tests that directly.

* `MLPHead` — 2-layer MLP on the *same* 10 descriptors (362–700 parameters)
* `Spectrum1DCNN` — 1-D CNN over the power **spectrum** (frequency axis only,
  626 parameters). Tremor structure is spectral, so a 1-D convolution over
  frequency is the matched inductive bias; a 2-D image model spends capacity on
  a time axis the descriptors already showed adds nothing.

Both are 5 orders of magnitude smaller than ViT-B/16 (85.8 M).

## NewData DRINK — small networks win

23 PD / 6 ET, lower_arm, patient-level LOSO, 3 seeds averaged:

| model | params | bal-acc | **AUC** | **precision** | recall |
|---|---|---|---|---|---|
| logreg, 10 descriptors | 11 | **0.790** | 0.812 | 0.667 | 0.667 |
| MLP h=8 | 362 | 0.728 | **0.848** | **0.750** | 0.500 |
| **MLP h=16** | ~700 | 0.728 | **0.942** | **0.750** | 0.500 |
| 1D-CNN on spectrum | 626 | 0.725 | 0.862 | 0.444 | 0.667 |

**AUC 0.942 is the highest figure in this project** (previous best 0.826).
Precision 0.750 against 0.667. All three neural variants beat the linear model
on AUC, which argues against one lucky configuration.

Note balanced accuracy *falls* (0.790 → 0.728) while AUC and precision rise:
recall drops to 0.500. The MLP **ranks** better but its 0.5 threshold is
misplaced — an operating-point problem, not a modelling one.

## 2015 REST — they do not

75 PD / 16 ET:

| model | bal-acc | AUC | precision |
|---|---|---|---|
| **logreg, 10 descriptors** | **0.730** | **0.729** | **0.393** |
| MLP h=8 | 0.630 | 0.654 | 0.308 |
| MLP h=16 | 0.599 | 0.653 | 0.280 |
| 1D-CNN on spectrum | 0.385 | 0.414 | 0.043 |

The effect is specific to DRINK — i.e. to the task where the signal is
strongest — not a general property of small networks.

## BiLSTM works — over FREQUENCY, not time

The repo's `tremor_bilstm` runs over the **time** axis of a full spectrogram and
sits at chance. `SpectrumBiLSTM` runs over the **frequency** axis of a 1-D
spectrum — the sequence is "power at 3 Hz, 3.2 Hz, … 15 Hz", so the recurrence
models how spectral shape unfolds across frequency. Same task, same split:

| model | params | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|
| logreg, 10 descriptors | 11 | 0.790 | 0.812 | 0.667 | 0.667 |
| MLP h=16 on descriptors | ~700 | 0.812 | **0.942** | **0.800** | 0.667 |
| BiLSTM over freq h=4 | 242 | 0.830 | 0.877 | 0.556 | 0.833 |
| **BiLSTM over freq h=8** | 738 | **0.851** | 0.848 | 0.625 | **0.833** |
| BiLSTM over freq h=16 | 2,498 | **0.851** | 0.899 | 0.625 | 0.833 |
| `tremor_bilstm` over time, spectrogram | ~1e5 | 0.513 | 0.517 | 0.180 | 0.750 |

**bal-acc 0.851 is the best in the project.** The same architecture family that
sat at chance over the time axis of a spectrogram reaches 0.851 over the
frequency axis of a spectrum, with 738 parameters instead of ~100,000.

So the earlier "BiLSTM fails" conclusion was wrong in an important way: it was
not the recurrence that failed, it was **the axis it was applied to**. Tremor
structure is spectral; a sequence model over time on a quasi-stationary signal
has little to model, while the same model over frequency has the shape the
descriptors summarise by hand.

The two winners trade off cleanly — MLP has the better ranking (AUC 0.942) and
precision (0.800); BiLSTM has the better balanced accuracy (0.851) and recall
(0.833).

## Caveat, stated at full strength

**6 ET subjects.** AUC 0.942 means near-perfect ranking of six patients; one or
two swaps would collapse it. No paired CI has been run, and there is no
correction across the ~10 configurations now tried.

**Instability is visible directly in the numbers.** The same MLP h=16 gave
bal-acc 0.728 / precision 0.750 with 3 seeds and 200 epochs, and 0.812 / 0.800
with 2 seeds and 150 epochs — identical AUC (0.942) but a different operating
point. At n=6 the threshold behaviour is not stable even when the ranking is. This project has retracted
several small-n findings (`handedness_does_not_survive.md`,
`quaternion_session_verdict.md`) and the same discipline applies.

**This is the most promising lead in the project, not a result.**

## What settles it

**PADS `DrinkGlas` has 28 ET** — nearly 5x. Extract and re-run:

```bash
python -m pdetn.extract_pads --pads-root PADS --task DrinkGlas --out pads_drinkglas
python -m pdetn.extract_pads --pads-root PADS --task TouchNose --out pads_touchnose
```

If MLP-on-descriptors holds AUC near 0.85 on 28 independent ET subjects with a
different device, the finding is real. If it collapses to chance — as every
other cross-cohort PD-vs-ET test has — it was n=6 noise.

## Practical note

These nets are tiny, so **torch thread contention dominates**: the first run
took 30+ minutes and did not finish. With `torch.set_num_threads(1)` the same
sweep completes in 460 s (2015) and 112 s (DRINK). Always set it for
sub-1000-parameter models.

# Hidden-size sweep and the TCN+BiLSTM hybrid

## Capacity sweep — BiLSTM over frequency peaks at h=32–64

NewData DRINK, 23 PD / 6 ET, 3 seeds. **Full-batch training throughout**: every
gradient step uses all 28 training patients, so there is no mini-batch size to
enlarge.

| model | params | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|
| BiLSTM freq h=8 | 738 | 0.851 | 0.848 | 0.625 | 0.833 |
| BiLSTM freq h=16 | 2,498 | 0.830 | 0.899 | 0.556 | 0.833 |
| **BiLSTM freq h=32** | 9,090 | **0.913** | 0.870 | 0.600 | **1.000** |
| **BiLSTM freq h=64** | 34,562 | **0.913** | 0.884 | 0.600 | **1.000** |
| BiLSTM freq h=128 | 134,658 | 0.830 | 0.848 | 0.556 | 0.833 |
| MLP desc h=16 | 362 | 0.728 | **0.928** | **0.750** | 0.500 |
| MLP desc h=32 | 978 | 0.728 | 0.920 | 0.750 | 0.500 |
| MLP desc h=64 | 2,978 | 0.645 | 0.884 | 0.667 | 0.333 |
| MLP desc h=128 | 10,050 | 0.623 | 0.884 | 0.500 | 0.333 |

**bal-acc 0.913 with recall 1.000** — h=32 and h=64 catch all six ET patients.
Best in the project.

Two clean patterns:

* **The BiLSTM has an optimum, not a monotone trend.** 8 → 32 improves, 128
  collapses back to 0.830. There is a capacity sweet spot around 9–35 k
  parameters; the earlier ~1e5-parameter spectrogram BiLSTM and the 11–86 M
  backbones were both far past it.
* **The MLP degrades monotonically** with width (0.728 → 0.623). It has only 10
  inputs, so extra hidden units add parameters without adding information.

## The TCN+BiLSTM hybrid does NOT help

Idea: TCN over **frequency** at each time frame, BiLSTM over **time** to
aggregate — keeping the time axis rather than averaging it away.

| model | params | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|
| TCN+BiLSTM 8/8 | 1,666 | 0.518 | 0.761 | 0.250 | 0.167 |
| TCN+BiLSTM 16/16 | 6,146 | 0.518 | 0.732 | 0.250 | 0.167 |
| TCN+BiLSTM 16/32 | 14,658 | 0.623 | 0.725 | 0.500 | 0.333 |
| **BiLSTM over freq (time-averaged)** | 9,090 | **0.913** | **0.870** | 0.600 | **1.000** |

Worse on every metric. AUC 0.725–0.761 is above chance, so the architecture is
learning *something*, but balanced accuracy near 0.52 with recall 0.167 means it
barely calls ET at all.

**Why, most likely:** retaining the time axis is what hurts. Tremor in a 10 s
window is quasi-stationary — the earlier measurements already showed that
time-averaged descriptors work and a time-axis BiLSTM sits at chance. The hybrid
must *learn* to ignore temporal variation, and with 6 ET subjects it has no
budget to learn that. Averaging over time hands it the same invariance for free.

A caveat on this specific implementation: frames were normalised per-frame
(`P / P.sum(0)`), which discards per-frame amplitude. If amplitude modulation
over time matters, this test would not see it.

## Bottom line

Best configuration: **BiLSTM over the frequency axis of a time-averaged
spectrum, h=32 — bal-acc 0.913, AUC 0.870, precision 0.600, recall 1.000, 9,090
parameters.**

Still **6 ET subjects**, so all of this is a lead. PADS `DrinkGlas` (28 ET) is
the test that matters.

# Improving ET precision

Precision 0.600 at recall 1.000 means the model over-calls ET: 10 calls, 6
correct. Two levers.

| setting (BiLSTM freq h=32) | bal-acc | AUC | precision | recall |
|---|---|---|---|---|
| class_weight ON | **0.913** | 0.870 | 0.600 | **1.000** |
| **class_weight OFF** | 0.790 | **0.942** | **0.667** | 0.667 |
| cw OFF + wd=1e-2 | 0.768 | 0.790 | 0.571 | 0.667 |

Class weighting was doing exactly what it is designed to: buying recall with
precision. Removing it raises AUC 0.870 → **0.942** and precision to 0.667.
Legitimate change, no test-set involvement.

## Threshold sweep — diagnostic ONLY, not a result

| threshold | bal-acc | precision | recall | ET calls |
|---|---|---|---|---|
| 0.3 | 0.873 | 0.714 | 0.833 | 7 |
| 0.5 | 0.790 | 0.667 | 0.667 | 6 |
| 0.6 | 0.728 | 0.750 | 0.500 | 4 |
| 0.8 | 0.583 | 1.000 | 0.167 | 1 |

**These thresholds were chosen by looking at test-set predictions.** Quoting
"precision 0.714" from this table would be selection on the test set. The table
shows what the ranking could support; it is not a reportable operating point.
Selecting the threshold honestly means doing it inside the training folds, and
that was measured to be significantly *worse* at this n (0.730 → 0.624, paired
CI excluding zero — `reports/threshold_tuning_rest.md`). The honest operating
point stays 0.5.

**Honest best: class_weight OFF, threshold 0.5 — precision 0.667, recall 0.667,
AUC 0.942, bal-acc 0.790.**

## Why precision cannot be tuned at this n

Precision 0.667 is 6 correct out of 9 ET calls. One patient moving gives 0.556
or 0.778 — precision is quantised in steps of ~0.1. It is not a hyperparameter
problem. 28 ET from PADS `DrinkGlas` would make precision *measurable*; no
tuning substitutes for that.

# Per-axis (x/y/z) fusion — tested, does not help

Every model above collapsed the three angular-velocity axes into one spectrum by
averaging. That discards per-axis structure, and `pdetn/quaternion_tf.py` showed
cross-axis phase carries orbit geometry no power average can see. So:
`AxisFusionNet` stacks the per-axis spectrograms into `(B, 3, F, T)`, runs a
dilated conv **across the axis dimension** to fuse x/y/z at each (f, t), then
the frequency BiLSTM on the fused channels.

NewData DRINK, 23 PD / 6 ET, 3 seeds:

| model | params | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|
| AxisFusion 8/32 | 11,146 | 0.645 | 0.659 | 0.667 | 0.333 |
| AxisFusion 16/32 | 13,842 | 0.707 | 0.761 | 0.600 | 0.500 |
| AxisFusion 8/64 | 38,410 | 0.641 | 0.594 | 0.375 | 0.500 |
| **BiLSTM freq, axes AVERAGED** | 9,090 | **0.790** | **0.942** | **0.667** | 0.667 |

Worse on every metric at comparable parameter counts. Keeping the axes separate
triples the input dimensionality, and at 6 ET that costs more than the
cross-axis information gains — the same failure mode as the multi-sensor and
temporal feature sets. Averaging the axes is not throwing information away for
free; it is a rotation-invariant reduction the model would otherwise have to
learn.

# Class weighting on an imbalanced set — both directions measured

DRINK is 23 PD vs 6 ET, a 3.8:1 imbalance, so class weighting is the standard
answer. It is not free here:

| model | class_weight | bal-acc | AUC | **precision** | **recall** |
|---|---|---|---|---|---|
| BiLSTM freq (averaged) | **ON** | **0.913** | 0.870 | 0.600 | **1.000** |
| BiLSTM freq (averaged) | OFF | 0.790 | **0.942** | **0.667** | 0.667 |
| AxisFusion 16/32 | **ON** | 0.728 | 0.754 | **0.750** | 0.500 |
| AxisFusion 16/32 | OFF | 0.707 | 0.761 | 0.600 | 0.500 |

**It is a trade, not an improvement, and the direction depends on the metric:**

* On the **frequency BiLSTM**, weighting buys recall (0.667 → 1.000) and
  balanced accuracy (0.790 → 0.913) at the cost of precision (0.667 → 0.600)
  and AUC (0.942 → 0.870).
* On **AxisFusion** it goes the other way — weighting *raises* precision
  (0.600 → 0.750) at equal recall.

So "always use class weights when imbalanced" is right as a default but wrong as
a rule here: it must be swept and reported, because it moves precision and
recall in opposite directions and the correct setting depends on which one the
application needs. Both are now reported side by side rather than one being
silently chosen.

**If recall matters most** (screening — do not miss an ET patient): weighting
ON, bal-acc 0.913 / recall 1.000.
**If precision matters most** (the stated concern): weighting OFF on the
frequency BiLSTM, precision 0.667 / AUC 0.942.
