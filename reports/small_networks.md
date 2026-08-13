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

## Caveat, stated at full strength

**6 ET subjects.** AUC 0.942 means near-perfect ranking of six patients; one or
two swaps would collapse it. No paired CI has been run, and there is no
correction across the six configurations tried. This project has retracted
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
