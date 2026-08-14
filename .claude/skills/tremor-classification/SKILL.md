---
name: tremor-classification
description: Work inside the TremorClassification2 repo — a PyTorch pipeline that classifies IMU recordings as Normal (N), Parkinson's (PD), or Essential Tremor (ET) from quaternion / amplitude / STFT features. Use this skill whenever the user loads tremor recordings, computes STFT/CWT/HHT/multitaper/SST time-frequency features, builds or trains the BiLSTM/TCN/AST/ResNet/MIL models, runs subject-level splits or CV benchmarks, debugs class imbalance (especially low ET F1), or invokes `tremor.train`, `tremor.cv_benchmark`, or `tremor.compare_preprocessing`. Trigger this even when the user just references quaternion data, angular velocity, displacement amplitude, spectrograms, the `tremor` package, or any of its modules — do not reinvent conventions the repo already fixes.
---

# Tremor Classification (TremorClassification2)

A PyTorch package (`tremor/`) for 3-class tremor classification from wearable IMU data:
**N=Normal, PD=Parkinson's, ET=Essential Tremor**. ET is severely underrepresented (only
**~15 unique ET subjects** in OUT) and is the hardest class — when a tradeoff exists, optimize
**macro-F1 and ET F1**, not raw accuracy.

## Non-negotiable invariants

The code already enforces these. Never write code that breaks them; they are the difference between
a real result and a leaked one.

1. **Splits are at the SUBJECT level**, never the recording level. All trials of one patient stay
   together in one fold. Use `tremor.splits.subject_level_split` (GroupShuffleSplit on subject) or
   `GroupKFold`. A subject ID is the filename with the trailing trial number stripped.
2. **Augmentation and oversampling apply to the TRAINING fold only** — never val or test. Random-
   pad/crop, oversampling, SpecAugment, `WeightedRandomSampler`. Applying any to val/test inflates
   metrics.
3. **Split first, then normalize.** Fit normalization statistics on the training fold only.
4. **Select the model by validation loss only.** Evaluate the test set exactly once, at the end.
5. **Subject IDs embed the action** (`ET 10_OUT`). Fine for single-action work, but 136/154 subjects
   appear in all three actions, so **cross-condition training silently leaks** while
   `_assert_disjoint_subjects` still passes. Strip `[_\s]+(OUT|REST|WING)$` after the trial suffix
   before any cross-action split.

## Data at a glance

```
<root>/Data/[ProcessedData/[raw data/]]<feature>/<ACTION>/<CLASS>/<file>.txt
  ACTION ∈ {DRINK, EAT, FNF, OUT, REST, WING}   CLASS ∈ {ET, N, PD}
```

Two **different feature families** live in the tree — do not confuse them:

| feature (`--feature`)            | channels | native fs | what it is |
|----------------------------------|----------|-----------|------------|
| `raw_quaternion`                 | 12→9     | **100 Hz**| orientation → angular velocity (`--data-mode quaternion`) |
| `filtered_amplitudes`            | 3        | **60 Hz** | displacement amplitude (`--data-mode raw`) |
| `downsize_filtered_amplitudes`   | 3        | **60 Hz** | pre-downsampled amplitude — the MATLAB pipeline's actual input |

> **fs TRAP — the most important single fact.** The `*_amplitudes` files are **already downsampled
> to 60 Hz**. Passing `--fs 100` on them mislabels every frequency by 1.67× and *silently produces
> plausible nonsense* (peaks land in the right-looking band but at the wrong Hz). Use **`--fs 60`**
> for any amplitude feature; `--fs 100` only for `raw_quaternion`. Do **NOT** pass `--apply-bandpass`
> on `filtered_*` features — they are already filtered.

Each quaternion `.txt` is `(T, 12)` = 3 sensors × 4 components, **scalar-last `(x, y, z, w)`**.
Class is the leading filename letter (`N`/`P`/`E`). `CLASS_MAP = {N:0, P:1, E:2}`,
`CLASS_NAMES = ("N","PD","ET")`, `SENSOR_NAMES = ("hand","lower_arm","upper_arm")` (distal→proximal).
Hand carries the most tremor **amplitude**, but a 168-feature scan found **proximal sensors
(lower/upper arm) carry MORE of the PD-vs-ET discriminative signal** — do not default to hand-only
when the goal is PD-vs-ET separation.

Loaders return `tremor.data.Recording(x:(channels,time), y, subject, path)`:
`load_quaternion_recordings` (raw quaternion), `load_recordings` (amplitude/time-domain),
`load_stft_recordings` (precomputed STFT, `STFTRecording`). Feature/quaternion/TFD/normalization
detail is in **`references/data-and-preprocessing.md`** — read it before touching feature code.

## Feature (TFD) methods

`--tfd-method {stft, cwt, hht, wavelet_packet, multitaper, sst}` (default `stft`), all producing a
`(channels·kept_bins, frames)` layout via `TremorDataset`. STFT defaults are MATLAB-aligned:
`--nperseg 128 --noverlap 96 --nfft 128 --f-max 30` (65 one-sided bins, 39 kept ≤30 Hz, 0.78 Hz/bin
at fs=100 → 0.47 Hz/bin at fs=60). Higher-resolution amplitude STFT (fs=60, longer window span)
resolves the 1–2 Hz PD/ET peak gap that the 100 Hz quaternion STFT blurs. `multitaper` (DPSS,
~26× lower spectrogram variance) and `sst` (synchrosqueezed STFT, ~2× sharper peaks) were added as
denoise/sharpen alternatives — neither improves *univariate* PD-vs-ET separation (see guardrails),
but both are legitimate comparison columns.

## Models (`tremor/models.py`)

Registry (`--model`), all `(B,F,T) → (B,num_classes)` logits:
`tremor_bilstm` (default), `bilstm`, `lstm`, `gru`, `resbilstm`, `restcn`, `ast`, `resnet18`.
Instantiate via `tremor.models.build_model(name, input_size, num_classes, ...)`. Gotchas:
- **`ast` `target_T` must be the post-TFD FRAME count** (`sample_x.shape[1]`), NOT the raw sample
  length — passing the raw length mis-sizes the positional embedding and crashes at `x + pos_embed`.
- `restcn` uses `base_filters=hidden` (use **≥64**; 32 underfits 129-bin input).
- `resnet18` needs `n_input_channels` to divide F, and `resize_to`.
- `ast`/`resnet18` honor `--no-pretrained` / `--no-freeze-backbone`. Pretrained weights download
  from HuggingFace/torchvision — offline/CI runs must pass `--no-pretrained`.
- **All five recurrent models end in `x.mean(dim=1)`**, averaging the time axis away. Tremor
  amplitude *fluctuation over time* is a plausible PD/ET cue that mean-pooling discards — this is
  the motivation for the MIL path below.

## Loss for imbalance (`tremor/losses.py`)

`--loss {ce, weighted_ce, focal}` (default `focal`). **For the ET imbalance, reach for the loss
function first, not augmentation.** Start with `focal` (`--focal-gamma`, default 1.5); if still weak,
try inverse-frequency `weighted_ce`. **Do NOT stack oversampling (`--auto-oversample` /
`--oversample-to`) AND a class-weighted loss (`focal`/`weighted_ce`)** — the double correction
overshoots and collapses the majority classes; the code warns when both are on. Pick one (focal is
cleaner; to run focal alone in `cv_benchmark`, pass `--oversample-to 0`). Augmentation is a later
lever, not the opening move.

## Cross-validation & statistics

- `cv_benchmark.py` does subject-level `GroupKFold`; `--loso-target-class ET` switches to
  **ET-targeted LOSO** (one fold per ET subject, each tested once) and prints a **POOLED per-class
  AUC** table — the stable ET number, since per-fold ET F1 swings wildly on 1–2 test samples.
- Report **subject-level (cluster) bootstrap CIs**, resampling *subjects* not recordings (trials of
  one patient are correlated). See `tremor/stats.py` (`bootstrap_ci`, `permutation_test`,
  `paired_permutation_test`, `format_ci`) once the pending patch is applied.
- Both `splits.py` and `cv_benchmark.py` group by subject but are **NOT stratified**;
  `StratifiedGroupKFold` is the correct-but-unimplemented fix.

## Multiple-Instance Learning (`tremor/mil.py`, pending patch)

MIL over spectrogram windows (Papadopoulos et al., IEEE JBHI 2019/2023): a recording is a *bag* of
`(K, F, W)` windows; the TFD is computed once per recording then sliced. Pooling: `mean`,
`attention`, `mean_stats`, `attention_stats`. **Critical ordering:** log-compress + z-score run on
the FULL spectrogram *before* slicing — per-window normalization flattens each window to zero-mean/
unit-variance and destroys the amplitude contrast attention needs. `MILDataset` never calls
`fit_length` (so it is immune to the pad-corruption bug below). ONNX-verified `[1,K,F,W]→[1,3]`.

## Running things

```bash
# Amplitude pipeline (the MATLAB-equivalent feature) — NOTE --fs 60, no bandpass
python -m tremor.train --data-root Data --action OUT \
    --data-mode raw --feature downsize_filtered_amplitudes \
    --fs 60 --f-max 30 --nperseg 128 --noverlap 96 --nfft 128 \
    --model tremor_bilstm --loss focal --seed 39 --output artifacts/amp_OUT/

# Quaternion → angular velocity pipeline (fs 100)
python -m tremor.train --data-root Data --action OUT \
    --data-mode quaternion --quaternion-mode angular_velocity \
    --model tremor_bilstm --tfd-method stft --f-max 30 \
    --normalize per_recording --loss focal --output artifacts/

# ET-targeted LOSO across TFD methods (pooled AUC)
python -m tremor.cv_benchmark --data-root Data --action OUT \
    --data-mode quaternion --model ast \
    --tfd-methods stft multitaper sst \
    --loso-target-class ET --oversample-to 0 --output cv_results/OUT/
```

The complete flag reference is in **`references/cli-reference.md`**.

## Empirical guardrails (measured, do not re-derive)

- **The published paper is NOT a baseline.** Its accuracy was obtained by deliberately selecting the
  random seed. On the real split sizes, ET F1 ranges seed-worst 0.000 / median 0.667 / best 1.000.
  Do not treat it as a target or comparison point; LOSO removes the split seed entirely.
- **PD-vs-ET is NOT univariately separable in OUT.** Peak-frequency Mann-Whitney **p≈0.97**, peak
  amplitude **p≈0.39**; across 168 scalar features, 0 survive Bonferroni. **N-vs-tremor separates at
  p<0.0001.** So in a confusion matrix, **ET→PD is expected** (points at pooling / better model);
  **ET→N means something else is wrong** (feature, fs, or normalization bug).
- **Only ~15 ET subjects.** A genuine +0.124 ET F1 improvement sits at **p=0.25**; a 0.571 ET F1
  carries a 95% CI of [0.333, 0.750]. "80% ET" is reachable by luck, not demonstrable — steer toward
  a defensible number **with an interval**.
- **`--length-mode pad` corrupts normalization** (zero-padding inflates log-std and buries the peak;
  padding fraction does not correlate with class, so it's corruption not leakage). **Avoid.** Use
  `truncate` (default) or MIL windows.
- **MIL for tremor is already published (JBHI ×2)** by the group likely to review this work. Novelty
  must come from *characterization* (why ET fails here, with evidence) + a pooling ablation — not
  from "we applied MIL."
- **Biggest available lever for a paper:** the single-condition / single-3-class-model constraints
  are self-imposed. Dropping them (cross-condition rest/kinetic ratio; two-stage
  tremor-detect-then-type) exploits the fact that N-vs-tremor is easy while PD-vs-ET is hard.

## Working style for this repo

- The user is a mechanical engineer: strong in signal processing, weaker in coding/ML methodology.
  Explain method *characteristics*, not mechanics. Be direct and technical; keep it short.
- Be honest about projected vs measured performance; never present an estimate as a result. The user
  has correctly pushed back on claims not grounded in the actual codebase — audit source first, do
  not invent variable names.
- Reuse `TremorDataset`, `build_model`, `build_loss_fn`, `subject_level_split` rather than rewriting.
- Read a module's docstring before editing — they carry hard-won rationale (e.g. why
  `per_freq_zscore` erases the tremor peak and should be avoided).
- Keep changes ONNX-export-friendly: the deployment path is C++ ONNX Runtime, so prefer ops that
  trace cleanly and keep train / export / inference preprocessing identical.

## Reporting metrics — precision is not optional

**Never lead with balanced accuracy on a minority class.** With
`class_weight="balanced"` the classifier deliberately trades precision for
recall so the ~10 %-prevalence ET class is not ignored. Balanced accuracy
rewards that trade; precision exposes what it costs. Measured on this cohort:

| headline | what it hides |
|---|---|
| PD-vs-ET bal-acc **0.730** | **ET precision 0.219** — 32 ET calls, 7 correct |
| max+mean frequency only | **ET precision 0.093** |
| PADS StretchHold (best anywhere) | ET precision 0.339, recall 0.679 |

The dominant error is **PD → ET**, not N → ET: the model confuses the two tremor
types, not tremor with health. `tfbench.merged.report` now prints P and R
alongside bal-acc, and `tfbench.merged.per_class_report` gives the full
per-class table — use it for any 3-class result.

Raw accuracy is worse than useless here. Majority baselines: **0.833** (2015
PD-vs-ET), **0.908** (PADS PD-vs-ET). A model can beat balanced chance and still
sit below the majority baseline on raw accuracy.

## Statistical discipline (learned the hard way in this repo)

Three results were reported and then retracted in a single session. All three
would have been caught by the checks below, all of which are cheap.

1. **Paired CI belongs in the same run as the point estimate.** For any "B beats
   A" claim, bootstrap the *difference* on the same resampled subjects. Two
   independent CIs are not a comparison. `tfbench.benchmark.rank_methods` does
   this.
2. **Correct for multiplicity, and say how many tests.** A 198-test screen makes
   p=0.00035 non-significant (Bonferroni 2.5e-4, BH q=0.070). `rank_methods`
   prints `*` (uncorrected CI) and `BONF-PASS`/`bonf-fail` separately — they are
   different tests and must not be merged.
3. **`n_boot` must resolve the threshold.** At `n_boot=1000` the smallest
   non-zero p is 0.001; a comparison read p=0.0040 (would pass α=0.00455) and at
   20 000 draws was p=0.0083 (fails). Use ≥20 000 when close.
4. **Check a second condition before writing anything down.** The headline
   handedness effect was OUT-only and *reversed sign* at WING. REST/WING cost
   nothing to run.
5. **In-CV threshold tuning can HURT at this n.** It makes the best config
   significantly worse (0.730 → 0.624, paired CI [−0.21, −0.01]). Tuning helps
   only where 0.5 is genuinely misplaced. Always run tuning-off as a control.

## Cross-dataset rules

* **Device-identity probe before any pooling**, on the minority class only,
  judged on **|AUC − 0.5|**. An AUC of 0.000 is *maximally* separable with
  LOO-inverted labels, not "safe" — that mistake was made here.
* **PADS cannot be training data.** Confirmed twice, before and after the label
  fix: pooling costs ET-F1 0.538 → 0.350 and the identity probe is 1.000.
* **PD-vs-ET does not transfer between cohorts**, even task-matched
  (rest→rest). External AUC 0.387–0.424, consistently *below* chance, because
  the cohorts disagree on the **direction** of the effect: 2015 has PD slower
  than ET at REST (5.47 vs 6.15 Hz, p=0.042), PADS has PD *faster* (6.91 vs
  5.90 Hz, p<0.0001).
* **N-vs-Tremor does transfer**: 2015-trained → PADS StretchHold, bal-acc 0.736,
  AUC 0.783.
* The best condition is **cohort-specific**: REST for 2015, StretchHold for
  PADS. Do not generalise a condition recommendation across cohorts.

## Dataset gotchas (each cost real time)

* **PADS labels.** Diagnoses are clinical free text; substring matching put 13
  non-ET patients into ET ("etiology", "asymmetric", "Retrocollis",
  "hypokinetic") and 21 parkinsonian mimics into PD. Use **exact** matching:
  `Healthy` / `Parkinson's` / `Essential Tremor` → **79/276/28**, which matches
  Varghese 2024. If ET reads 41, the filter has regressed.
* **PADS task tokens differ from PADS's own script.** Files contain `Relaxed`
  and `Entrainment`, not `Relaxed1`/`Relaxed2`. Match the task field **exactly**
  — `Relaxed` as a substring also catches `RelaxedTask`, a different condition.
* **PADS task durations differ**: Relaxed 2048 samples, StretchHold 1024.
  Control for it before comparing tasks.
* **NewData (2025) is unsegmented.** ~38 s `Free_Form` exports with an *empty*
  Annotations table; whole-recording analysis leaves only **9.9 %** of power in
  the tremor band vs 76.5 % (2015) and 81.2 % (PADS). `load_2025(segment=True)`
  is the default; any 10 s window works, so this is not a selection artifact.
* **NewData is ET-only (6 subjects)** — no classification metric exists for it
  alone, and any device cue is perfectly confounded with the ET label.

## Cohort combinability

Run `python -m tfbench.combinability` before pooling. Both must hold:
frequency-distribution equivalence within ±1 Hz **and** a passing device probe.
2015 + NewData fails at both conditions, for opposite reasons (REST:
frequencies differ, p=0.017; OUT: device-separable). Merging moves the ET class
from 38 % to 50 % of patients below the PD median — it becomes bimodal,
straddling PD.

**Adding NewData ET has no measurable effect** (paired CI spans zero for both
stft512 and welch, with opposite signs). The often-quoted merged 0.740/ET-F1
0.557 is scored on a *different patient set*; on the same 2015 patients it is
0.728/0.480.

## Signal-processing facts (verified, do not re-derive)

* **Power vs amplitude.** Four transforms returned |S| not |S|², so every
  power-weighted descriptor used amplitude weights. Verify any new transform
  against Parseval: doubling the signal must **quadruple** reported power.
* **Only Welch is Parseval-exact.** Integrated-power / true-power ranges
  0.15–1.4e4 across the 12 transforms, so `total_power` is comparable *within* a
  method only.
* **Bin width matters.** VMD and the S-transform have non-uniform frequency
  grids; weight by power × bin width or mean/median frequency is biased.
* **The quaternion → angular-velocity conversion is faithful** — verified
  against the raw gyroscope stream in the 2025 h5 files (identical to three
  decimals). There is no ω tilt to correct; de-tilting makes domain shift
  *worse* (probe 0.629 → 1.000).
* **Plain HHT is noise-dominated** (EMD puts broadband noise in IMF1). Drop
  IMF1: a clean 6 Hz tone is recovered at 6.00 Hz instead of the band edge.
* **`lower_arm` alone beats every sensor combination** at both OUT and REST.
  Adding sensors dilutes: 30 features against 16 ET subjects overfits.
* **Condition is worth ~0.2 balanced accuracy; method choice ~0.04**, and the
  method effect is only resolvable at n≈28 ET. Spend effort on condition and
  cohort, not transforms.

## Current best results

| axis | config | metric |
|---|---|---|
| N vs Tremor | 2015 OUT, merged, → PADS external | internal 0.814, **external 0.736 / AUC 0.783** |
| PD vs ET | 2015 REST, lower_arm, stft512, thr 0.5 | bal-acc 0.730, **ET precision 0.219** |

Nine-plus feature families have now landed within CI on PD-vs-ET. Treat a new
feature family as unlikely to help unless it survives a paired CI.

## Deep learning — what actually works here

Ten architectures tested on PD-vs-ET. The variable that mattered was never
capacity; it was **what the model is asked to read**.

| model | params | AUC | note |
|---|---|---|---|
| **BiLSTM over FREQUENCY, time-averaged spectrum, h=32** | **9,090** | **0.870–0.942** | best; `tfbench.small_nets.best_model` |
| MLP on the 10 descriptors, h=16 | 362 | 0.928 | best precision |
| 1D-CNN over the spectrum | 626 | 0.862 | |
| logreg on 10 descriptors | 11 | 0.812 | the bar to beat |
| BiLSTM over TIME on a spectrogram | ~1e5 | 0.517 | chance |
| ResNet18 / WideResNet / ViT (frozen) | 11–86 M | 0.47–0.54 | chance |
| TCN(freq)+BiLSTM(time) | 1.6–15 k | 0.73–0.76 | worse |
| per-axis x/y/z fusion | 11–38 k | 0.59–0.76 | worse than averaging axes |

Rules that fall out of this, all measured:

* **Frequency axis, not time.** Tremor is quasi-stationary over a 10 s window,
  so a sequence model over time has little to model. The same BiLSTM family
  goes from chance (time) to best-in-project (frequency).
* **Capacity has an optimum, ~9–35 k.** h=8 → 0.851, h=32 → 0.913, h=128 →
  0.830. Both the ~1e5 spectrogram BiLSTM and the 11–86 M backbones are past it.
* **Feed features, not raw spectrograms.** Feature *learning* is what fails at
  this n, not neural networks.
* **Average the x/y/z axes.** That is a rotation-invariant reduction, not a free
  loss; keeping them separate triples input dimensionality and measures worse.
* **Class weighting is a trade.** On the frequency BiLSTM it buys recall
  (0.667 → 1.000) and costs precision (0.667 → 0.600) and AUC (0.942 → 0.870).
  Sweep it and report both; do not assume "imbalanced ⇒ use weights".
* **torch threading destroys these models.** Tiny tensors mean thread sync
  dominates: `torch.set_num_threads(1)` turns 30+ minutes into ~2.

## Task choice beats everything else

ET is a **kinetic** tremor. On the 2025 cohort, PD-vs-ET by task:

| task | bal-acc | AUC | ET precision |
|---|---|---|---|
| **DRINK** | 0.790 | 0.812 | **0.667** |
| **FINGER_NOSE** | 0.708 | **0.826** | 0.400 |
| POUR / PRON_SUP / TAP | 0.42–0.57 | 0.44–0.53 | ≤0.23 |
| REST / OUT | 0.39–0.49 | 0.20–0.27 | ≤0.18 |

Every earlier attempt varied the *method* on rest/postural recordings and
failed. Varying the *task* produced the largest jump in the project. Caveat:
**6 ET**, no paired CI, no correction across 7 tasks, no replication cohort yet
— PADS `DrinkGlas`/`TouchNose` (938 files each, 28 ET) is the decisive test.

## Known-unfixed bugs (as of this skill revision)

- `tremor/quaternion_data.py`: the `candidates` list contains the **same path twice**, so the
  `ProcessedData/` and MATLAB `raw data/` layouts never resolve; and it calls
  `process_quaternion_data` unconditionally (crashes on 3-col amplitude files). Fixed by the pending
  `tremor_pipeline.patch`.
- `cv_benchmark.py` has **no bandpass** flag/call at all, so its numbers are not comparable to
  `train.py` when bandpass is used.
- `_seed_everything` drives split seed and init seed together — variance cannot be attributed to one
  or the other.

### Merged-cohort evaluation: report leave-one-cohort-out, not pooled k-fold

Measured on 2015 + NewData + PADS (n=355, 49 ET, PADS capped at 60/class),
frequency-axis spectra, class weights on:

| model | pooled macroF1 | mean LOCO macroF1 | gap |
|---|---|---|---|
| logreg | 0.472 +/- 0.014 | **0.435** | -0.037 |
| MLPHead h=16 (1k) | 0.508 +/- 0.022 | 0.434 | -0.074 |
| Spectrum1DCNN (1k) | **0.574 +/- 0.008** | 0.425 | -0.149 |
| SpectrumTCN (3k) | 0.484 +/- 0.016 | 0.396 | -0.088 |
| SpectrumBiLSTM h=32 (9k) | 0.496 +/- 0.025 | 0.366 | -0.130 |

Pooled k-fold puts PADS patients in both train and test, and PADS is 42 % of
the pool with the cleanest ET. Every deep model's advantage is cohort fitting:
the pooled-to-LOCO gap grows with capacity, and under LOCO **no model beats
logistic regression on average**. Spectrum1DCNN predicts ZERO ET on an unseen
NewData (precision 0.000, recall 0.000) while logreg gets macroF1 0.420 there.

Do not quote a pooled merged-cohort number without the LOCO number beside it.

### Do not judge a model on one fold split

Single split said "BiLSTM h=32 (0.470) loses to logreg (0.479)". Over 5 splits
BiLSTM is 0.496 +/- 0.025 against logreg 0.472 +/- 0.014 -- the opposite
conclusion. Split-to-split sd on macroF1 here is 0.008-0.025; differences below
~0.05 need repeated splits before they mean anything.

### Bilateral limb asymmetry is the only feature that moves PD-vs-ET

On PADS StretchHold (n=304, 28 ET), four unsigned between-limb dissimilarity
descriptors (`tfbench.small_nets.asym_feats`) reach AUC 0.730 where the
single-limb spectrum sits at 0.527 and 122 concatenated bilateral bins reach
0.605. Paired subject bootstrap dAUC +0.183 [+0.031, +0.343]; permutation null
p = 0.000; null on N-vs-Tremor (+0.032 [-0.041, +0.104]), which is the correct
axis-specificity.

Two caveats that are easy to get wrong:
* PADS has a handedness bias (ET 0.179 left-dominant vs PD 0.388), so the
  SIGNED log-ratio features can read side rather than asymmetry. Dropping them
  raises AUC to 0.730. Use the 4 unsigned features.
* `concat+asym` (128 dim, AUC 0.554) is WORSE than `asym` alone (6 dim, 0.709).
  At 28 positives, 122 uninformative dims bury 6 informative ones. This is also
  the mechanistic reason a bilateral transformer over 2F frequency tokens
  should not be expected to help at this n.

### Contrastive / InfoNCE losses are contraindicated here

Contrastive learning builds invariance to whatever the positive pairs declare
to be nuisance. Every natural choice on this data deletes a measured biomarker:
amplitude scaling removes tremor amplitude; time warping removes tremor
frequency (the primary PD/ET discriminator); left-vs-right-wrist positives
remove the asymmetry finding above; same-patient-different-task positives learn
patient identity, i.e. device and session.

### Window-level metrics do not inflate results on this data

Tested because a published paper reports window-level metrics. 2 s windows at
50 % overlap, patient-grouped folds both ways: window-level is LOWER than
patient-level on N-vs-Tremor (0.774 vs 0.815 on 2015; 0.673 vs 0.714 on
NewData). Averaging to patients denoises more than correlated rows inflate. Do
not use "they report window-level" as an explanation for a performance gap.
