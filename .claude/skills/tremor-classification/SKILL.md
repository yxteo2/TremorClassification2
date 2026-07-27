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

### Moveo Explorer exports — a THIRD, incompatible source (`tremor/moveo_data.py`)

The newer recruitment batches (Drive: `Tremor Classification IMU/{ET,PD,HC}/`, 116 subjects not in
`Data/`) ship as APDM/Mobility Lab subject exports, and they are **not** `raw_quaternion`:
**fs=128 Hz** (not 100 — the fs trap again, 1.28× error), quaternions **scalar-first `(w,x,y,z)`**,
and the payload is **4 joint-angle streams** (elbow/wrist × L/R) rather than 3 per-sensor
orientations. Every trial is labelled `Free Form` — there are **no OUT/REST/WING labels** — and the
h5 *filename* carries the acquisition-station ID, not the study ID (`PD 88`'s files say `PD_1`, and
`PD 1` already exists), so subject IDs come from the export **directory**. Use
`load_moveo_recordings` / `moveo_inventory`, and `joint_quaternions_from_sensors` to convert the old
3-sensor files into the same elbow/wrist representation. The exports also contain **patient names
and DOBs** (`SubjectMetadata.xml`, `*_Trial.csv`) — never commit any of the tree; it is gitignored.
Full assessment, blockers and open questions: **`reports/track4_moveo_export.md`**.

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

## Known-unfixed bugs (as of this skill revision)

- `tremor/quaternion_data.py`: the `candidates` list contains the **same path twice**, so the
  `ProcessedData/` and MATLAB `raw data/` layouts never resolve; and it calls
  `process_quaternion_data` unconditionally (crashes on 3-col amplitude files). Fixed by the pending
  `tremor_pipeline.patch`.
- `cv_benchmark.py` has **no bandpass** flag/call at all, so its numbers are not comparable to
  `train.py` when bandpass is used.
- `_seed_everything` drives split seed and init seed together — variance cannot be attributed to one
  or the other.
