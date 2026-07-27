# Honest ET-LOSO baseline (OUT action) — result + why

**Goal.** A defensible Essential-Tremor (ET) classification number with a
subject-level confidence interval, and an explanation of what limits it. All
numbers below are leakage-free: subject-level splits, normalization fit on the
training fold only, model selected by validation loss, test scored once.

## Protocol

- **Data:** `Data/raw_quaternion/OUT`, 274 recordings / 151 subjects
  (N=120 rec / 61 subj, PD=125 / 75, **ET=29 / 15**).
- **Feature:** quaternion → angular velocity @ 100 Hz, all 3 sensors (9 ch),
  STFT with `f_max=15` Hz, per-recording normalization.
- **Model / loss:** `tremor_bilstm`, focal loss (γ=1.5), **no oversampling**
  (focal is the only imbalance correction, per repo guidance — do not stack).
- **Validation scheme:** ET-targeted leave-one-subject-out. One fold per ET
  subject; each ET subject is tested exactly once; per-fold test sets add 2
  subjects/other-class for balance. Per-fold predictions are pooled and scored
  once — the correct protocol when a class is this small.
- **Uncertainty:** 95% CI from a **subject-level** cluster bootstrap (resample
  subjects, not recordings); significance from a **subject-level** label
  permutation test. Both in `tremor/stats.py`.

## Headline (pooled, honest)

| metric | value | 95% CI (subject bootstrap) |
|---|---|---|
| **macro-F1** | **0.619** | [0.498, 0.731] |
| accuracy | 0.712 | — |
| N F1 | 0.853 | [0.725, 0.937] |
| PD F1 | 0.679 | [0.500, 0.811] |
| **ET F1** | **0.324** | **[0.069, 0.560]** |

AUC: N 0.90, PD 0.80, **ET 0.76**.
Permutation test: **p(macro-F1) = 0.0005** (n=2000) — the classifier carries
real subject-label information; it is not chance.

The ET-F1 CI is wide by necessity: 15 ET subjects, ~2 per test fold. This is the
honest consequence of cohort size, and it is exactly why every number carries a
subject-level CI rather than a bare point estimate.

## Why ET is hard — three converging views

### 1. It's a PD-vs-ET problem, not an N-detection problem
Pooled confusion matrix (rows=true, cols=pred):

```
        pred:  N   PD   ET
  true  N      55    5    0
  true PD      10   38    2
  true ET       4   19    6
```

Decomposed from the *same* predictions:

- **Stage A — N vs tremor (PD+ET):** macro-F1 **0.863**. Normal is essentially
  solved.
- **Stage B — PD vs ET:** macro-F1 **0.574**. This is the entire ceiling. Of 29
  true ET recordings, **19 are called PD**; only 6 are called ET.

### 2. ET is rankable but under-called
ET AUC is 0.76 (well above chance) yet ET sensitivity is only **0.21** at
specificity **0.98**. The model rarely *falsely* flags ET but misses most real
ET — a classic minority-class threshold collapse. Focal loss alone did not move
the decision boundary far enough.

Sweeping the ET one-vs-rest operating point:

| p_ET ≥ | sens | spec | prec | ET-F1 |
|---|---|---|---|---|
| 0.20 | 0.86 | 0.26 | 0.24 | 0.370 |
| 0.25 | 0.83 | 0.52 | 0.31 | 0.453 |
| 0.30 | 0.59 | 0.82 | 0.46 | 0.515 |
| 0.333 | 0.45 | 0.93 | 0.62 | 0.520 |
| 0.50 | 0.14 | 1.00 | 1.00 | 0.242 |

Moving the threshold lifts ET-F1 from 0.32 to **~0.52**. **Caveat:** this
threshold is tuned on the test predictions, so 0.52 is an *optimistic upper
bound*. Banking it honestly requires selecting the threshold inside CV (on
validation folds) — a concrete next step, not a result yet.

### 3. The ceiling is a handful of subjects
Per-ET-subject outcome (15 subjects):

- **2 fully correct** (mean p_ET 0.52–0.58), **2 partial**, **11 fully missed**.
- Most missed subjects sit at mean p_ET ≈ 0.28–0.36 — *just below* the 3-class
  boundary. They are near-misses pushed into PD, not confident errors.
- One subject (ET 16, p_ET 0.035) looks like Normal to the model and is
  effectively unlearnable from this feature.

So the gap is dominated by near-threshold ET subjects plus ~1–2 genuinely hard
ones — consistent with the threshold-collapse view above.

## The ceiling, stated plainly

- With focal-only argmax, the **honest ET-F1 is ≈ 0.32 [0.07, 0.56]**.
- The **realistic reachable ceiling** on this feature, via cost-sensitive
  thresholding, is **~0.5** — bounded by heavy PD↔ET spectral overlap and ~1–2
  ET subjects that resemble N/PD.
- **N detection is solved** (Stage-A macro-F1 0.86); effort should target the
  PD-vs-ET boundary, not N.

## Not runnable in this environment

- **amplitude @ 60 Hz feature arm** (the prior MATLAB pipeline's representation):
  only `raw_quaternion` is present in this container, so the amplitude-vs-angular-
  velocity comparison could not be run. It needs the amplitude feature folder and
  remains an open follow-up.

## Reproduce

```bash
python -m tremor.cv_benchmark --data-root Data --action OUT \
  --data-mode quaternion --model tremor_bilstm \
  --quaternion-sensors hand,lower_arm,upper_arm \
  --tfd-methods stft --f-max 15 --normalize per_recording \
  --loss focal --focal-gamma 1.5 --oversample-to -1 \
  --loso-target-class ET --epochs 80 --patience 15 \
  --n-boot 2000 --n-perm 2000 --seed 42 \
  --output artifacts/phase1_baseline/
```

Outputs: `artifacts/phase1_baseline/stft/pooled_report.json` (metrics + CI +
p-value) and `pooled_predictions.csv` (per-recording, for the tables above).
Runs in ~3 min on CPU.
