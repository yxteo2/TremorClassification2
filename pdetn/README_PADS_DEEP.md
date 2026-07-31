# Deep BiLSTM cross-dataset experiment (local + PADS)

Runs the **deep** models for the PADS cross-dataset study, framed on the
**PD-vs-ET differential**. Compares two variants under two protocols. Run on
your machine (needs torch + PADS; the dev sandbox can't download PADS).

## Variants compared
- **3class** — one 3-class BiLSTM; report PD-vs-ET sub-accuracy from it.
- **two_stage** — deep N-vs-tremor BiLSTM, then a **dedicated** deep PD-vs-ET
  BiLSTM (trains only on PD+ET so the hard axis gets full model capacity), with
  an ET threshold tuned on a validation split.

## Protocols
- **P1** train LOCAL → test PADS — generalisation (the reviewer-gold number).
- **P2** pooled GroupKFold (5-fold, subject-grouped) over LOCAL+PADS — the
  n-fix (44 ET), tractable for deep training (5 fits/variant, not full LOSO).

## Setup
1. Download PADS (PhysioNet DOI 10.13026/m0w9-zx22 or Kaggle mirror).
2. Confirm the 4 `VERIFY:` constants in `tremor/pads_data.py` against the files.
3. GPU strongly recommended (deep training × several folds).

## Run
```bash
# dry run first (local only) — proves both variants train/predict
python -m pdetn.pads_deep_experiment --data-root Data --epochs 3

# full comparison (uses lower_arm ~ PADS wrist)
python -m pdetn.pads_deep_experiment --data-root Data --pads-root PADS \
    --action OUT --pads-condition OUT --sensor lower_arm --epochs 60
```

## Reports (per variant, per protocol)
PD-vs-ET accuracy (the headline), macro-F1, per-class F1, confusion matrix.
Saved to `artifacts/pads_deep/results.json`.

## Notes
- **Single sensor** (local `lower_arm` ~ PADS wrist) — spatial features excluded
  (PADS is wrist-only). Both are 100 Hz gyroscope, STFT-256 spectrograms.
- The deep model was weak on single-sensor *in-house* (data-hungry); PADS adds
  the volume it needs, which is the point of this experiment.
- If P1 generalisation is poor but P2 pooled is strong, that's a domain-shift
  signal — consider a domain adapter before trusting pooled numbers.
