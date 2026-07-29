# pdetn — condition-aware N/PD/ET separation

Kept **separate from the deep-learning pipeline in `tremor/`**. This package
attacks the 3-class separation with **interpretable, condition-aware features**
and a **two-stage** classifier, and lets you compare against the deep TFD models.

## Idea
This session's findings drive the design:
- **N-vs-tremor is easy; PD-vs-ET is the ceiling.** → a **two-stage** model
  (stage 1: N vs tremor; stage 2: PD vs ET) so the hard decision isn't diluted.
- **PD-vs-ET separates at REST**, and the **rest-vs-action power contrast**
  encodes the PD=rest / ET=action dichotomy. → each patient becomes one vector
  of spectral biomarkers per condition **plus** cross-condition contrasts.
- **ET collapses under the PD majority** (RF gives ET-F1=0). → stage 2 uses an
  **ET probability threshold tuned by internal CV** on the training fold
  (leakage-free), which recovers ET.

## Files
| file | purpose |
|---|---|
| `features.py` | per-patient condition-aware feature table (REST/OUT/WING + contrasts) |
| `model.py` | `FlatClassifier`, `TwoStageClassifier` (tuned ET threshold), estimator factory |
| `evaluate.py` | leave-one-patient-out + subject bootstrap CI + permutation test |
| `run.py` | CLI: `python -m pdetn.run --data-root Data` |
| `experiments.ipynb` | **run/compare here** — flat vs two-stage vs deep STFT/CWT/HHT |

## Quick start
```bash
python -m pdetn.run --data-root Data --estimator logreg   # CLI
# or open pdetn/experiments.ipynb and run top-to-bottom
```

## Where it lands (honest, leave-one-patient-out, subject CIs)
- Two-stage tuned (`logreg`): macro-F1 ~0.58, ET-F1 ~0.32 — interpretable, CPU-only.
- Deep STFT (reference): macro-F1 ~0.66, ET-F1 ~0.57 (threshold-tuned).
- On ~16 ET patients most differences fall inside the CIs; the real lever is
  external ET data (PADS, 16→44) — see `reports/track3_external_data.md`.
