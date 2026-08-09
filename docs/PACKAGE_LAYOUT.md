# Package layout

Two packages, one direction of dependency: **`tfbench` → `pdetn`**. `pdetn`
never imports `tfbench`.

## `pdetn/` — data layer and earlier exploration

| module | role |
|---|---|
| `crossdataset.py` | PADS loading (`load_pads_extracted`, strict labels), local single-sensor loading |
| `extract_pads.py` | PADS extraction CLI (exact diagnosis + exact task matching) |
| `load_2025.py` | NewData loading, with task-epoch segmentation on by default |
| `deep_crossdataset.py` | **the canonical training loop** (`train_bilstm`, any architecture) |
| `deep_eval.py` | patient-grouped CV + fine-tuning on top of that loop |
| `model.py`, `evaluate.py` | two-stage classical classifier + LOSO evaluation |
| `separability.py`, `features.py`, `signal_features.py`, `spatial_features.py` | earlier feature families |
| `quaternion_tf.py`, `quaternion_repr.py` | orientation/orbit-geometry work (headline finding **retracted**, see `reports/handedness_does_not_survive.md`) |
| `extra_transforms.py` | VMD helper used by `tfbench.transforms` |

## `tfbench/` — the current benchmark pipeline

| module | role |
|---|---|
| `transforms.py` | 12 TF methods → (freqs, power). All power-scaled and Parseval-checked. |
| `descriptors.py` | max/mean/median frequency + 7 more, bin-width integrated |
| `benchmark.py` | stage 1: BH-corrected screen + LOSO ranking with paired CIs |
| `cache.py` | persists the 12 descriptor tables (~8 min to rebuild) |
| `merged.py` | the chosen design: merge 2015+NewData, validate on PADS |
| `frequency_report.py` | the mean/max frequency report across all cohorts |
| `deep.py` | stage 2: method × architecture grid |

Notebooks: `01_signal_processing_benchmark`, `02_deep_model_comparison`,
`03_cohort_comparison`.

## Deduplication done

* **One training loop.** `tfbench/deep.py` carried an independent copy of the
  Adam + focal-loss + early-stopping loop already in
  `pdetn/deep_crossdataset.py::train_bilstm`. A fix to either had to be made
  twice and could silently diverge. `train_bilstm` gained an `arch` parameter
  (any name in `tremor.models.MODELS`) and `tfbench.deep.train_one` is now a
  thin adapter onto it. 174 → 144 lines.
* **`predict_logits` was not TFD-aware** — it hardcoded the STFT defaults, so a
  model trained on CWT was evaluated on STFT images with no error. Found while
  merging the two loops; now takes the TFD parameters.
* **`tfbench.cache` was orphaned.** Wired into `build_all(cache=...)`.
* **Deleted as unreferenced** (no code, notebook or import references):
  `dig_deeper.py`, `inspect_pads_extracted.py`, `pads_deep_experiment.py`,
  `decomposition_sweep.py`, `nonlinear_features.py`.

## Remaining overlap, deliberately kept

`patient_table` and `compare` exist in both packages with the same names but
different jobs — `tfbench` operates on the 10 frequency descriptors,
`pdetn.quaternion_repr` on the orbit-geometry blocks. They are not
interchangeable. `rank_methods` likewise: `tfbench.benchmark`'s is the current
paired-CI version; `pdetn.separability`'s is the older Fisher/silhouette
ranking retained because published reports cite its numbers.
