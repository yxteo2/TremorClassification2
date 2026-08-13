# ResNet18 / WideResNet / ViT on the kinetic task

`resnet50`, `wide_resnet50_2` and `vit_b_16` are now in
`tremor.models.MODELS` via `TorchvisionBackbone`, which generalises the existing
`ResNet18Pretrained` front-end (channel adapter → resize → swapped head).
`vit_b_16` forces `resize_to=224`; its patch embedding is fixed at that size.

Frozen-backbone parameter counts, 2-class head:

| backbone | trainable | total | trainable % |
|---|---|---|---|
| resnet18 | 1,038 | 11.2 M | 0.009 % |
| wide_resnet50_2 | 4,110 | 66.8 M | 0.006 % |
| vit_b_16 | 1,550 | 85.8 M | 0.002 % |

## Tested on DRINK — the task where the signal is

NewData DRINK, lower_arm, 29 N / 23 PD / 6 ET, PD-vs-ET, patient-level LOSO.
**ImageNet weights cannot be downloaded here** (proxy 403), so these run with
random weights — a frozen random backbone is a random-projection baseline.

| model | weights | bal-acc | AUC | precision | recall |
|---|---|---|---|---|---|
| resnet18 (11.2 M) | random | 0.525 | 0.522 | 0.217 | 0.833 |
| wide_resnet50_2 (66.8 M) | random | 0.486 | 0.467 | 0.201 | 0.667 |
| vit_b_16 (85.8 M) | random | 0.549 | 0.540 | 0.231 | 0.750 |
| **logreg, 10 descriptors (11 params)** | — | **0.790** | **0.812** | **0.667** | 0.667 |
| **MLP h=16 on descriptors (~700)** | — | 0.728 | **0.942** | **0.750** | 0.500 |

**All three backbones sit at chance** (AUC 0.467–0.540) on the one task where
the classical model reaches 0.812 and a ~700-parameter MLP reaches 0.942.
Ranking them by size — resnet18 0.522, wide_resnet50_2 0.467, vit_b_16 0.540 —
shows no relationship with capacity at all: 85.8 M parameters buys the same
chance performance as 11.2 M.

The contrast with `reports/small_networks.md` is the point. A 700-parameter MLP
on 10 descriptors reaches AUC 0.942 on this same task and split. The difference
is not capacity, architecture family, or attention — it is **what the model is
asked to learn from**. Given descriptors, a tiny network wins; given raw
spectrograms, an 85 M-parameter transformer cannot get off chance.

## What this does and does not establish

**Does:** the architecture contributes nothing on its own. A frozen random
backbone is a random projection, and random projections of these spectrograms
carry no PD/ET information. Freezing keeps the trainable count tiny (1–4 k) but
that is not sufficient — the *features* have to be informative.

**Does not:** whether ImageNet features help. Still untested, still the only
open question about this whole family, and still blocked here.

## Recommended order when running locally

1. **`resnet18`, `pretrained=True`, `freeze_backbone=True`, on DRINK.**
   Cheapest, and the bar is **AUC 0.812** from 10 hand-computed descriptors.
2. Only if that clears the bar, try `wide_resnet50_2`, then `vit_b_16`.

The reason for that order is not caution for its own sake: the domain gap does
not shrink with model size. ImageNet is natural images; tremor is a 3–15 Hz
oscillation rendered as a spectrogram. A larger model trained on the same
mismatched domain has the same mismatch, and a wider frozen embedding means a
larger head against **6 ET subjects** on this task. Two feature families of 23
and 30 dimensions already made results *significantly worse* at 16 ET
(`reports/temporal_features.md`, `reports/sensor_combination_rest.md`).

```python
from tfbench.deep import run_cv
from pdetn.load_2025 import load_2025_all
from tremor.data import Recording
recs = load_2025_all(conditions=("DRINK",))
recs = [Recording(x=r.x[3:6], y=r.y, subject=r.subject, path=r.path,
                  condition=r.condition) for r in recs]
Pp, yp, pats = run_cv(recs, "stft512", "resnet18", axis="PD_vs_ET", n_splits=3,
                      seed=0, pretrained=True, freeze_backbone=True,
                      oversample_to=40, epochs=25)
```

## The higher-value experiment

Extracting **PADS DrinkGlas and TouchNose** (938 files each, 28 ET) tests
whether the kinetic finding replicates on an independent cohort. That question
matters more than which backbone to use: if AUC 0.81–0.83 does not survive on
28 ET, no architecture makes it real. See `reports/kinetic_tasks_2025.md`.
