# Pretrained backbones (ResNet18 / AST / ViT) on PD-vs-ET

The idea: a pretrained feature extractor with a small trainable head is the
low-parameter regime, which is the only regime that works at 16 ET. Sound
reasoning — the winning model here *is* a linear classifier on 10 features.

`freeze_backbone`, `pretrained` and `resize_to` are now plumbed through
`train_bilstm` and `tfbench.deep.train_one`, and the trainer prints the
trainable/total parameter split for transfer models.

## What could be tested here

**ImageNet weights cannot be downloaded in this environment** — torchvision's
fetch is blocked by the proxy (`403 Forbidden`). So `pretrained=True` is
untested; only the random-weight control ran.

| config | trainable params | bal-acc | AUC | precision |
|---|---|---|---|---|
| resnet18 **frozen** backbone, random weights | **1,722** / 11,178,234 (0.0 %) | 0.522 | **0.446** | 0.184 |
| resnet18 unfrozen, random weights | 11.2 M | 0.471 | **0.485** | 0.157 |
| BiLSTM, best of 6 configs | ~1e5 | 0.513 | 0.517 | 0.180 |
| **classical logreg, 10 descriptors** | **11** | **0.730** | **0.729** | **0.393** |

## What this does and does not show

**Does show:** the low-parameter argument alone is not enough. The frozen
ResNet trains only **1,722 parameters** — a genuinely small head — and still
sits at chance (AUC 0.446). So "freeze the backbone so it can't overfit" does
not by itself produce a working model. The frozen backbone here is a *random*
projection, so this is a random-features baseline, and random features carry
nothing.

**Does not show:** whether *pretrained* features help. That is the whole
question and it is untested. The one thing that would make this approach work —
ImageNet or AudioSet features containing something relevant to a 6 Hz
oscillation — is exactly what could not be fetched.

## To test it properly, run locally

```python
from tfbench.deep import run_cv
from common.quaternion_data import load_quaternion_recordings
from common.data import Recording
recs = load_quaternion_recordings("Data", action="REST", mode="angular_velocity")
recs = [Recording(x=r.x[3:6], y=r.y, subject=r.subject, path=r.path,
                  condition=r.condition) for r in recs]
Pp, yp, pats = run_cv(recs, "stft512", "resnet18", axis="PD_vs_ET", n_splits=4,
                      seed=0, pretrained=True, freeze_backbone=True,
                      oversample_to=60, epochs=40)
```

**The bar is AUC 0.729** — what logistic regression on 10 descriptors achieves.
Anything below that is not worth the dependency.

## On WideResNet and ViT

Bigger backbones do not address the constraint, and make two things worse:

* **Domain gap.** ImageNet is natural images; AST is AudioSet at 16 kHz with mel
  bins from ~50 Hz. Tremor is **3–15 Hz** — orders of magnitude below anything
  either model saw in training. Scale is the problem, and a larger model trained
  on the same mismatched domain does not reduce it.
* **Head size grows with backbone width.** A frozen ViT-B gives a 768-d
  embedding; a linear head on it is ~1.5 k parameters for 2 classes, against 16
  ET subjects. That is the same overfitting regime that made 30 multi-sensor
  features (−0.107) and 23 temporal features (−0.224) significantly *worse*.

If a frozen ResNet18 with pretrained weights cannot beat 0.729, a larger
backbone will not. Test the cheapest one first, and stop if it fails.

## Where this sits among the alternatives tried

| route | ET precision |
|---|---|
| **classical logreg, 10 descriptors** | **0.393** |
| BiLSTM, best of 6 configs | 0.180 |
| frozen ResNet18 (random weights) | 0.184 |
| pool all three cohorts (16 → 50 ET) | 0.163 |
| frozen ResNet18 (**pretrained**) | **untested — run locally** |
