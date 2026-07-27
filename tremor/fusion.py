"""Condition-aware fusion for PD/ET/N differential diagnosis.

Clinical premise: PD is a *rest*-tremor disorder, ET an *action/postural* one,
so the discriminative signal lives in the **contrast across conditions**
(REST vs OUT vs WING), not in any single recording. The condition-agnostic
baseline (pool all recordings, classify each alone) throws that contrast away
and, empirically, does not beat single-condition training.

This module builds a per-**patient** model instead: encode each of a patient's
available condition recordings with a *shared* encoder, tag each with a learned
condition embedding, and fuse the set into one per-patient decision with masked
attention (so patients missing a condition are handled without dropping them).

Two fusion modes let us prove the attention earns its keep:
  * ``mean``      — masked average of the raw condition embeddings (no
                    condition tag, no learned weights). The trivial-fusion
                    baseline.
  * ``attention`` — condition-tagged embeddings + masked attention pooling.

The encoder is swappable (:class:`SpecCNNEncoder` is a small pure-torch default
that needs no downloads; a pretrained resnet18 can be supplied where torchvision
and network access are available).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tremor.datasets import TremorDataset

CONDITION_IDS = {"OUT": 0, "REST": 1, "WING": 2}


# --------------------------------------------------------------------------- #
# Dataset: one sample per patient = the set of that patient's condition specs
# --------------------------------------------------------------------------- #
class PatientFusionDataset(torch.utils.data.Dataset):
    """Group a per-recording spectrogram dataset into per-patient sets.

    Wraps a :class:`TremorDataset` (which turns each Recording into a
    ``(F, T)`` spectrogram) and, for a given subset of patients, yields the
    full set of that patient's recordings together with their condition ids
    and single class label.
    """

    def __init__(self, base: TremorDataset, patients: list[str]):
        self.base = base
        # recording index -> subject / condition, from the (un-oversampled) recs
        by_patient: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(base.recs):
            by_patient[r.subject].append(i)
        self.patients = [p for p in patients if p in by_patient]
        self.by_patient = by_patient

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, p_idx: int):
        pid = self.patients[p_idx]
        idxs = self.by_patient[pid]
        specs, conds = [], []
        label = None
        for i in idxs:
            x, y = self.base[i]                     # x: (F, T) float tensor
            specs.append(x.unsqueeze(0))            # -> (1, F, T)  single-channel image
            conds.append(CONDITION_IDS.get(self.base.recs[i].condition, 0))
            label = int(y)
        return specs, conds, label, pid


def collate_patients(batch):
    """Pad variable-size patient sets to the batch max, with a validity mask."""
    max_n = max(len(specs) for specs, *_ in batch)
    B = len(batch)
    F_, T_ = batch[0][0][0].shape[-2:]
    specs_t = torch.zeros(B, max_n, 1, F_, T_)
    conds_t = torch.zeros(B, max_n, dtype=torch.long)
    mask_t = torch.zeros(B, max_n, dtype=torch.bool)
    labels, subjects = [], []
    for b, (specs, conds, label, pid) in enumerate(batch):
        for j, (s, c) in enumerate(zip(specs, conds)):
            specs_t[b, j] = s
            conds_t[b, j] = c
            mask_t[b, j] = True
        labels.append(label)
        subjects.append(pid)
    return specs_t, conds_t, mask_t, torch.tensor(labels), subjects


# --------------------------------------------------------------------------- #
# Encoder (pure-torch default) + condition-aware fusion model
# --------------------------------------------------------------------------- #
class SpecCNNEncoder(nn.Module):
    """Small 2D CNN over a single-channel spectrogram image -> embedding.

    Deliberately compact (no pretraining, few params) because the binding
    constraint is ~16 ET subjects: a heavy encoder overfits instantly under
    LOSO. Global average pooling makes it agnostic to the (F, T) size.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, embed_dim),
        )

    def forward(self, x):                # x: (N, 1, F, T)
        return self.proj(self.pool(self.features(x)))   # (N, embed_dim)


class ResNet18Encoder(nn.Module):
    """ImageNet-pretrained resnet18 feature encoder (1-channel spectrogram).

    NOT runnable in the sandbox used to develop this (no torchvision, and the
    pretrained-weights host is network-blocked). Provided for local runs with
    torchvision + internet. Untested in-container by construction.
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.3,
                 resize_to: int = 96, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as tvm
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError(
                "encoder='resnet18' needs torchvision (pip install torchvision) "
                "and network access to fetch pretrained weights."
            ) from e
        self.resize_to = resize_to
        self.adapter = nn.Conv2d(1, 3, kernel_size=1)
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, embed_dim),
        )

    def forward(self, x):                          # x: (N, 1, F, T)
        x = self.adapter(x)
        x = F.interpolate(x, size=(self.resize_to, self.resize_to),
                          mode="bilinear", align_corners=False)
        return self.proj(self.backbone(x))


def build_encoder(name: str, embed_dim: int, dropout: float,
                  resize_to: int = 96) -> nn.Module:
    if name == "cnn":
        return SpecCNNEncoder(embed_dim, dropout)
    if name == "resnet18":
        return ResNet18Encoder(embed_dim, dropout, resize_to=resize_to)
    raise ValueError(f"unknown encoder {name!r} (use 'cnn' or 'resnet18')")


class ConditionAwareFusion(nn.Module):
    """Shared encoder + masked fusion over a patient's condition recordings."""

    def __init__(self, num_classes: int = 3, embed_dim: int = 128,
                 n_conditions: int = 3, fusion: str = "attention",
                 dropout: float = 0.3, encoder: nn.Module | None = None):
        super().__init__()
        if fusion not in ("attention", "mean"):
            raise ValueError(f"fusion must be attention|mean, got {fusion!r}")
        self.fusion = fusion
        self.encoder = encoder or SpecCNNEncoder(embed_dim, dropout)
        if fusion == "attention":
            self.cond_embed = nn.Embedding(n_conditions, embed_dim)
            self.attn = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, specs, conds, mask):
        # specs: (B, N, 1, F, T)  conds: (B, N)  mask: (B, N) bool
        B, N = specs.shape[:2]
        emb = self.encoder(specs.view(B * N, *specs.shape[2:])).view(B, N, -1)

        if self.fusion == "mean":
            m = mask.unsqueeze(-1).float()
            pooled = (emb * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            emb = emb + self.cond_embed(conds)             # tag with condition
            scores = self.attn(emb).squeeze(-1)            # (B, N)
            scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, N, 1)
            pooled = (emb * w).sum(1)                       # (B, embed_dim)
        return self.head(pooled)
