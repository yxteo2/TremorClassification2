"""Loss functions for imbalanced tremor classification.

The training fold may have several times more N samples than PD or ET.
Two complementary tools help:

* Inverse-frequency class weights  (``compute_class_weights``):
  scale each example's contribution to the loss by ``1 / freq_i``,
  normalised to mean 1 across classes.

* Focal loss  (``FocalLossMulticlass``):
  multiplies CE by ``(1 - p_t)^gamma``, down-weighting easy examples
  (high-confidence correct predictions) and focusing gradient on
  hard / misclassified ones. With ``alpha=class_weights`` it also
  handles class imbalance.

This file mirrors the implementation in the user's Drive
``training.py`` so behaviour is consistent across pipelines.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def compute_class_weights(
    labels: Sequence[int], num_classes: int,
) -> torch.Tensor:
    """Inverse-frequency class weights, normalised over non-empty classes.

    ``weight_i = (1 / count_i) / mean_{j: count_j > 0}(1 / count_j)``

    Classes with zero training samples get a weight of 0 (cannot occur in
    the loss anyway). The mean over non-empty classes is 1, so the
    weighted-loss magnitude stays comparable to plain cross-entropy and
    the relative weighting is purely a function of sample size:
    halving a class's count doubles its weight.
    """
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    if nonzero.any():
        inv = 1.0 / counts[nonzero]
        weights[nonzero] = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLossMulticlass(nn.Module):
    """Multi-class focal loss with optional per-class alpha and label smoothing.

    Args:
        alpha: ``None``, scalar, or 1-D tensor of length ``num_classes``.
            When a tensor, acts as per-class weighting (combines well with
            ``compute_class_weights``).
        gamma: focusing parameter ``>= 0``. ``gamma=0`` reduces to weighted
            cross-entropy. Common range: 1.0 - 2.5.
        reduction: ``"mean"``, ``"sum"`` or ``"none"``.
        ignore_index: target value to skip.
        label_smoothing: float in [0, 1). Spreads probability mass evenly
            across the non-target classes, regularising overconfident
            predictions.
    """

    def __init__(
        self,
        alpha: Optional[Union[torch.Tensor, Sequence[float], float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction {reduction!r}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")

        self.gamma = float(gamma)
        self.reduction = reduction
        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)

        if alpha is None:
            self.register_buffer("alpha", torch.zeros(0), persistent=False)
            self._alpha_set = False
        else:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor([float(alpha)], dtype=torch.float32)
            elif not isinstance(alpha, torch.Tensor):
                raise TypeError(f"unsupported alpha type {type(alpha)}")
            self.register_buffer("alpha", alpha.float(), persistent=False)
            self._alpha_set = True

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"expected (N, C) logits, got shape {tuple(logits.shape)}")
        num_classes = logits.shape[1]
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid].long()
        if logits.numel() == 0:
            return logits.new_tensor(0.0)

        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-8)

        if self.label_smoothing > 0:
            eps = self.label_smoothing
            ce = -(1.0 - eps) * log_pt - eps * log_probs.mean(dim=1)
        else:
            ce = -log_pt

        focal_weight = (1.0 - pt) ** self.gamma

        if self._alpha_set:
            alpha = self.alpha
            if alpha.numel() == 1:
                alpha_t = torch.full_like(pt, float(alpha.item()))
            elif alpha.numel() == num_classes:
                alpha_t = alpha.to(device=pt.device, dtype=pt.dtype)[targets]
            else:
                raise ValueError(
                    f"alpha length {alpha.numel()} must be 1 or {num_classes}"
                )
        else:
            alpha_t = torch.ones_like(pt)

        loss = alpha_t * focal_weight * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(
    loss_type: str,
    train_labels: Sequence[int],
    num_classes: int,
    focal_gamma: float = 1.5,
    label_smoothing: float = 0.0,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Construct one of ``focal``, ``weighted_ce`` or ``ce``.

    Class weights are computed automatically from ``train_labels`` for
    the ``focal`` and ``weighted_ce`` variants.
    """
    if loss_type not in ("focal", "weighted_ce", "ce"):
        raise ValueError(f"unknown loss_type {loss_type!r}")

    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    weights = compute_class_weights(train_labels, num_classes).to(device)

    if loss_type == "focal":
        return FocalLossMulticlass(
            alpha=weights, gamma=focal_gamma,
            label_smoothing=label_smoothing, reduction="mean",
        )

    return nn.CrossEntropyLoss(
        weight=weights, label_smoothing=label_smoothing,
    )
