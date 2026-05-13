"""Group-aware splits at the SUBJECT level.

CRITICAL: held-out partitions are selected by subject, not by recording.
Multiple trials from the same subject must never be split across the
train / val / test boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def subject_level_split(
    subjects: Sequence[str],
    labels: Sequence[int],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 0,
) -> Split:
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    n = len(subjects)

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(outer.split(np.zeros(n), labels, groups=subjects))

    inner_subjects = subjects[trainval_idx]
    inner_labels = labels[trainval_idx]
    val_frac = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + 1)
    inner_train_local, inner_val_local = next(
        inner.split(np.zeros(len(trainval_idx)), inner_labels, groups=inner_subjects)
    )

    train_idx = trainval_idx[inner_train_local]
    val_idx = trainval_idx[inner_val_local]

    _assert_disjoint_subjects(subjects, train_idx, val_idx, test_idx)
    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _assert_disjoint_subjects(subjects: np.ndarray, *idx_groups: np.ndarray) -> None:
    seen: set[str] = set()
    for idx in idx_groups:
        s = set(subjects[idx].tolist())
        if s & seen:
            raise AssertionError("subject leakage between splits")
        seen |= s
