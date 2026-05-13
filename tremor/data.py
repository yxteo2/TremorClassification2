"""Load tremor recordings from the ProcessedData directory.

File layout mirrors the original MATLAB project:
    <root>/ProcessedData/raw data/<feature>/<action>/*/*.txt

Each .txt contains an amplitudes table where columns are (L, H, U) sensor
channels and rows are timesteps. The leading character of the file name
encodes the class: N (Normal), P (PD), E (ET). The trailing underscore-
delimited integer (e.g. ``_1``, ``_2``) is a TRIAL index for the same
subject. To prevent subject-level leakage the subject id is the file
stem with the trial suffix stripped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_MAP = {"N": 0, "P": 1, "E": 2}
CLASS_NAMES = ("N", "PD", "ET")

_TRIAL_SUFFIX = re.compile(r"_\d+$")


@dataclass
class Recording:
    x: np.ndarray  # (channels, time)
    y: int
    subject: str
    path: Path


def _parse_subject_and_class(filename: str) -> tuple[str, int]:
    stem = Path(filename).stem
    leading = stem[0]
    if leading not in CLASS_MAP:
        raise ValueError(f"Unknown class letter in file: {filename}")
    subject = _TRIAL_SUFFIX.sub("", stem)
    return subject, CLASS_MAP[leading]


def load_recordings(
    root: Path | str,
    feature: str = "filtered_amplitudes",
    action: str = "DRINK",
) -> list[Recording]:
    data_dir = Path(root) / "ProcessedData" / "raw data" / feature / action
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    recordings: list[Recording] = []
    for path in sorted(data_dir.rglob("*.txt")):
        df = pd.read_csv(path, sep=None, engine="python", header=0)
        arr = df.to_numpy(dtype=np.float32).T
        if arr.size == 0:
            continue
        subject, label = _parse_subject_and_class(path.name)
        recordings.append(Recording(x=arr, y=label, subject=subject, path=path))
    return recordings


def filter_by_length(
    recs: list[Recording], min_len: int, max_len: int
) -> list[Recording]:
    return [r for r in recs if min_len <= r.x.shape[1] <= max_len]
