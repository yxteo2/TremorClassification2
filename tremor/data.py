"""Load tremor recordings from amplitude/time-domain folders.

The user's Drive has the time-domain (`filtered_amplitudes` etc.)
data at the **top level**:
    <root>/filtered_amplitudes/<ACTION>/<file>.txt

The processed-feature folders (`stft`, `raw_quaternion`, …) live
under `ProcessedData/`:
    <root>/ProcessedData/<feature>/<ACTION>/<CLASS>/<file>.txt

To accept both, ``load_recordings`` searches several candidate roots
in order and uses the first that exists.

Each .txt contains an amplitudes table where columns are (L, H, U)
sensor channels and rows are timesteps. The class label is encoded in
the **leading character** of the filename — N (Normal), P (PD), E
(ET) — and that's how this loader assigns labels (class subfolders,
if present, are walked transparently via ``rglob``).

Subject IDs strip the trailing trial number so all trials of one
patient share a group ID and stay together under group-aware splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_MAP = {"N": 0, "P": 1, "E": 2}
CLASS_NAMES = ("N", "PD", "ET")

_TRIAL_SUFFIX = re.compile(r"[_\s]+\d+$")


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


def _resolve_data_dir(root: Path, feature: str, action: str) -> Path:
    """Find the first matching layout among the conventions we support."""
    candidates = [
        root / feature / action,                          # top-level, e.g. filtered_amplitudes/
        root / "ProcessedData" / feature / action,        # processed-feature, e.g. stft/, raw_quaternion/
        root / "ProcessedData" / "raw data" / feature / action,  # legacy MATLAB layout
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "No data directory found for feature='{f}' action='{a}'. "
        "Tried: {c}".format(f=feature, a=action,
                            c=", ".join(str(p) for p in candidates))
    )


def load_recordings(
    root: Path | str,
    feature: str = "filtered_amplitudes",
    action: str = "DRINK",
) -> list[Recording]:
    """Load all amplitude recordings for one (feature, action) combination.

    Walks ``data_dir`` recursively, so class subfolders (if present)
    are handled transparently. Each file's class label is parsed from
    its filename's leading letter; rows are interpreted as timesteps
    and columns as sensor channels.
    """
    data_dir = _resolve_data_dir(Path(root), feature, action)

    recordings: list[Recording] = []
    for path in sorted(data_dir.rglob("*.txt")):
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        arr = df.to_numpy(dtype=np.float32).T
        if arr.size == 0:
            continue
        try:
            subject, label = _parse_subject_and_class(path.name)
        except ValueError:
            # Skip files whose leading character isn't a known class
            continue
        recordings.append(Recording(x=arr, y=label, subject=subject, path=path))
    return recordings


def filter_by_length(
    recs: list[Recording], min_len: int, max_len: int
) -> list[Recording]:
    return [r for r in recs if min_len <= r.x.shape[1] <= max_len]
