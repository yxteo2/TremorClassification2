"""Load precomputed STFT magnitude matrices from the ``stft`` folder.

Folder layout (matches the user's Drive ``ProcessedData/stft``):
    <root>/ProcessedData/stft/<ACTION>/<CLASS>/<class> <subject>_<action> <trial>.txt

Each .txt is comma-separated, shape ``(time_frames, n_sensors * n_freq_bins)``
where the column blocks are sensor-major (lowerarm | hand | upperarm,
each of width ``n_freq_bins``). Default n_freq_bins=65 (one-sided FFT
of a 128-point window) and n_sensors=3.

Subject IDs strip the trailing trial number so that all trials of the
same patient share one group ID and stay together under group-aware
splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tremor.data import CLASS_MAP


CLASS_FOLDER_TO_LETTER = {"N": "N", "PD": "P", "ET": "E"}

_TRIAL_SUFFIX = re.compile(r"\s+\d+$")


@dataclass
class STFTRecording:
    x: np.ndarray  # (features, time_frames)
    y: int
    subject: str
    path: Path


def _parse_subject(stem: str) -> str:
    """``'N 44_OUT 1'`` → ``'N 44_OUT'``."""
    return _TRIAL_SUFFIX.sub("", stem)


def load_stft_recordings(
    root: Path | str,
    action: str,
    feature: str = "stft",
) -> list[STFTRecording]:
    data_dir = Path(root) / "ProcessedData" / feature / action
    if not data_dir.is_dir():
        raise FileNotFoundError(f"STFT directory not found: {data_dir}")

    recordings: list[STFTRecording] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        letter = CLASS_FOLDER_TO_LETTER.get(class_dir.name)
        if letter is None:
            continue
        label = CLASS_MAP[letter]
        for path in sorted(class_dir.glob("*.txt")):
            arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
            if arr.size == 0:
                continue
            x = arr.T  # files are (time, features); model wants (features, time)
            subject = _parse_subject(path.stem)
            recordings.append(STFTRecording(x=x, y=label, subject=subject, path=path))
    return recordings
