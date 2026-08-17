"""Load IMU quaternion recordings.

Folder layout (the user keeps the data outside ProcessedData):
    <root>/<feature>/<ACTION>/<CLASS>/<class> <subject>_<action> <trial>.txt
    e.g.  Data/raw_quaternion/OUT/PD/PD 14_OUT.txt

Each .txt is comma-separated, shape ``(T, 12)`` — 3 sensors x 4 unit
quaternion components, scalar-last ``(x, y, z, w)``, sampled at fs=100 Hz.

The loader converts each file into a :class:`common.data.Recording`
with ``x`` shape ``(channels, time)`` so it can flow into the same
``TremorDataset`` -> ``apply_stft`` -> model pipeline as the raw
amplitude data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import CLASS_MAP, Recording
from signal_processing.quaternion import process_quaternion_data


CLASS_FOLDER_TO_LETTER = {"N": "N", "PD": "P", "ET": "E"}

_TRIAL_SUFFIX = re.compile(r"[_\s]+\d+$")


def load_quaternion_recordings(
    root: Path | str,
    action: str,
    fs: float = 100.0,
    mode: str = "angular_velocity",
    convention: str = "xyzw",
    feature: str = "raw_quaternion",
    n_sensors: int = 3,
) -> list[Recording]:
    # Two layouts are supported: <root>/<feature>/<ACTION>/  (user's local
    # convention) or <root>/ProcessedData/<feature>/<ACTION>/ (older
    # processed-data convention). Use the first one that exists.
    candidates = [
        Path(root) / feature / action,
        Path(root) / "ProcessedData" / feature / action,
    ]
    data_dir = next((p for p in candidates if p.is_dir()), None)
    if data_dir is None:
        raise FileNotFoundError(
            f"Quaternion directory not found. Tried: "
            f"{', '.join(str(p) for p in candidates)}"
        )

    recordings: list[Recording] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        letter = CLASS_FOLDER_TO_LETTER.get(class_dir.name)
        if letter is None:
            continue
        label = CLASS_MAP[letter]
        for path in sorted(class_dir.glob("*.txt")):
            df = pd.read_csv(path, sep=None, engine="python", header=None)
            Q12 = df.to_numpy(dtype=np.float32)
            if Q12.size == 0:
                continue
            expected_cols = n_sensors * 4
            if Q12.ndim != 2 or Q12.shape[1] != expected_cols:
                raise ValueError(
                    f"{path}: expected (T, {expected_cols}) = "
                    f"{n_sensors} sensors x 4 quaternion components, "
                    f"got shape {Q12.shape}"
                )
            try:
                x = process_quaternion_data(
                    Q12, fs=fs, mode=mode, convention=convention,
                    n_sensors=n_sensors,
                )
            except ValueError as e:
                raise ValueError(f"failed to process {path}: {e}") from e
            subject = _TRIAL_SUFFIX.sub("", path.stem)
            recordings.append(
                Recording(x=x, y=label, subject=subject, path=path, condition=action)
            )
    return recordings


def load_quaternion_recordings_multi(
    root: Path | str,
    actions: list[str],
    **kwargs,
) -> list[Recording]:
    """Load several actions and key subjects at the PATIENT level.

    The single-action loader keys subjects as ``"ET 10_OUT"`` — the action is
    baked into the id. That is fine within one action, but pooling actions that
    way would place the SAME patient's OUT and REST recordings in different
    LOSO folds, i.e. subject leakage. Here we strip the trailing ``_<ACTION>``
    so every recording of patient ``"ET 10"`` shares one group id across
    conditions, and tag each recording with its ``condition``.
    """
    recordings: list[Recording] = []
    for action in actions:
        recs = load_quaternion_recordings(root, action=action, **kwargs)
        suffix = re.compile(rf"_{re.escape(action)}$")
        for r in recs:
            r.subject = suffix.sub("", r.subject)
            r.condition = action
        recordings.extend(recs)
    return recordings
