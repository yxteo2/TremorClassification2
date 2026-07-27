"""Adapter: PADS (Parkinson's Disease Smartwatch) -> tremor.Recording.

PADS (Varghese et al. 2024, PhysioNet DOI 10.13026/m0w9-zx22) is a near-exact
modality match to this project's data: 3-axis **gyroscope (angular velocity)**
at **100 Hz**, standardized seated tasks, and N/PD/ET labels. This adapter maps
it into the same :class:`tremor.data.Recording` stream so it flows through the
existing multi-condition / LOSO / threshold harness (cv_benchmark, fusion_train).

STATUS: SKELETON. PhysioNet blocks automated fetch, so the exact JSON field
names and timeseries column order below could NOT be verified from code. Every
assumption that must be checked against the downloaded files is marked
``VERIFY:``. Download the dataset, confirm the four constants, then this loads.

Design decisions (see reports/track3_external_data.md):
  * Use the **gyroscope** channels only (angular velocity), to match our
    quaternion-derived angular_velocity representation. Accelerometer ignored.
  * PADS is wrist-worn; our data has 3 arm sensors. We harmonise by using our
    ``hand`` sensor only (3 ch) against PADS one-wrist gyroscope (3 ch).
  * Task -> condition map aligns StretchHold->OUT, Relaxed->REST, action->WING.
  * Label map: HC->N, PD->PD, ET->ET; all other differential diagnoses dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tremor.data import CLASS_MAP, Recording

# --------------------------------------------------------------------------- #
# VERIFY these four against the downloaded PADS files, then the rest just works.
# --------------------------------------------------------------------------- #

# VERIFY: column order in movement/timeseries/<id>_<Task>_<Wrist>.txt.
# Documented as 6-axis (accelerometer x/y/z then gyroscope x/y/z). Set the
# indices of the three GYROSCOPE columns here (0-based).
GYRO_COLS = [3, 4, 5]

# VERIFY: in patients/patient_<id>.json, the field holding the diagnosis string
# and the exact strings used for each class.
LABEL_FIELD = "condition"                       # VERIFY field name
LABEL_TO_LETTER = {                             # VERIFY the label strings
    "Healthy": "N", "Control": "N",
    "Parkinson's": "P", "Parkinson": "P",
    "Essential Tremor": "E",
}

# Map PADS task filename tokens -> our condition names (OUT/REST/WING).
TASK_TO_CONDITION = {
    "StretchHold": "OUT",    # arms outstretched / postural
    "Relaxed": "REST",       # rest
    "DrinkGlas": "WING",     # action proxy (choose your action task)
}

SAMPLE_RATE_HZ = 100.0       # documented; kept for callers that need it


def _parse_label(patient_json: Path) -> str | None:
    meta = json.loads(patient_json.read_text())
    raw = str(meta.get(LABEL_FIELD, "")).strip()
    return LABEL_TO_LETTER.get(raw)


def load_pads_recordings(
    root: Path | str,
    conditions: list[str] | None = None,
    wrist: str = "RightWrist",
) -> list[Recording]:
    """Load PADS into Recording objects (gyroscope, one wrist).

    Parameters
    ----------
    root       : PADS dataset root (contains ``movement/`` and ``patients/``).
    conditions : subset of {OUT, REST, WING} to keep (default: all mapped).
    wrist      : which wrist file to use ('RightWrist' or 'LeftWrist').

    Returns Recording(x=(3, T) angular velocity, y, subject=PADS patient id,
    path, condition). Subject id is patient-level, so subject-level LOSO across
    the pooled cohort is leakage-free out of the box.
    """
    root = Path(root)
    ts_dir = root / "movement" / "timeseries"     # VERIFY: path layout
    pat_dir = root / "patients"
    if not ts_dir.is_dir():
        raise FileNotFoundError(f"PADS timeseries dir not found: {ts_dir}")

    keep = set(conditions) if conditions else set(TASK_TO_CONDITION.values())
    want_tasks = {t: c for t, c in TASK_TO_CONDITION.items() if c in keep}

    # Cache patient labels.
    labels: dict[str, str] = {}
    for pj in pat_dir.glob("patient_*.json"):
        pid = pj.stem.split("_")[-1]
        letter = _parse_label(pj)
        if letter is not None:
            labels[pid] = letter

    recordings: list[Recording] = []
    for path in sorted(ts_dir.glob(f"*_{wrist}.txt")):
        # filename: <id>_<Task>_<Wrist>.txt
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        pid, task = parts[0], parts[1]
        if task not in want_tasks:
            continue
        letter = labels.get(pid)
        if letter is None:                        # non-ET differential -> dropped
            continue
        arr = pd.read_csv(path, sep=None, engine="python", header=None).to_numpy(
            dtype=np.float32
        )
        if arr.ndim != 2 or arr.shape[1] <= max(GYRO_COLS):
            raise ValueError(f"{path}: unexpected shape {arr.shape}; check GYRO_COLS")
        x = arr[:, GYRO_COLS].T                    # (3, T) angular velocity
        recordings.append(Recording(
            x=x, y=CLASS_MAP[letter], subject=f"PADS_{pid}",
            path=path, condition=want_tasks[task],
        ))
    return recordings
