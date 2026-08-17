"""PADS loader.

Extracted from the former ``common.loaders``; the rest of that module
(protocol drivers, feature builders, the dataset-identity probe) was
superseded by ``common.cohorts`` and ``metrics.merged``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.data import Recording


def load_pads_extracted(folder, strict=True, task=None):
    """Load the StretchHold data extracted by common.extract_pads.

    The filename class token is NOT trusted. An earlier version of
    ``extract_pads`` mapped diagnoses by substring, so the bare token "et"
    matched "etiology", "asymmetric", "Retrocollis" and "hypokinetic": 13 of 41
    files named ``ET_*`` are not Essential Tremor, including parkinsonian cases
    (a hypokinetic-rigid syndrome, a Lewy-Body dementia). 20 ``PD_*`` files are
    Atypical Parkinsonism, which PADS treats as a separate group.

    With ``strict=True`` (default) the class is re-derived from the manifest's
    ``raw_label`` by EXACT match, and every ambiguous or mixed diagnosis is
    dropped. That gives N=79 / PD=276 / ET=28, and the ET count then agrees with
    the published PADS cohort (Varghese 2024: 28 ET).

    ``strict=False`` reproduces the old contaminated behaviour; it exists only
    to re-derive the superseded numbers and should not be used for new results.

    ``task`` filters by PADS task substring (e.g. ``"Relaxed"`` matches both
    Relaxed1 and Relaxed2, ``"StretchHold"`` the postural task). ``None`` loads
    everything in the folder, which is correct for a single-task folder.
    """
    import csv
    import re

    cmap = {"N": 0, "PD": 1, "ET": 2}
    # Accept several folders. PADS repetitions (Relaxed1, Relaxed2) must be
    # extracted separately because each run rewrites manifest.csv, so the
    # natural layout is one folder per repetition -- load them together here.
    if isinstance(folder, (list, tuple)):
        out = []
        for one in folder:
            out.extend(load_pads_extracted(one, strict=strict, task=task))
        return out
    folder = Path(folder)
    manifest = folder / "manifest.csv"
    exact = {"healthy": "N", "parkinson's": "PD", "essential tremor": "ET"}

    true_cls, file_task = {}, {}
    if strict:
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} is required for strict labelling; pass strict=False "
                "to fall back to the (contaminated) filename labels.")
        for row in csv.DictReader(manifest.open()):
            lab = exact.get(row["raw_label"].strip().lower())
            if lab:
                true_cls[row["file"]] = lab
            file_task[row["file"]] = (row.get("task") or "").strip()

    recs = []
    for f in sorted(folder.glob("*.txt")):
        # two layouts: legacy <cls>_<pid>_<wrist> and current
        # <cls>_<pid>_<task>_<wrist> (the task token was added so repetitions
        # like Relaxed1/Relaxed2 stop overwriting each other)
        m = re.match(r"(N|PD|ET)_(\d+)_(\w+)", f.stem)
        if not m:
            continue
        cls, pid, _ = m.groups()
        if task is not None:
            parts = f.stem.split("_")
            t = file_task.get(f.name) or (parts[2] if len(parts) > 3 else "")
            if task.lower() not in t.lower():
                continue
        if strict:
            cls = true_cls.get(f.name)
            if cls is None:            # ambiguous / non-N-PD-ET diagnosis
                continue
        x = np.loadtxt(f, delimiter=",", ndmin=2).T          # (3, T) gyro
        recs.append(Recording(x=x.astype(np.float32), y=cmap[cls],
                             subject=f"PADS_{pid}", path=f, condition="OUT"))
    return recs
