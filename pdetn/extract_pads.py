#!/usr/bin/env python3
"""Extract ONLY the StretchHold (arms-outstretched) recordings from PADS.

StretchHold is PADS's arms-outstretched postural task = your OUT condition.
This pulls just those recordings, keeps the gyroscope (angular velocity) axes,
maps the diagnosis to N/PD/ET (skipping other differential diagnoses), and saves
one array per recording plus a manifest.

Because the exact PADS file layout could not be verified when writing this,
run --inspect FIRST to confirm the format, then adjust the two VERIFY blocks if
needed and run the extraction.

    # 1. confirm the format
    python -m pdetn.extract_pads --pads-root PADS --inspect
    # 2. extract StretchHold -> ./pads_stretchhold/
    python -m pdetn.extract_pads --pads-root PADS --out pads_stretchhold
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

TASK = "StretchHold"

# ---- VERIFY 1: gyroscope columns in each timeseries file --------------------
# PADS records 6 axes: accelerometer (x,y,z) then gyroscope (x,y,z). Gyro =
# angular velocity, which matches your data. Adjust if --inspect shows otherwise.
GYRO_COLS = [3, 4, 5]

# ---- VERIFY 2: how the diagnosis is stored in patients/patient_<id>.json -----
# We search these keys (and nested dicts) for a diagnosis string, then normalise.
LABEL_KEYS = ["condition", "disease", "diagnosis", "group", "label",
              "study_group", "cohort", "class"]
# substring -> N / PD / ET. Anything not matching is skipped (other disorders).
LABEL_MAP = {
    "healthy": "N", "control": "N", "hc": "N",
    "parkinson": "PD", "pd": "PD",
    "essential tremor": "ET", "essential-tremor": "ET", "et": "ET",
}


def _norm(s: str) -> str:
    return str(s).strip().lower()


def find_label(meta):
    """Return (N|PD|ET, raw_string) or (None, None) by searching the JSON."""
    def search(obj):
        if isinstance(obj, dict):
            for k in LABEL_KEYS:
                if k in obj and isinstance(obj[k], (str, int)):
                    v = _norm(obj[k])
                    for key, lab in LABEL_MAP.items():
                        if key in v:
                            return lab, obj[k]
            for v in obj.values():
                r = search(v)
                if r[0]:
                    return r
        elif isinstance(obj, str):
            v = _norm(obj)
            for key, lab in LABEL_MAP.items():
                if key in v:
                    return lab, obj
        return None, None
    return search(meta)


def load_timeseries(path: Path) -> np.ndarray:
    """Load a PADS timeseries file as (T, n_axes). Handles CSV / whitespace / JSON."""
    text = path.read_text().strip()
    if text.startswith("[") or text.startswith("{"):     # JSON array
        arr = np.asarray(json.loads(text), dtype=float)
    else:
        delim = "," if "," in text.splitlines()[0] else None
        arr = np.loadtxt(path, delimiter=delim)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def patient_id_from_name(stem: str) -> str:
    # filenames look like "<id>_StretchHold_RightWrist"
    return stem.split("_")[0]


def find_dirs(root: Path):
    ts = next(iter(root.rglob("timeseries")), None) or root
    pat = next(iter(root.rglob("patients")), None)
    return ts, pat


def load_patient_labels(pat_dir: Path) -> dict[str, tuple[str, str]]:
    labels = {}
    for pj in (pat_dir.glob("patient_*.json") if pat_dir else []):
        pid = pj.stem.split("_")[-1].lstrip("0") or "0"
        try:
            meta = json.loads(pj.read_text())
        except Exception:
            continue
        lab, raw = find_label(meta)
        if lab:
            labels[pid] = (lab, raw)
            labels[pj.stem.split("_")[-1]] = (lab, raw)   # also keep zero-padded key
    return labels


def inspect(root: Path):
    ts_dir, pat_dir = find_dirs(root)
    print(f"timeseries dir: {ts_dir}\npatients dir:   {pat_dir}\n")
    sample = next(iter(ts_dir.rglob(f"*{TASK}*.txt")), None) or next(iter(ts_dir.rglob("*.txt")), None)
    if sample:
        print(f"--- sample timeseries: {sample.name} ---")
        for line in sample.read_text().splitlines()[:3]:
            print("   ", line[:120])
        arr = load_timeseries(sample)
        print(f"   shape (T, axes) = {arr.shape}  -> using gyro cols {GYRO_COLS}\n")
    pj = next(iter(pat_dir.glob("patient_*.json")), None) if pat_dir else None
    if pj:
        meta = json.loads(pj.read_text())
        print(f"--- sample patient json: {pj.name} ---")
        print("   top-level keys:", list(meta.keys()))
        lab, raw = find_label(meta)
        print(f"   detected label: {lab}  (from value {raw!r})")


def extract(root: Path, out: Path, wrist: str, gyro_only: bool):
    ts_dir, pat_dir = find_dirs(root)
    labels = load_patient_labels(pat_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {"N": 0, "PD": 0, "ET": 0}
    skipped = 0
    for f in sorted(ts_dir.rglob(f"*{TASK}*")):
        if f.suffix.lower() not in (".txt", ".csv", ".json"):
            continue
        if wrist != "both" and wrist.lower() not in f.stem.lower():
            continue
        pid = patient_id_from_name(f.stem)
        lab = labels.get(pid) or labels.get(pid.zfill(3))
        if lab is None:            # unknown or non-N/PD/ET diagnosis
            skipped += 1
            continue
        cls = lab[0]
        arr = load_timeseries(f)
        if gyro_only:
            if arr.shape[1] <= max(GYRO_COLS):
                raise SystemExit(f"{f}: has {arr.shape[1]} cols; GYRO_COLS={GYRO_COLS} invalid — check --inspect")
            arr = arr[:, GYRO_COLS]
        wr = "RightWrist" if "right" in f.stem.lower() else ("LeftWrist" if "left" in f.stem.lower() else "NA")
        outfile = out / f"{cls}_{pid}_{wr}.npy"
        np.save(outfile, arr.astype(np.float32))
        manifest.append({"file": outfile.name, "patient": pid, "class": cls,
                         "wrist": wr, "n_samples": arr.shape[0], "n_channels": arr.shape[1],
                         "raw_label": lab[1]})
        counts[cls] += 1

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "patient", "class", "wrist",
                                           "n_samples", "n_channels", "raw_label"])
        w.writeheader(); w.writerows(manifest)
    n_pat = len({m["patient"] for m in manifest})
    print(f"extracted {len(manifest)} StretchHold recordings, {n_pat} patients")
    print(f"  per class: {counts}   (skipped {skipped} non-N/PD/ET or unlabeled)")
    print(f"  saved -> {out}/ (one .npy per recording + manifest.csv)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pads-root", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("pads_stretchhold"))
    p.add_argument("--wrist", default="both", choices=["both", "RightWrist", "LeftWrist"])
    p.add_argument("--gyro-only", action="store_true", default=True,
                   help="keep only gyroscope (angular velocity) axes (default).")
    p.add_argument("--all-axes", dest="gyro_only", action="store_false",
                   help="keep all 6 axes (accel + gyro).")
    p.add_argument("--inspect", action="store_true",
                   help="print file-format structure and exit (run this first).")
    args = p.parse_args()

    if args.inspect:
        inspect(args.pads_root)
    else:
        extract(args.pads_root, args.out, args.wrist, args.gyro_only)


if __name__ == "__main__":
    main()
