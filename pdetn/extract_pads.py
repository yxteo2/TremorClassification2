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

# Task tokens as they appear in the ACTUAL filenames (<id>_<Task>_<Wrist>.txt),
# confirmed by --list-tasks against a real download: 938 files each = 469
# patients x 2 wrists.
#
# NOTE these differ from the names in PADS's own run_preprocessing.py, which
# uses 'Relaxed1'/'Relaxed2'/'Entrainment1'/'Entrainment2'. On disk there is a
# single 'Relaxed' and a single 'Entrainment'. Matching against the script's
# names finds nothing.
ALL_TASKS = ['Relaxed', 'RelaxedTask', 'StretchHold', 'LiftHold', 'HoldWeight',
             'PointFinger', 'DrinkGlas', 'CrossArms', 'TouchIndex', 'TouchNose',
             'Entrainment']
# NOTE: PADS's own preprocessing drops these tasks
#   to_remove = 'Time|LiftHold|PointFinger|TouchIndex'
# so LiftHold is NOT used by the dataset authors. StretchHold is the
# arms-outstretched postural task and the appropriate match for the local OUT.
PADS_EXCLUDED_TASKS = ('LiftHold', 'PointFinger', 'TouchIndex')

# ---- Gyroscope columns (CONFIRMED from PADS scripts/load_specific_txt_file.py):
# channel order per file is
#   ['Accelerometer_X','Accelerometer_Y','Accelerometer_Z',
#    'Gyroscope_X','Gyroscope_Y','Gyroscope_Z']
# so cols 3,4,5 are the gyroscope (angular velocity), matching the local data.
# There is NO time/index column (6 channels, all sensor).
GYRO_COLS = [3, 4, 5]
FS_HZ = 100.0   # PADS documented sampling rate; written to the manifest

# ---- VERIFY 2: how the diagnosis is stored in patients/patient_<id>.json -----
# We search these keys (and nested dicts) for a diagnosis string, then normalise.
LABEL_KEYS = ["condition", "disease", "diagnosis", "group", "label",
              "study_group", "cohort", "class"]

# EXACT diagnosis string -> class. Anything else is SKIPPED.
#
# This used to be a substring map including the bare tokens "et" and "pd", which
# was badly wrong: PADS free-text diagnoses put "et" inside "etiology",
# "asymmetric", "Retrocollis" and "hypokinetic", so 13 of 41 extracted "ET"
# patients were not Essential Tremor at all -- among them a hypokinetic-rigid
# syndrome and a Lewy-Body dementia, i.e. PARKINSONIAN cases sitting in the ET
# class. "parkinson" likewise swept in 20 Atypical Parkinsonism cases, which
# PADS treats as a separate differential-diagnosis group.
#
# Exact matching yields N=79 / PD=276 / ET=28, and the ET count now agrees with
# the published PADS cohort (Varghese 2024: 28 ET).
#
# Mixed/differential diagnoses that merely CONTAIN "Essential Tremor"
# (e.g. "Essential Tremor, DD functional Tremor") are deliberately excluded:
# they are ambiguous by construction and cannot support a clean PD-vs-ET claim.
LABEL_MAP_EXACT = {
    "healthy": "N",
    "parkinson's": "PD",
    "essential tremor": "ET",
}


def _norm(s: str) -> str:
    return str(s).strip().lower()


def map_label(raw: str) -> str | None:
    """Exact diagnosis -> N/PD/ET, or None to skip. Never substring-matches."""
    return LABEL_MAP_EXACT.get(_norm(raw))


def find_label(meta):
    """Return (N|PD|ET, raw_string) or (None, None) by searching the JSON.

    Matching is EXACT on the normalised diagnosis string -- see LABEL_MAP_EXACT
    for why substring matching is unsafe here.
    """
    def search(obj):
        if isinstance(obj, dict):
            for k in LABEL_KEYS:
                if k in obj and isinstance(obj[k], (str, int)):
                    lab = map_label(obj[k])
                    if lab:
                        return lab, obj[k]
            for v in obj.values():
                r = search(v)
                if r[0]:
                    return r
        elif isinstance(obj, str):
            lab = map_label(obj)
            if lab:
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
    """Locate the timeseries/ and patients/ dirs inside a PADS download.

    Fails loudly. Earlier this returned ``(root, None)`` for a non-existent
    root, so ``--inspect`` printed two header lines and exited 0 -- looking like
    "ran fine, found nothing" instead of "that folder does not exist".
    """
    if not root.exists():
        raise SystemExit(
            f"--pads-root '{root}' does not exist.\n"
            f"  This repo does NOT ship the raw PADS dataset -- only the\n"
            f"  pre-extracted 'pads_stretchhold/' output. Download PADS first:\n"
            f"    https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/\n"
            f"  (PhysioNet DOI 10.13026/m0w9-zx22), unpack it, and point\n"
            f"  --pads-root at the unpacked folder.")
    if not root.is_dir():
        raise SystemExit(f"--pads-root '{root}' is a file, not a directory.")

    ts = next(iter(root.rglob("timeseries")), None)
    pat = next(iter(root.rglob("patients")), None)
    if ts is None:
        n_txt = sum(1 for _ in root.rglob("*.txt"))
        raise SystemExit(
            f"no 'timeseries/' directory found under '{root}' "
            f"({n_txt} .txt files seen anywhere below it).\n"
            f"  Expected the PADS layout:\n"
            f"    <root>/movement/timeseries/<id>_<Task>_<Wrist>.txt\n"
            f"    <root>/patients/patient_<id>.json\n"
            f"  If your copy is laid out differently, point --pads-root at the\n"
            f"  directory that CONTAINS 'timeseries'.")
    if pat is None:
        raise SystemExit(
            f"found timeseries at '{ts}' but no 'patients/' directory under "
            f"'{root}'.\n  Diagnoses live in patients/patient_<id>.json and are "
            f"required for labelling.")
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
    if sample is None:
        raise SystemExit(f"no .txt recordings found under '{ts_dir}'.")
    if sample:
        print(f"--- sample timeseries: {sample.name} ---")
        for line in sample.read_text().splitlines()[:3]:
            print("   ", line[:120])
        arr = load_timeseries(sample)
        print(f"   shape (T, axes) = {arr.shape}  -> using gyro cols {GYRO_COLS}")
        # time-column safety check: PADS has no time column (6 sensor axes). If
        # column 0 is monotonically increasing it is a time/index column and the
        # gyro columns would be off by one.
        col0 = arr[:, 0]
        mono = bool(np.all(np.diff(col0) > 0))
        print(f"   column 0 monotonic (time/index?) = {mono}  "
              f"{'<-- WARNING: shift GYRO_COLS by +1' if mono else '(ok, sensor data)'}")
        print(f"   duration = {arr.shape[0]/FS_HZ:.1f}s at assumed fs={FS_HZ:g}Hz\n")
    pj = next(iter(pat_dir.glob("patient_*.json")), None) if pat_dir else None
    if pj:
        meta = json.loads(pj.read_text())
        print(f"--- sample patient json: {pj.name} ---")
        print("   top-level keys:", list(meta.keys()))
        lab, raw = find_label(meta)
        print(f"   detected label: {lab}  (from value {raw!r})")


def list_tasks(root: Path):
    """Enumerate the task tokens actually present, with file counts."""
    ts_dir, _ = find_dirs(root)
    counts = {}
    for f in ts_dir.rglob("*.txt"):
        parts = f.stem.split("_")
        if len(parts) >= 2:
            counts[parts[1]] = counts.get(parts[1], 0) + 1
    if not counts:
        raise SystemExit(f"no .txt recordings found under '{ts_dir}'.")
    print(f"tasks found in {ts_dir}:")
    for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        flag = "  <-- excluded by PADS's own pipeline" if t in PADS_EXCLUDED_TASKS else ""
        star = "  <-- currently extracted" if t == TASK else ""
        print(f"  {t:<16} {n:>5} files{star}{flag}")


def extract(root: Path, out: Path, wrist: str, gyro_only: bool,
            trim_start: float = 0.0, trim_end: float = 0.0, task: str = TASK):
    ts_dir, pat_dir = find_dirs(root)
    labels = load_patient_labels(pat_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = {"N": 0, "PD": 0, "ET": 0}
    skipped = 0
    # Match the task field EXACTLY, not as a substring. Filenames are
    # <id>_<Task>_<Wrist>, and a substring match on "Relaxed" also pulls in
    # every "RelaxedTask" file -- a different condition (rest WITH a cognitive
    # task). Both would then be written under the same output name and silently
    # overwrite each other.
    for f in sorted(ts_dir.rglob("*")):
        if f.suffix.lower() not in (".txt", ".csv", ".json"):
            continue
        parts = f.stem.split("_")
        if len(parts) < 2 or parts[1].lower() != task.lower():
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
        # Trim the arm-raising onset (and optional offset) so only the steady
        # outstretched hold remains — PADS StretchHold includes a ~1 s raise
        # transient that the local OUT data does not.
        s0 = int(round(trim_start * FS_HZ)); e0 = int(round(trim_end * FS_HZ))
        if (s0 or e0) and arr.shape[0] - s0 - e0 > int(FS_HZ):   # keep >= 1 s
            arr = arr[s0: arr.shape[0] - e0 if e0 else arr.shape[0]]
        wr = "RightWrist" if "right" in f.stem.lower() else ("LeftWrist" if "left" in f.stem.lower() else "NA")
        # The task token MUST be in the filename. Several PADS tasks have
        # repetitions (Relaxed1/Relaxed2, Entrainment1/Entrainment2); without it
        # the second repetition silently overwrites the first and half the data
        # disappears with no error. The exact token is taken from the source
        # stem, so Relaxed1 and Relaxed2 stay distinct.
        tok = parts[1]                      # exact token from the source filename
        outfile = out / f"{cls}_{pid}_{tok}_{wr}.txt"
        # comma-separated text (same format as the local raw_quaternion data)
        np.savetxt(outfile, arr.astype(np.float32), delimiter=",", fmt="%.6f")
        manifest.append({"file": outfile.name, "patient": pid, "class": cls,
                         "task": tok,
                         "wrist": wr, "n_samples": arr.shape[0], "n_channels": arr.shape[1],
                         "fs_hz": FS_HZ, "duration_s": round(arr.shape[0] / FS_HZ, 3),
                         "raw_label": lab[1]})
        counts[cls] += 1

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "patient", "class", "task", "wrist",
                                           "n_samples", "n_channels", "fs_hz",
                                           "duration_s", "raw_label"])
        w.writeheader(); w.writerows(manifest)
    n_pat = len({m["patient"] for m in manifest})
    tasks = sorted({m["task"] for m in manifest})
    print(f"extracted {len(manifest)} recordings ({', '.join(tasks)}), {n_pat} patients")
    print(f"  per class: {counts}   (skipped {skipped} non-N/PD/ET or unlabeled)")
    print(f"  saved -> {out}/ (one .txt per recording + manifest.csv)")


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
    p.add_argument("--trim-start", type=float, default=0.0,
                   help="seconds to drop from the start (arm-raising onset). "
                        "Default 0 (tested: trimming 1.5s did NOT help -- "
                        "PADS-only ET-F1 0.26->0.20 and domain shift unchanged).")
    p.add_argument("--trim-end", type=float, default=0.0,
                   help="seconds to drop from the end (arm-lowering offset).")
    p.add_argument("--task", default=TASK,
                   help=f"PADS task token to extract (default {TASK}). Known: "
                        + ", ".join(ALL_TASKS))
    p.add_argument("--list-tasks", action="store_true",
                   help="list task tokens present in the dataset and exit.")
    p.add_argument("--inspect", action="store_true",
                   help="print file-format structure and exit (run this first).")
    args = p.parse_args()

    if args.list_tasks:
        list_tasks(args.pads_root)
    elif args.inspect:
        inspect(args.pads_root)
    else:
        extract(args.pads_root, args.out, args.wrist, args.gyro_only,
                trim_start=args.trim_start, trim_end=args.trim_end,
                task=args.task)


if __name__ == "__main__":
    main()
