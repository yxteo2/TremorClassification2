"""Adapter for **Moveo Explorer subject exports** (APDM / Mobility Lab HDF5).

This is the format of the newer in-house recruitment batches that live in the
Google Drive folder ``Tremor Classification IMU/{ET,PD,HC}/``. It is **not** the
same representation as ``Data/raw_quaternion/`` — read
``reports/track4_moveo_export.md`` before pooling the two.

Export layout (one directory per subject)::

    Moveo_Explorer_Subject_Export_<GROUP>_<ID>_<yyyymmdd-hhmmssTZ>/
        NN_1_<yyyymmdd-hhmmss>_<Condition>_<GROUP>_<localid>_Analysis.h5
        NN_1_<...>SGT_<Condition>_Trial.csv                 <- per-trial summary
        NN_1_<...>SGT_<Condition>_Trial_Joint_Angles.csv    <- same series as CSV
        <Condition>_trials.csv, <Condition>_DTSv3.csv, ...
        SubjectMetadata.xml
        DataTransferSpecificationNotes.DTSv3.txt

What the ``*_Analysis.h5`` files actually contain (verified against ET 21,
PD 88 and HC 100 exports)::

    Measures/Duration                              (1, 1)   seconds
    Processed/Joint Angles                          attrs: sampleRate=128.0, nSamples
    Processed/Joint Angles/{Elbow,Wrist}/{Left,Right}/
        Quaternion                                 (T, 4)   unit, SCALAR-FIRST (w,x,y,z)
        X, Y, Z                                    (T, 1)   Euler degrees (order attr)

Three facts that make this different from ``raw_quaternion`` and that every
caller has to respect:

1. **fs = 128 Hz**, not the 100 Hz the rest of the package defaults to. Passing
   ``fs=100`` mislabels every frequency by 1.28x (the fs trap, again).
2. Quaternions are **scalar-first ``(w, x, y, z)``**; the package convention is
   scalar-last. :func:`read_analysis_h5` reorders by default.
3. These are **joint angles** (relative orientation across a joint), not the
   three per-sensor absolute orientations in ``raw_quaternion``. Use
   :func:`joint_quaternions_from_sensors` to bring the old data into this
   representation rather than pretending the channels are interchangeable.

The first ``startDelay`` seconds of every trial are the standing calibration
pose, not the task: ``nSamples / sampleRate - Duration == 3.0 s`` held on all
three spot-checked exports, so :data:`CALIBRATION_S` is trimmed by default.

CLI::

    python -m tremor.moveo_data --root /path/to/Tremor\\ Classification\\ IMU --inventory
    python -m tremor.moveo_data --root ... --inventory --out inventory.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from tremor.data import CLASS_MAP, Recording
from tremor.quaternion import process_quaternion_data, quat_conjugate, quat_multiply

#: Sampling rate declared by the exporter (``Processed/Joint Angles`` attrs).
MOVEO_FS = 128.0

#: Standing-calibration pose at the head of every trial (``startDelay``).
CALIBRATION_S = 3.0

#: Joint-angle streams present in an Analysis h5, in canonical channel order.
MOVEO_JOINTS: tuple[str, ...] = (
    "Elbow/Left",
    "Elbow/Right",
    "Wrist/Left",
    "Wrist/Right",
)

#: Export group folder -> the class letter used by :data:`tremor.data.CLASS_MAP`.
#: The exports label controls ``HC``; the rest of the package calls them ``N``.
GROUP_TO_LETTER = {"HC": "N", "N": "N", "PD": "P", "ET": "E"}

_JOINT_ANGLES = "Processed/Joint Angles"

_EXPORT_DIR = re.compile(
    r"^Moveo_Explorer_Subject_Export_(?P<group>[A-Za-z]+)_(?P<num>\d+)_(?P<stamp>.+)$"
)


def parse_export_dir(path: Path | str) -> tuple[str, str, str]:
    """Split an export directory name into ``(group, subject, class_letter)``.

    The subject ID comes from the **directory** name, never from the h5
    filename: the ``PD 88`` export ships files called
    ``..._Free_Form_PD_1_Analysis.h5`` because the h5 name carries the
    acquisition-station local ID, not the study ID. Trusting the filename would
    silently relabel PD 88 as PD 1 — and PD 1 already exists in
    ``Data/raw_quaternion``.

    Subjects are namespaced ``MV-<GROUP><num>`` so that pooling with the older
    cohort can never collide on a subject key even where the numbering overlaps.
    """
    name = Path(path).name
    m = _EXPORT_DIR.match(name)
    if m is None:
        raise ValueError(f"not a Moveo Explorer export directory: {name!r}")
    group = m.group("group").upper()
    letter = GROUP_TO_LETTER.get(group)
    if letter is None:
        raise ValueError(
            f"unknown subject group {group!r} in {name!r}; "
            f"expected one of {sorted(GROUP_TO_LETTER)}"
        )
    return group, f"MV-{group}{m.group('num')}", letter


def _require_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "reading Moveo Explorer exports needs h5py (`pip install h5py`)"
        ) from exc
    return h5py


def read_analysis_h5(
    path: Path | str,
    joints: tuple[str, ...] | list[str] = MOVEO_JOINTS,
    convention: str = "xyzw",
    trim_calibration_s: float = CALIBRATION_S,
) -> tuple[np.ndarray, float]:
    """Read joint-angle quaternions out of one ``*_Analysis.h5``.

    Args:
        path: the Analysis h5 file.
        joints: which ``<Joint>/<Side>`` streams to stack, in channel order.
        convention: ``'xyzw'`` (default, matches the rest of the package) to
            reorder the exporter's scalar-first quaternions, or ``'wxyz'`` to
            keep them as stored.
        trim_calibration_s: seconds dropped from the head of the trial. The
            exporter prepends a standing calibration pose; keeping it would feed
            a few hundred still samples into every spectrogram.

    Returns:
        ``(Q, fs)`` where ``Q`` is ``(T, len(joints) * 4)`` — the layout
        :func:`tremor.quaternion.process_quaternion_data` expects with
        ``n_sensors=len(joints)``.
    """
    h5py = _require_h5py()
    with h5py.File(str(path), "r") as f:
        if _JOINT_ANGLES not in f:
            raise ValueError(f"{path}: no {_JOINT_ANGLES!r} group")
        grp = f[_JOINT_ANGLES]
        fs = float(grp.attrs.get("sampleRate", MOVEO_FS))
        blocks = []
        for joint in joints:
            key = f"{joint}/Quaternion"
            if key not in grp:
                raise ValueError(
                    f"{path}: missing joint stream {key!r} "
                    f"(present: {sorted(_available_joints(grp))})"
                )
            blocks.append(np.asarray(grp[key], dtype=np.float64))

    lengths = {b.shape[0] for b in blocks}
    if len(lengths) != 1:
        raise ValueError(f"{path}: joint streams disagree on length: {sorted(lengths)}")

    Q = np.concatenate(blocks, axis=1)  # (T, J*4), scalar-first per joint
    if convention == "xyzw":
        Q = Q.reshape(Q.shape[0], len(joints), 4)[:, :, [1, 2, 3, 0]]
        Q = Q.reshape(Q.shape[0], len(joints) * 4)
    elif convention != "wxyz":
        raise ValueError(f"unknown convention {convention!r}; expected 'xyzw'/'wxyz'")

    if trim_calibration_s > 0:
        start = int(round(trim_calibration_s * fs))
        if start < Q.shape[0]:
            Q = Q[start:]
    return np.ascontiguousarray(Q, dtype=np.float32), fs


def _available_joints(grp) -> set[str]:
    found: set[str] = set()

    def visit(name, obj):
        if name.endswith("Quaternion"):
            found.add(name.rsplit("/", 1)[0])

    grp.visititems(visit)
    return found


def load_moveo_recordings(
    root: Path | str,
    groups: tuple[str, ...] | list[str] | None = None,
    joints: tuple[str, ...] | list[str] = MOVEO_JOINTS,
    mode: str = "angular_velocity",
    trim_calibration_s: float = CALIBRATION_S,
    min_duration_s: float = 5.0,
    condition: str = "FREEFORM",
) -> list[Recording]:
    """Load a locally-synced export tree into :class:`Recording` objects.

    ``root`` is the folder that holds the per-class subfolders
    (``ET/``, ``PD/``, ``HC/``), or any directory above them — export
    directories are found recursively.

    Every trial is labelled ``condition='FREEFORM'``, because that is all the
    export says: ``conditionName="Free Form"`` on every trial, with empty trial
    notes. These recordings therefore **cannot** be mapped onto OUT/REST/WING
    without an external session log — see the report.
    """
    root = Path(root)
    wanted = {g.upper() for g in groups} if groups else None
    recordings: list[Recording] = []

    for export_dir in sorted(_iter_export_dirs(root)):
        group, subject, letter = parse_export_dir(export_dir)
        if wanted is not None and group not in wanted:
            continue
        label = CLASS_MAP[letter]
        for h5_path in sorted(export_dir.glob("*_Analysis.h5")):
            Q, fs = read_analysis_h5(
                h5_path,
                joints=joints,
                convention="xyzw",
                trim_calibration_s=trim_calibration_s,
            )
            if Q.shape[0] < max(3, int(round(min_duration_s * fs))):
                continue
            x = process_quaternion_data(
                Q, fs=fs, mode=mode, convention="xyzw", n_sensors=len(joints)
            )
            recordings.append(
                Recording(
                    x=x,
                    y=label,
                    subject=subject,
                    path=h5_path,
                    condition=condition,
                )
            )
    return recordings


def _iter_export_dirs(root: Path):
    if _EXPORT_DIR.match(root.name):
        yield root
        return
    for path in root.rglob("Moveo_Explorer_Subject_Export_*"):
        if path.is_dir() and _EXPORT_DIR.match(path.name):
            yield path


def moveo_inventory(root: Path | str) -> list[dict]:
    """Per-subject inventory of a synced export tree, without loading signals."""
    h5py = _require_h5py()
    rows: list[dict] = []
    for export_dir in sorted(_iter_export_dirs(Path(root))):
        group, subject, letter = parse_export_dir(export_dir)
        trials, durations, rates, joint_sets = 0, [], set(), set()
        for h5_path in sorted(export_dir.glob("*_Analysis.h5")):
            trials += 1
            with h5py.File(str(h5_path), "r") as f:
                if _JOINT_ANGLES not in f:
                    joint_sets.add("<missing>")
                    continue
                grp = f[_JOINT_ANGLES]
                rates.add(float(grp.attrs.get("sampleRate", float("nan"))))
                n = float(grp.attrs.get("nSamples", 0.0))
                fs = float(grp.attrs.get("sampleRate", MOVEO_FS)) or MOVEO_FS
                durations.append(n / fs)
                joint_sets.add(",".join(sorted(_available_joints(grp))))
        rows.append(
            {
                "group": group,
                "class": letter,
                "subject": subject,
                "export_dir": export_dir.name,
                "n_trials": trials,
                "fs": sorted(rates),
                "total_s": round(float(np.sum(durations)), 1) if durations else 0.0,
                "median_trial_s": (
                    round(float(np.median(durations)), 1) if durations else 0.0
                ),
                "joints": sorted(joint_sets),
            }
        )
    return rows


def joint_quaternions_from_sensors(
    Q12: np.ndarray, convention: str = "xyzw"
) -> np.ndarray:
    """Convert 3-sensor ``raw_quaternion`` data to elbow/wrist joint angles.

    This is the harmonisation direction that *can* be done: the old data holds
    absolute segment orientations for ``(hand, lower_arm, upper_arm)``, and a
    joint angle is the child segment expressed in the parent's frame::

        elbow = conj(upper_arm) * lower_arm
        wrist = conj(lower_arm) * hand

    Args:
        Q12: ``(T, 12)`` = 3 sensors x 4 components, sensor order
            ``(hand, lower_arm, upper_arm)`` per ``tremor.quaternion.SENSOR_NAMES``.
        convention: component order of both input and output.

    Returns:
        ``(T, 8)`` = ``(elbow, wrist)`` x 4 components, ready for
        ``process_quaternion_data(..., n_sensors=2)``.

    The parent-relative direction above is the standard definition, but the sign
    and axis order the exporter uses for *its* joint angles were not verifiable
    from the exports alone (the two cohorts share no subject, so there is nothing
    to cross-check against). Treat a pooled elbow/wrist representation as an
    assumption to validate — e.g. with the dataset-identity probe in
    ``reports/track3_external_data.md`` — not as an established match.
    """
    Q12 = np.asarray(Q12)
    if Q12.ndim != 2 or Q12.shape[1] != 12:
        raise ValueError(f"expected (T, 12) quaternion data, got {Q12.shape}")
    Q = Q12.reshape(Q12.shape[0], 3, 4)
    hand, lower_arm, upper_arm = Q[:, 0, :], Q[:, 1, :], Q[:, 2, :]
    elbow = quat_multiply(quat_conjugate(upper_arm, convention), lower_arm, convention)
    wrist = quat_multiply(quat_conjugate(lower_arm, convention), hand, convention)
    return np.concatenate([elbow, wrist], axis=1).astype(np.float32, copy=False)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="synced export tree")
    ap.add_argument("--inventory", action="store_true", help="summarise, do not load")
    ap.add_argument("--out", help="write the inventory to this CSV")
    ap.add_argument("--groups", nargs="*", help="restrict to ET / PD / HC")
    args = ap.parse_args(argv)

    if args.inventory:
        rows = moveo_inventory(args.root)
        if args.groups:
            wanted = {g.upper() for g in args.groups}
            rows = [r for r in rows if r["group"] in wanted]
        if not rows:
            print(f"no Moveo Explorer export directories under {args.root}")
            return 1
        header = f"{'subject':12s} {'cls':4s} {'trials':>6s} {'total_s':>8s} {'fs':>10s}"
        print(header)
        print("-" * len(header))
        for r in rows:
            fs = "/".join(f"{v:g}" for v in r["fs"]) or "-"
            print(
                f"{r['subject']:12s} {r['class']:4s} {r['n_trials']:6d} "
                f"{r['total_s']:8.1f} {fs:>10s}"
            )
        by_class: dict[str, int] = {}
        for r in rows:
            by_class[r["class"]] = by_class.get(r["class"], 0) + 1
        print(
            f"\n{len(rows)} subjects, "
            f"{sum(r['n_trials'] for r in rows)} trials, "
            f"per class: " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items()))
        )
        if args.out:
            import pandas as pd  # noqa: PLC0415

            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"wrote {args.out}")
        return 0

    recs = load_moveo_recordings(args.root, groups=args.groups)
    if not recs:
        print(f"no trials loaded from {args.root}")
        return 1
    shapes = {r.x.shape[0] for r in recs}
    print(
        f"loaded {len(recs)} recordings from "
        f"{len({r.subject for r in recs})} subjects; channels={sorted(shapes)}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
