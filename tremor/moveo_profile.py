"""Descriptive tremor-band profile of a Moveo Explorer export tree.

Answers "what is actually in this data?" with numbers instead of impressions:
per-joint angular-velocity magnitude, where the spectral peak sits, and how much
of the movement power falls in the tremor band — aggregated with **the subject as
the unit**, which is the only aggregation that means anything when trial counts
differ from 9 to 14 per subject.

This is exploratory description, not a classifier and not evidence of class
separation. Every trial in these exports is labelled `Free Form` with no task
recorded (see ``reports/track4_moveo_export.md``), so a group difference here can
just as easily be a difference in what the subjects were asked to do.

Usage::

    python -m tremor.moveo_profile --root "/path/to/Tremor Classification IMU"
    python -m tremor.moveo_profile --root ... --groups ET --out et_profile.csv
    python -m tremor.moveo_profile --root ... --per-subject
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from tremor.moveo_data import (
    MOVEO_JOINTS,
    _iter_export_dirs,
    parse_export_dir,
    read_analysis_h5,
)
from tremor.quaternion import process_quaternion_data

#: Broad tremor band. Pathological hand tremor lives here; 0.5-3 Hz is
#: voluntary movement and >12 Hz is mostly sensor/soft-tissue noise.
TREMOR_BAND = (3.0, 12.0)

#: Denominator for the band ratio — everything the body plausibly produces.
TOTAL_BAND = (0.5, 30.0)

#: Literature-typical sub-bands, reported side by side rather than merged:
#: PD rest tremor clusters 4-6 Hz, ET postural tremor 5-10 Hz, and they overlap.
SUB_BANDS = {"pd_band_3_7": (3.0, 7.0), "et_band_5_12": (5.0, 12.0)}


#: Where the *global* spectral maximum is looked for, to expose the fact that
#: voluntary movement below ~1.5 Hz normally dominates these recordings.
GLOBAL_BAND = (0.5, 20.0)

#: Range the 1/f^a background is fitted over before looking for a tremor peak.
BACKGROUND_BAND = (1.0, 20.0)


def _welch(w: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a (3, T) angular-velocity block, summed over axes."""
    from scipy.signal import welch  # noqa: PLC0415

    nperseg = int(min(8 * fs, w.shape[1]))
    f, P = welch(w, fs=fs, nperseg=nperseg, axis=1)
    return f, P.sum(0)


def _background(f: np.ndarray, P: np.ndarray) -> np.ndarray:
    """A 1/f^a background fitted in log-log space over :data:`BACKGROUND_BAND`.

    Raw movement spectra fall off steeply from ~0.5 Hz, so ``argmax`` inside a
    3-12 Hz window lands on the window's lower edge whenever no real peak exists,
    and a peak-to-band-median ratio rewards exactly that decay. Dividing the
    spectrum by a fitted background removes the slope, so what is left is genuine
    narrowband excess. Without this step a monotonically decaying spectrum reports
    a confident 3.00 Hz "tremor" that is nothing but the band boundary.
    """
    sel = (f >= BACKGROUND_BAND[0]) & (f <= BACKGROUND_BAND[1]) & (P > 0) & (f > 0)
    if sel.sum() < 4:
        return np.full_like(P, np.nan)
    coef = np.polyfit(np.log10(f[sel]), np.log10(P[sel]), 1)
    with np.errstate(divide="ignore"):
        base = 10.0 ** np.polyval(coef, np.log10(np.where(f > 0, f, np.nan)))
    return base


def _descriptors(w: np.ndarray, fs: float) -> dict:
    """Spectral descriptors for one joint's (3, T) angular-velocity block."""
    f, P = _welch(w, fs)
    df_hz = float(f[1] - f[0])
    band = (f >= TREMOR_BAND[0]) & (f <= TREMOR_BAND[1])
    total = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1])
    glob = (f >= GLOBAL_BAND[0]) & (f <= GLOBAL_BAND[1])
    tot = float(P[total].sum())

    # Excess over the fitted 1/f^a background — this, not raw power, is what a
    # narrowband tremor shows up in.
    base = _background(f, P)
    R = np.where(np.isfinite(base) & (base > 0), P / base, np.nan)
    k = int(np.nanargmax(R[band])) if np.isfinite(R[band]).any() else 0
    peak_hz = float(f[band][k])

    row = {
        "fs": fs,
        "rms_rad_s": float(np.sqrt((w**2).mean())),
        # where the whole spectrum peaks: normally <1.5 Hz voluntary movement
        "global_peak_hz": float(f[glob][int(P[glob].argmax())]),
        "peak_hz": peak_hz,
        # >1 means the peak stands above the 1/f background; ~1 means nothing
        "peak_excess": float(R[band][k]),
        # a peak sitting on the window boundary is a search artefact, not a peak
        "peak_at_edge": bool(
            peak_hz <= TREMOR_BAND[0] + df_hz or peak_hz >= TREMOR_BAND[1] - df_hz
        ),
        "tremor_frac": float(P[band].sum() / tot) if tot > 0 else np.nan,
    }
    for name, (lo, hi) in SUB_BANDS.items():
        sel = (f >= lo) & (f <= hi)
        row[name] = float(P[sel].sum() / tot) if tot > 0 else np.nan
    return row


def trial_profile(
    path: Path | str, joints: tuple[str, ...] | list[str] = MOVEO_JOINTS
) -> list[dict]:
    """One row of descriptors per joint for a single ``*_Analysis.h5`` trial."""
    Q, fs = read_analysis_h5(path, joints=joints)
    x = process_quaternion_data(
        Q, fs=fs, mode="angular_velocity", convention="xyzw", n_sensors=len(joints)
    )
    rows = []
    for i, joint in enumerate(joints):
        row = {"joint": joint, "seconds": round(Q.shape[0] / fs, 2)}
        row.update(_descriptors(x[3 * i : 3 * i + 3], fs))
        rows.append(row)
    return rows


def profile_tree(root: Path | str, groups=None):
    """Profile every trial under ``root``; returns a tidy DataFrame."""
    import pandas as pd  # noqa: PLC0415

    wanted = {g.upper() for g in groups} if groups else None
    rows: list[dict] = []
    for export_dir in sorted(_iter_export_dirs(Path(root))):
        group, subject, letter = parse_export_dir(export_dir)
        if wanted is not None and group not in wanted:
            continue
        for h5_path in sorted(export_dir.glob("*_Analysis.h5")):
            try:
                trial_rows = trial_profile(h5_path)
            except Exception as exc:  # a bad trial must not kill the sweep
                print(f"  ! skipped {h5_path.name}: {exc}")
                continue
            for r in trial_rows:
                rows.append(
                    {"group": group, "class": letter, "subject": subject,
                     "trial": h5_path.name.split("_")[0], **r}
                )
    return pd.DataFrame(rows)


#: Joint names produced from the repo's 3-sensor data by harmonisation. Only one
#: arm is instrumented there, so there is no Left/Right distinction.
LOCAL_JOINTS = ("Elbow", "Wrist")

_TRIAL_THEN_ACTION = re.compile(r"[_\s]+\d+$"), re.compile(r"[_\s]+(OUT|REST|WING)$")


def local_subject_id(stem: str) -> str:
    """``'ET 10_REST 1'`` -> ``'ET 10'``: strip the trial number, then the action.

    Both steps are required. Leaving the action on the ID is the cross-condition
    leak the skill warns about: `_assert_disjoint_subjects` still passes because
    ``ET 10_OUT`` and ``ET 10_REST`` look like different subjects.
    """
    trial_re, action_re = _TRIAL_THEN_ACTION
    return action_re.sub("", trial_re.sub("", stem))


def profile_local_tree(
    root: Path | str,
    action: str,
    fs: float = 100.0,
    feature: str = "raw_quaternion",
):
    """Profile the repo's own ``raw_quaternion`` data in the export's representation.

    The 3-sensor files are harmonised to elbow/wrist joint angles first (see
    :func:`tremor.moveo_data.joint_quaternions_from_sensors`), so the descriptors
    are computed on the same kind of quantity as the Moveo exports and the two
    cohorts can be read side by side.

    Unlike the exports, these recordings **are** task-labelled, so passing a single
    ``action`` removes the task confound that makes the Drive numbers unusable.

    ``fs`` defaults to the package's 100 Hz. That value is unverified — see the
    report's "is the existing data really 100 Hz?" section — and every frequency
    here scales linearly with it.
    """
    import pandas as pd  # noqa: PLC0415

    from tremor.data import CLASS_MAP  # noqa: PLC0415
    from tremor.moveo_data import joint_quaternions_from_sensors  # noqa: PLC0415

    root = Path(root)
    data_dir = next(
        (
            p
            for p in (root / feature / action, root / "ProcessedData" / feature / action)
            if p.is_dir()
        ),
        None,
    )
    if data_dir is None:
        raise FileNotFoundError(f"no {feature}/{action} under {root}")

    letters = {"N": "N", "PD": "P", "ET": "E"}
    rows: list[dict] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        letter = letters.get(class_dir.name)
        if letter is None:
            continue
        for path in sorted(class_dir.glob("*.txt")):
            Q12 = np.loadtxt(path, delimiter=",")
            if Q12.ndim != 2 or Q12.shape[1] != 12 or Q12.shape[0] < int(4 * fs):
                continue
            J = joint_quaternions_from_sensors(Q12)
            x = process_quaternion_data(
                J, fs=fs, mode="angular_velocity", convention="xyzw", n_sensors=2
            )
            for i, joint in enumerate(LOCAL_JOINTS):
                row = {
                    "group": class_dir.name,
                    "class": letter,
                    "subject": local_subject_id(path.stem),
                    "trial": path.stem,
                    "condition": action,
                    "joint": joint,
                    "seconds": round(Q12.shape[0] / fs, 2),
                }
                row.update(_descriptors(x[3 * i : 3 * i + 3], fs))
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(df, metric: str = "tremor_frac"):
    """Subject medians first, then class-level median and IQR across subjects."""
    per_subject = (
        df.groupby(["class", "subject", "joint"])[metric].median().reset_index()
    )
    out = (
        per_subject.groupby(["class", "joint"])[metric]
        .agg(
            n_subjects="count",
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    return out


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="Moveo Explorer export tree")
    ap.add_argument("--local-root", help="repo data root (e.g. Data) — profiles raw_quaternion")
    ap.add_argument(
        "--action", default="OUT", help="condition for --local-root: OUT / REST / WING"
    )
    ap.add_argument("--fs", type=float, default=100.0, help="fs for --local-root")
    ap.add_argument("--groups", nargs="*", help="restrict to ET / PD / HC")
    ap.add_argument("--out", help="write the per-trial table to this CSV")
    ap.add_argument("--per-subject", action="store_true", help="also print subjects")
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=["tremor_frac", "peak_hz", "peak_excess", "rms_rad_s"],
    )
    args = ap.parse_args(argv)
    if not args.root and not args.local_root:
        ap.error("pass --root (export tree) or --local-root (repo data)")

    if args.local_root:
        df = profile_local_tree(args.local_root, args.action, fs=args.fs)
        label = f"{args.local_root} [{args.action}] @ {args.fs:g} Hz"
    else:
        df = profile_tree(args.root, groups=args.groups)
        label = str(args.root)
    if df.empty:
        print(f"no trials profiled under {label}")
        return 1
    print(f"source: {label}")

    n_sub = df["subject"].nunique()
    n_trials = df.groupby(["subject", "trial"]).ngroups
    print(
        f"{n_trials} trials / {n_sub} subjects / "
        f"{sorted(df['fs'].unique())} Hz / "
        f"{df['seconds'].sum() / len(MOVEO_JOINTS) / 60:.1f} min of signal\n"
    )

    if args.per_subject:
        per = (
            df.groupby(["class", "subject", "joint"])[args.metrics]
            .median()
            .round(3)
            .reset_index()
        )
        print(per.to_string(index=False))
        print()

    for metric in args.metrics:
        print(f"--- {metric} (median of subject medians, IQR across subjects) ---")
        s = summarise(df, metric)
        for _, r in s.iterrows():
            print(
                f"{r['class']:3s} {r['joint']:14s} n={int(r['n_subjects']):3d}  "
                f"{r['median']:8.3f}  [{r['q1']:.3f}, {r['q3']:.3f}]"
            )
        print()

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
