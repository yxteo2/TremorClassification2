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


def _welch(w: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Power spectrum of a (3, T) angular-velocity block, summed over axes."""
    from scipy.signal import welch  # noqa: PLC0415

    nperseg = int(min(4 * fs, w.shape[1]))
    f, P = welch(w, fs=fs, nperseg=nperseg, axis=1)
    return f, P.sum(0)


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
        w = x[3 * i : 3 * i + 3]
        f, P = _welch(w, fs)
        band = (f >= TREMOR_BAND[0]) & (f <= TREMOR_BAND[1])
        total = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1])
        tot = float(P[total].sum())
        peak_i = int(P[band].argmax())
        row = {
            "joint": joint,
            "fs": fs,
            "seconds": round(Q.shape[0] / fs, 2),
            "rms_rad_s": float(np.sqrt((w**2).mean())),
            "peak_hz": float(f[band][peak_i]),
            # how far the peak stands above the rest of the band: 1.0 = flat
            "peak_prominence": float(P[band][peak_i] / np.median(P[band]))
            if np.median(P[band]) > 0
            else np.nan,
            "tremor_frac": float(P[band].sum() / tot) if tot > 0 else np.nan,
        }
        for name, (lo, hi) in SUB_BANDS.items():
            sel = (f >= lo) & (f <= hi)
            row[name] = float(P[sel].sum() / tot) if tot > 0 else np.nan
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
    ap.add_argument("--root", required=True)
    ap.add_argument("--groups", nargs="*", help="restrict to ET / PD / HC")
    ap.add_argument("--out", help="write the per-trial table to this CSV")
    ap.add_argument("--per-subject", action="store_true", help="also print subjects")
    ap.add_argument(
        "--metrics",
        nargs="*",
        default=["tremor_frac", "peak_hz", "rms_rad_s", "peak_prominence"],
    )
    args = ap.parse_args(argv)

    df = profile_tree(args.root, groups=args.groups)
    if df.empty:
        print(f"no trials profiled under {args.root}")
        return 1

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
