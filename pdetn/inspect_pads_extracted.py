#!/usr/bin/env python3
"""Inspect the extracted PADS StretchHold data (from pdetn.extract_pads).

Checks: class/patient counts, a LABEL AUDIT (what raw diagnosis strings were
mapped to each class -- to catch mislabels), signal-quality stats (lengths,
channels, NaN/flat), and per-class spectral sanity (dominant tremor frequency,
which should sit ~4-6 Hz for PD and higher for ET).

    python -m pdetn.inspect_pads_extracted --data pads_stretchhold
    # add --plot to also save a PSD-by-class figure
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import welch

TREMOR_LO, TREMOR_HI = 3.0, 15.0


def load_manifest(data: Path):
    rows = list(csv.DictReader(open(data / "manifest.csv")))
    if not rows:
        raise SystemExit(f"empty manifest in {data}")
    return rows


def tremor_psd(x, fs):
    """Sum PSD over channels, restricted to the tremor band."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    nper = int(min(256, x.shape[0]))
    f, P = welch(x, fs=fs, nperseg=nper, axis=0)
    P = P.sum(axis=1)
    band = (f >= TREMOR_LO) & (f < TREMOR_HI)
    return f[band], P[band]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("pads_stretchhold"))
    ap.add_argument("--fs", type=float, default=100.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rows = load_manifest(args.data)
    print(f"=== {len(rows)} recordings in {args.data} ===\n")

    # ---- 1. class / patient counts ----
    rec_by_cls = Counter(r["class"] for r in rows)
    pat_by_cls = {c: len({r["patient"] for r in rows if r["class"] == c})
                  for c in rec_by_cls}
    print("counts (recordings | unique patients):")
    for c in ("N", "PD", "ET"):
        print(f"  {c:>3}: {rec_by_cls.get(c,0):>4} | {pat_by_cls.get(c,0):>3}")
    print(f"  total patients: {len({r['patient'] for r in rows})}")

    # ---- 2. LABEL AUDIT (catch mislabels) ----
    print("\n=== label audit: raw diagnosis strings -> mapped class ===")
    raw_by_cls = defaultdict(Counter)
    for r in rows:
        raw_by_cls[r["class"]][r.get("raw_label", "?")] += 1
    for c in ("N", "PD", "ET"):
        print(f"  {c}:")
        for raw, n in raw_by_cls[c].most_common():
            flag = "  <-- CHECK" if c == "ET" and "essential" not in str(raw).lower() else ""
            print(f"      {n:>4}  {raw!r}{flag}")

    # ---- 3. signal-quality stats ----
    lengths, chans, nan_ct, flat_ct = [], Counter(), 0, 0
    for r in rows:
        x = np.loadtxt(args.data / r["file"], delimiter=",", ndmin=2)
        lengths.append(x.shape[0]); chans[x.shape[1] if x.ndim > 1 else 1] += 1
        if not np.isfinite(x).all():
            nan_ct += 1
        if np.nanstd(x) < 1e-9:
            flat_ct += 1
    lengths = np.array(lengths)
    print("\n=== signal quality ===")
    print(f"  length (samples): min {lengths.min()}  median {int(np.median(lengths))}  "
          f"max {lengths.max()}  (~{np.median(lengths)/args.fs:.1f}s @ {args.fs:g}Hz)")
    print(f"  channels per file: {dict(chans)}")
    print(f"  files with NaN/Inf: {nan_ct}   near-flat (std~0): {flat_ct}")

    # ---- 4. per-class spectral sanity ----
    print("\n=== spectral sanity (dominant tremor freq, Hz) ===")
    dom = defaultdict(list)
    for r in rows:
        f, P = tremor_psd(np.loadtxt(args.data / r["file"], delimiter=",", ndmin=2), args.fs)
        if len(P):
            dom[r["class"]].append(float(f[int(np.argmax(P))]))
    for c in ("N", "PD", "ET"):
        d = np.array(dom[c])
        if len(d):
            print(f"  {c:>3}: median {np.median(d):.1f}  mean {d.mean():.1f}  "
                  f"(n={len(d)})")
    print("  (expect PD lower ~4-6 Hz, ET higher ~5-9 Hz; N mixed)")

    # ---- 5. optional PSD-by-class figure ----
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        grid = np.linspace(TREMOR_LO, TREMOR_HI, 60)
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, col in [("N", "#2c7fb8"), ("PD", "#d95f02"), ("ET", "#1b9e77")]:
            curves = []
            for r in rows:
                if r["class"] != c:
                    continue
                f, P = tremor_psd(np.loadtxt(args.data / r["file"], delimiter=",", ndmin=2), args.fs)
                if len(P):
                    curves.append(np.interp(grid, f, np.log1p(P)))
            if curves:
                ax.plot(grid, np.mean(curves, 0), color=col, label=f"{c} (n={len(curves)})")
        ax.set_xlabel("Hz"); ax.set_ylabel("log(1+PSD)")
        ax.set_title("PADS StretchHold — mean tremor-band PSD by class"); ax.legend()
        out = args.data / "inspect_psd_by_class.png"
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"\nfigure saved -> {out}")


if __name__ == "__main__":
    main()
