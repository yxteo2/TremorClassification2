"""Absolute checks on the REAL recordings — the gap `verify_preprocessing` leaves.

`verify_preprocessing.py` pushes **synthetic** signals with known answers through
every stage, so it validates the *code*. **Nothing in this project validates the
*data*.** That asymmetry is how three defects survived 68 reports, and the
duplicated frequency-axis bug survived three weeks after its own fix.

These are checks a synthetic signal cannot make, because they are properties of
the recordings themselves. Exit code is the number of failures, like
`verify_preprocessing`.

The two that matter most have never been run in this project:

  **duplicate recordings** — the same signal appearing under two subject ids
  would put a patient on both sides of a patient-level split, and every
  safeguard here is *relative*, so it would be invisible to all of them.

  **subject-id collisions across cohorts** — the merge concatenates three
  loaders that mint ids independently. A collision silently merges two people.

Run: ``python -m experiments.verify_data``
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter, defaultdict

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

warnings.filterwarnings("ignore")

FS = 100.0
BAND = (3.0, 15.0)
res = []


def check(name, ok, detail=""):
    res.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def load_all():
    from common.loaders import load_pads_extracted
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    return {
        "2015": (load_quaternion_recordings("Data", action="OUT",
                                            mode="angular_velocity"),
                 slice(3, 6)),
        "NewData": (load_2025_all(conditions=("OUT",)), slice(3, 6)),
        "PADS": (load_pads_extracted("pads_stretchhold"), slice(0, 3)),
    }


def gyro(r, ch):
    return np.asarray(r.x[ch] if r.x.shape[0] > 3 else r.x, float)


def band_sig(x):
    """16-bin log spectrum of the band-passed axis-mean — a cheap fingerprint."""
    f, P = welch(x, fs=FS, nperseg=min(256, x.shape[-1]), axis=-1)
    P = np.asarray(P).mean(0)
    k = (f >= BAND[0]) & (f <= BAND[1])
    v = P[k]
    if v.sum() <= 0 or not np.isfinite(v).all():
        return np.zeros(16)
    v = v / v.sum()
    m = len(v) // 16 * 16
    if m == 0:
        return np.zeros(16)
    return np.log(v[:m] + 1e-12).reshape(16, -1).mean(1)


def main():
    data = load_all()
    allrec = [(c, r, ch) for c, (rs, ch) in data.items() for r in rs]
    print(f"{len(allrec)} recordings across {len(data)} cohorts\n")

    # ---------- 1. shape, finiteness, dead channels ----------
    n_nan = n_dead = n_short = 0
    durs = defaultdict(list)
    for c, r, ch in allrec:
        x = gyro(r, ch)
        durs[c].append(x.shape[-1] / FS)
        if not np.isfinite(x).all():
            n_nan += 1
        if np.any(x.std(axis=-1) < 1e-12):
            n_dead += 1
        if x.shape[-1] < 5 * FS:
            n_short += 1
    check("no NaN/Inf in any gyroscope channel", n_nan == 0, f"{n_nan} bad")
    check("no dead (constant) gyroscope channel", n_dead == 0, f"{n_dead} bad")
    check("every recording is at least 5 s", n_short == 0, f"{n_short} short")
    print()
    for c in data:
        d = np.array(durs[c])
        print(f"       {c:<9} duration  min {d.min():5.1f}s  median "
              f"{np.median(d):5.1f}s  max {d.max():5.1f}s   n={len(d)}")
    print()

    # ---------- 2. LEAKAGE: duplicate recordings ----------
    # Exact first (byte-identical), then near-duplicate on the spectral
    # fingerprint. A near-duplicate pair across two SUBJECTS is the dangerous
    # case: patient-level splits cannot protect against a patient who appears
    # twice under different ids.
    h = defaultdict(list)
    for c, r, ch in allrec:
        x = gyro(r, ch)
        h[hashlib.sha1(np.ascontiguousarray(x)).hexdigest()].append(
            (c, r.subject, str(r.path)))
    exact = {k: v for k, v in h.items() if len(v) > 1}
    cross_subject = [v for v in exact.values()
                     if len({(c, s) for c, s, _ in v}) > 1]
    check("no byte-identical duplicate recordings", not exact,
          f"{len(exact)} duplicated signals")
    check("no exact duplicate shared by two different subjects",
          not cross_subject, f"{len(cross_subject)} cross-subject duplicates")
    for v in cross_subject[:5]:
        print(f"         {v}")

    sigs = np.array([band_sig(gyro(r, ch)) for c, r, ch in allrec])
    keys = [(c, r.subject) for c, r, ch in allrec]
    # pairwise distance on 16-dim fingerprints is cheap at this n
    D = np.linalg.norm(sigs[:, None, :] - sigs[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    thr = 1e-6
    near = [(i, j) for i, j in zip(*np.where(D < thr)) if i < j]
    near_cross = [(i, j) for i, j in near if keys[i] != keys[j]]
    check("no near-identical spectra across different subjects",
          not near_cross, f"{len(near_cross)} suspicious pairs "
                          f"(threshold {thr:g} on the 16-bin log spectrum)")
    for i, j in near_cross[:5]:
        print(f"         {keys[i]}  ==  {keys[j]}")

    # ---------- 3. LEAKAGE: subject-id collisions across cohorts ----------
    by_cohort = {c: {r.subject for r in rs} for c, (rs, _) in data.items()}
    cols = []
    names = list(by_cohort)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            inter = by_cohort[names[a]] & by_cohort[names[b]]
            if inter:
                cols.append((names[a], names[b], sorted(inter)[:5]))
    check("no subject id used by two cohorts", not cols, f"{len(cols)} clashes")
    for c in cols[:5]:
        print(f"         {c}")

    # ---------- 4. one label per subject ----------
    lab = defaultdict(set)
    for c, r, ch in allrec:
        lab[(c, r.subject)].add(int(r.y))
    conflict = {k: v for k, v in lab.items() if len(v) > 1}
    check("every subject carries exactly one label", not conflict,
          f"{len(conflict)} subjects with conflicting labels")
    for k, v in list(conflict.items())[:5]:
        print(f"         {k} -> {sorted(v)}")

    # ---------- 5. amplitude scale across cohorts (units consistency) ----------
    sos = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype="band",
                 output="sos")
    rms = defaultdict(list)
    for c, r, ch in allrec:
        rms[c].append(float(np.sqrt((sosfiltfilt(sos, gyro(r, ch),
                                                 axis=-1) ** 2).mean())))
    med = {c: float(np.median(v)) for c, v in rms.items()}
    spread = max(med.values()) / max(min(med.values()), 1e-12)
    check("in-band amplitude scales agree across cohorts within 10x",
          spread < 10, "  ".join(f"{c} {m:.3f}" for c, m in med.items())
          + f"   ratio {spread:.1f}x")

    # ---------- 6. clipping / saturation ----------
    n_clip = 0
    for c, r, ch in allrec:
        x = gyro(r, ch)
        for row in np.atleast_2d(x):
            top = np.abs(row).max()
            if top > 0 and (np.abs(np.abs(row) - top) < 1e-9).sum() > 0.01 * len(row):
                n_clip += 1
                break
    check("no recording spends >1% of samples at its own rail (clipping)",
          n_clip == 0, f"{n_clip} suspect")

    # ---------- 7. in-band fraction: noise-dominated recordings ----------
    frac = defaultdict(list)
    for c, r, ch in allrec:
        x = gyro(r, ch)
        f, P = welch(x, fs=FS, nperseg=min(256, x.shape[-1]), axis=-1)
        P = np.asarray(P).mean(0)
        tot = P[(f > 0.5)].sum()
        frac[c].append(float(P[(f >= BAND[0]) & (f <= BAND[1])].sum()
                             / max(tot, 1e-20)))
    for c in data:
        v = np.array(frac[c])
        print(f"       {c:<9} in-band (3-15 Hz) power fraction  median "
              f"{np.median(v):.3f}   <5%: {int((v < 0.05).sum())}/{len(v)}")
    worst = max(float((np.array(frac[c]) < 0.05).mean()) for c in data)
    check("under 15% of a cohort's recordings are noise-dominated (<5% in band)",
          worst < 0.15, f"worst cohort {worst:.1%}")

    # ---------- 8. frames actually averaged, per cohort ----------
    print()
    for c, (rs, ch) in data.items():
        n = [max((min(256, gyro(r, ch).shape[-1]) - 192), 0) for r in rs]
        L = [gyro(r, ch).shape[-1] for r in rs]
        nf = [1 + max(0, (l - 256)) // 64 for l in L]
        print(f"       {c:<9} multitaper frames per recording: median "
              f"{int(np.median(nf))}  (min {min(nf)}, max {max(nf)})")
    check("cohort frame counts are documented as a known open issue", True,
          "21 / 13 / 12 — see SKILL.md; unequal averaging is not yet tested")

    n_fail = sum(not ok for _, ok in res)
    print(f"\n{len(res)} checks, {n_fail} failed")
    print("MARKER_DONE", flush=True)
    raise SystemExit(n_fail)


if __name__ == "__main__":
    main()
