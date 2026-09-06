"""Fixed wavelet modulation features; no fitted filters or patient statistics.

Anden & Mallat, Deep Scattering Spectrum, IEEE TSP 62(16), 2014:
https://arxiv.org/abs/1304.6763
This is a tremor adaptation to test, not a demonstrated clinical improvement.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from signal_processing.waveform import waveform, LENGTH, FS_OUT


@lru_cache(maxsize=1)
def _transform():
    # Lazy frontend avoids importing Kymatio's unrelated 3-D scipy dependencies.
    from kymatio import Scattering1D
    return Scattering1D(J=5, shape=LENGTH, Q=4, max_order=2, frontend="numpy")


def from_waveform(w):
    """Return (first-order, first+second-order) for a 384-point 40 Hz signal.

    Retain 3-15 Hz carrier filters. First order measures local oscillation
    strength; second order measures changes in its envelope. Average coefficients
    over time, not waveforms across recordings. Normalize second order by its
    parent first order before log compression. Constants are fixed a priori.
    """
    w = np.asarray(w, dtype=float)
    if w.shape != (LENGTH,) or not np.isfinite(w).all():
        raise ValueError(f"Expected a finite ({LENGTH},) waveform")
    transform = _transform()
    meta = transform.meta()
    s = np.maximum(transform(w).mean(axis=-1), 0)
    carriers = meta["xi"][:, 0] * FS_OUT
    band = (carriers >= 3) & (carriers <= 15)
    first = np.flatnonzero((meta["order"] == 1) & band)
    second = np.flatnonzero((meta["order"] == 2) & band)
    lookup = {key: i for i, key in enumerate(meta["key"])}
    parents = [lookup[meta["key"][i][:1]] for i in second]
    f1 = np.log(np.maximum(s[first], 1e-8))
    f2 = np.log(np.maximum(s[second] / np.maximum(s[parents], 1e-8), 1e-8))
    return f1, np.concatenate([f1, f2])


def patient_table(recs, ch=slice(3, 6)):
    """One row per patient, averaging ALL usable recordings with equal weight.

    Matches the existing postural waveform preprocessing. Reject short or
    nonfinite records rather than allowing padding to identify a cohort.
    Returns first order, both orders, labels, patient IDs, and exclusions.
    This fixed central-window representation is NOT an action-phase encoder.
    """
    rows, labels, excluded = {}, {}, []
    for r in recs:
        if r.subject in labels and labels[r.subject] != r.y:
            raise ValueError(f"Conflicting labels for patient {r.subject}")
        labels[r.subject] = r.y
        x = np.asarray(r.x[ch] if r.x.shape[0] > 3 else r.x)
        if x.shape[-1] < 960 or not np.isfinite(x).all():
            excluded.append(str(r.path))
            continue
        w = waveform(x)
        if w is None:
            excluded.append(str(r.path))
            continue
        rows.setdefault(r.subject, []).append(from_waveform(w))
    if not rows:
        raise ValueError("No usable recordings for scattering features")
    patients = np.array(sorted(rows))
    first = np.array([np.mean([v[0] for v in rows[p]], axis=0) for p in patients])
    both = np.array([np.mean([v[1] for v in rows[p]], axis=0) for p in patients])
    return first, both, np.array([labels[p] for p in patients]), patients, excluded
