"""Interpretable frequency descriptors computed from any method's spectrum.

These are the quantities the comparison is *about*: max frequency, mean
frequency and their relatives. Every method in `transforms.METHODS` is reduced
to a (freqs, power) pair, so all methods are described by the same numbers and
the comparison is like-for-like.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-20

DESCRIPTOR_NAMES = [
    "max_freq", "mean_freq", "median_freq", "spectral_spread",
    "spectral_entropy", "q_factor", "peak_share", "freq_iqr",
    "low_high_ratio", "total_power",
]


def describe(freqs, power):
    """Interpretable descriptors of one spectrum. Returns a dict."""
    f = np.asarray(freqs, dtype=float)
    P = np.clip(np.asarray(power, dtype=float), 0, None)
    if f.size == 0 or P.sum() <= EPS:
        return dict.fromkeys(DESCRIPTOR_NAMES, 0.0)

    w = P / (P.sum() + EPS)                       # power distribution over freq
    pk = int(np.argmax(P))
    max_freq = float(f[pk])
    mean_freq = float((f * w).sum())
    cdf = np.cumsum(w)
    median_freq = float(np.interp(0.5, cdf, f))
    q25 = float(np.interp(0.25, cdf, f))
    q75 = float(np.interp(0.75, cdf, f))
    spread = float(np.sqrt(((f - mean_freq) ** 2 * w).sum()))
    entropy = float(-(w * np.log(w + EPS)).sum() / np.log(len(w) + EPS))

    # Q-factor: peak frequency / half-power bandwidth
    half = P[pk] / 2.0
    above = np.where(P >= half)[0]
    bw = float(f[above[-1]] - f[above[0]]) if len(above) > 1 else float(np.diff(f).mean())
    q = max_freq / (bw + EPS)

    mid = 0.5 * (f[0] + f[-1])
    lo, hi = P[f <= mid].sum(), P[f > mid].sum()

    return {
        "max_freq": max_freq,
        "mean_freq": mean_freq,
        "median_freq": median_freq,
        "spectral_spread": spread,
        "spectral_entropy": entropy,
        "q_factor": float(q),
        "peak_share": float(P[pk] / (P.sum() + EPS)),
        "freq_iqr": q75 - q25,
        "low_high_ratio": float(np.log10((lo + EPS) / (hi + EPS))),
        "total_power": float(np.log10(P.sum() + EPS)),
    }
