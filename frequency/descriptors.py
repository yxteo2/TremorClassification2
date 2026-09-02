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


#: Half-power bandwidth of the CONTIGUOUS peak (True) or the span of every bin
#: above half-max anywhere in the band (False -- the behaviour before the fix).
#: The span definition collapses Q whenever a second component (a harmonic, a
#: second oscillator) clears half-max: a 6 Hz tone with a 0.8-amplitude 12 Hz
#: harmonic reads Q 0.94 instead of 15. On real recordings that happens to 85 %
#: of PADS N, 74 % of PD and 30 % of ET (stft512), so the old ``q_factor`` was
#: measuring "has secondary spectral content" as much as peak sharpness, and
#: doing so in a class-ordered way. Kept switchable for the paired audit.
Q_CONTIGUOUS = True


def describe(freqs, power):
    """Interpretable descriptors of one spectrum. Returns a dict."""
    f = np.asarray(freqs, dtype=float)
    P = np.clip(np.asarray(power, dtype=float), 0, None)
    if f.size == 0 or P.sum() <= EPS:
        return dict.fromkeys(DESCRIPTOR_NAMES, 0.0)

    # Weight by power x BIN WIDTH, not power alone. Several methods return a
    # non-uniform frequency grid (VMD mode centres, the S-transform), where
    # summing P treats a wide bin and a narrow one as equal and biases
    # mean/median frequency toward wherever the grid happens to be dense.
    if f.size > 1:
        edges = np.concatenate([[f[0]], 0.5*(f[1:] + f[:-1]), [f[-1]]])
        dfb = np.diff(edges)
        dfb[dfb <= 0] = np.median(dfb[dfb > 0]) if (dfb > 0).any() else 1.0
    else:
        dfb = np.ones_like(f)
    Pw = P * dfb
    w = Pw / (Pw.sum() + EPS)                     # power distribution over freq
    pk = int(np.argmax(P))
    max_freq = float(f[pk])
    mean_freq = float((f * w).sum())
    cdf = np.cumsum(w)
    median_freq = float(np.interp(0.5, cdf, f))
    q25 = float(np.interp(0.25, cdf, f))
    q75 = float(np.interp(0.75, cdf, f))
    spread = float(np.sqrt(((f - mean_freq) ** 2 * w).sum()))
    entropy = float(-(w * np.log(w + EPS)).sum() / np.log(len(w) + EPS))

    # Q-factor: peak frequency / half-power bandwidth of the peak itself
    half = P[pk] / 2.0
    if Q_CONTIGUOUS:
        lo = pk
        while lo > 0 and P[lo - 1] >= half:
            lo -= 1
        hi = pk
        while hi < len(P) - 1 and P[hi + 1] >= half:
            hi += 1
        above = np.arange(lo, hi + 1)
    else:                                  # pre-fix: span of ALL supra-half bins
        above = np.where(P >= half)[0]
    # a peak narrower than one bin is reported at one bin's width, never zero
    bw = float(f[above[-1]] - f[above[0]]) if len(above) > 1 else float(np.diff(f).mean())
    q = max_freq / (bw + EPS)

    mid = 0.5 * (f[0] + f[-1])
    lo, hi = Pw[f <= mid].sum(), Pw[f > mid].sum()

    return {
        "max_freq": max_freq,
        "mean_freq": mean_freq,
        "median_freq": median_freq,
        "spectral_spread": spread,
        "spectral_entropy": entropy,
        "q_factor": float(q),
        "peak_share": float(Pw[pk] / (Pw.sum() + EPS)),
        "freq_iqr": q75 - q25,
        "low_high_ratio": float(np.log10((lo + EPS) / (hi + EPS))),
        # integrated power. NOTE: absolute calibration differs per method
        # (verified: integ/true ranges 0.15-1.4e4 across the 12 transforms), so
        # total_power is comparable WITHIN a method, not across methods.
        "total_power": float(np.log10(Pw.sum() + EPS)),
    }
