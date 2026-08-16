"""Tremor Stability Index and related temporal-fluctuation features.

Motivated by a literature finding that contradicts this repo's whole feature
strategy. Di Biase et al., *Brain* 2017 ("Tremor stability index: a new tool for
differential diagnosis in tremor syndromes") report that PD and ET patients do
**not** differ in peak power frequency, median power frequency, power
dispersion, harmonic index or relative power contribution -- i.e. in precisely
the static spectral quantities that every descriptor and every spectrum model
here is built from.

What does separate them is the stability of the **instantaneous** frequency over
time. In essential tremor the cycle-to-cycle frequency stays inside a narrow
band; in Parkinsonian tremor it wanders over a broader one. Their Tremor
Stability Index thresholds this at 1.05.

This is not the same quantity as "does the spectrum change over time", which was
tested here and found to be at chance -- a BiLSTM over spectrogram frames asks
whether spectral SHAPE evolves. Instantaneous-frequency stability is a different
measurement and had never been computed.

Features implemented:

``tsi``            Tremor Stability Index: the width of the central 90 % of the
                   cycle-to-cycle frequency-change distribution.
``if_std``         standard deviation of the Hilbert instantaneous frequency.
``if_iqr``         interquartile range of the same.
``amp_cv``         coefficient of variation of the instantaneous amplitude --
                   PD rest tremor is classically more amplitude-modulated.
``autocorr_decay`` decay of the envelope autocorrelation, the "waveform
                   asymmetry" measure reported alongside TSI.
``harm_ratio``     power at 2f relative to power at f.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch

FEATURE_NAMES = ("tsi", "if_std", "if_iqr", "amp_cv", "autocorr_decay",
                 "harm_ratio")


def _dominant_freq(x, fs, lo, hi):
    f, P = welch(x, fs=fs, nperseg=min(512, len(x)))
    m = (f >= lo) & (f <= hi)
    if not m.any():
        return float("nan")
    return float(f[m][np.argmax(P[m])])


def stability_features(x, fs=100.0, f_lo=3.0, f_hi=15.0, bw=2.0):
    """Temporal-fluctuation features from ONE channel of angular velocity.

    The signal is band-passed around its own dominant tremor frequency before
    the Hilbert transform, because instantaneous frequency is only meaningful
    for a narrowband signal -- applied to a broadband recording it is dominated
    by noise rather than by the tremor.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    out = {k: float("nan") for k in FEATURE_NAMES}
    if len(x) < int(2 * fs):
        return out

    f0 = _dominant_freq(x, fs, f_lo, f_hi)
    if not np.isfinite(f0) or f0 <= 0:
        return out

    lo = max(f0 - bw, 0.5) / (fs / 2)
    hi = min(f0 + bw, fs / 2 - 1.0) / (fs / 2)
    if not (0 < lo < hi < 1):
        return out
    try:
        b, a = butter(4, [lo, hi], btype="band")
        xb = filtfilt(b, a, x)
    except Exception:
        return out

    z = hilbert(xb)
    phase = np.unwrap(np.angle(z))
    amp = np.abs(z)
    inst_f = np.diff(phase) * fs / (2 * np.pi)
    # keep only physically plausible instantaneous frequencies
    inst_f = inst_f[(inst_f > f_lo * 0.5) & (inst_f < f_hi * 1.5)]
    if len(inst_f) < 20:
        return out

    # TSI: width of the central 90 % of the cycle-to-cycle frequency change
    d = np.diff(inst_f)
    out["tsi"] = float(np.percentile(d, 95) - np.percentile(d, 5))
    out["if_std"] = float(np.std(inst_f))
    out["if_iqr"] = float(np.percentile(inst_f, 75) - np.percentile(inst_f, 25))
    out["amp_cv"] = float(np.std(amp) / (np.mean(amp) + 1e-12))

    # envelope autocorrelation decay -- how fast tremor amplitude decorrelates
    e = amp - amp.mean()
    if np.dot(e, e) > 0:
        n = min(len(e) - 1, int(fs))
        ac = np.correlate(e, e, mode="full")[len(e) - 1:len(e) - 1 + n]
        ac = ac / (ac[0] + 1e-12)
        below = np.flatnonzero(ac < 1 / np.e)
        out["autocorr_decay"] = float(below[0] / fs) if below.size else float(n / fs)

    f, P = welch(x, fs=fs, nperseg=min(512, len(x)))
    def band(c):
        m = (f >= c - 0.75) & (f <= c + 0.75)
        return float(P[m].sum()) if m.any() else 0.0
    p1 = band(f0)
    out["harm_ratio"] = float(band(2 * f0) / (p1 + 1e-20)) if p1 > 0 else 0.0
    return out


def recording_features(x, fs=100.0, **kw):
    """Average the stability features over the channels of one recording."""
    x = np.atleast_2d(np.asarray(x))
    rows = [stability_features(ch, fs=fs, **kw) for ch in x]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in FEATURE_NAMES}


def patient_table(recs, ch=slice(0, 3), fs=100.0, **kw):
    """(patients, 6) stability features, patient order matching spectrum_table."""
    from collections import defaultdict
    rows, lab = defaultdict(list), {}
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        d = recording_features(sig, fs=fs, **kw)
        rows[r.subject].append([d[k] for k in FEATURE_NAMES])
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.nanmean(rows[p], axis=0) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))
