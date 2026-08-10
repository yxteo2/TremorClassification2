"""Temporal descriptors — what the tremor DOES over time, not its average spectrum.

Every descriptor in :mod:`tfbench.descriptors` is computed from a spectrum that
has already been averaged over time, so the time axis is discarded before any
feature is measured. That throws away the dynamics: whether tremor comes in
bursts or runs continuously, whether its frequency is stable or wanders, how
deeply it is amplitude-modulated.

Those are clinically meaningful. PD rest tremor is classically intermittent and
re-emergent; ET is more continuous and action-linked. Nothing measured so far in
this repo could see that difference.

These operate on the (freq, time) STFT matrix and summarise **along time**.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft, welch

EPS = 1e-20

TEMPORAL_FEATURE_NAMES = [
    "amp_cv", "burst_fraction", "longest_burst_s", "n_bursts_per_s",
    "freq_wander", "freq_wander_iqr", "spectral_flux", "flux_cv",
    "env_autocorr_1s", "env_rhythm_freq", "env_rhythm_strength",
    "band_power_slope", "stationarity",
]


def _tf(x, fs, nperseg, noverlap, f_lo, f_hi):
    """(freqs, times, power) restricted to the tremor band, channels averaged."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n = min(nperseg, x.shape[-1])
    f, t, Z = stft(x, fs=fs, nperseg=n, noverlap=min(noverlap, n - 1), nfft=n,
                   axis=-1, boundary=None, padded=False)
    P = (np.abs(Z) ** 2).mean(0)                 # (F, T), power not amplitude
    k = (f >= f_lo) & (f <= f_hi)
    return f[k], t, P[k]


def temporal_features(x, fs=100.0, nperseg=256, noverlap=224, f_lo=3.0, f_hi=15.0):
    """Dynamics of the tremor band over time. Returns a dict."""
    f, t, P = _tf(x, fs, nperseg, noverlap, f_lo, f_hi)
    if P.size == 0 or P.shape[1] < 4:
        return dict.fromkeys(TEMPORAL_FEATURE_NAMES, 0.0)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / fs
    band = P.sum(0)                              # tremor-band power per frame
    band_n = band / (band.mean() + EPS)          # scale-free envelope

    # --- amplitude modulation -------------------------------------------------
    amp_cv = float(band.std() / (band.mean() + EPS))

    # --- bursting: time spent above the MEAN band power -----------------------
    # Thresholding at the median would return 0.5 by construction. The mean is
    # pulled up by bursts, so a bursty recording spends LESS than half its time
    # above it while a continuous one sits near 0.5 -- the statistic carries the
    # skew of the power distribution, which is the thing of interest.
    thr = band.mean()
    above = band > thr
    burst_fraction = float(above.mean())
    # run lengths of consecutive above-threshold frames
    runs, cur = [], 0
    for a in above:
        if a:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    longest = float(max(runs) * dt) if runs else 0.0
    n_bursts = float(len(runs) / (len(band) * dt + EPS))

    # --- does the peak frequency hold still? ----------------------------------
    pk = f[np.argmax(P, axis=0)]
    freq_wander = float(pk.std())
    freq_wander_iqr = float(np.percentile(pk, 75) - np.percentile(pk, 25))

    # --- how fast the spectral SHAPE changes ----------------------------------
    Pn = P / (P.sum(0, keepdims=True) + EPS)
    flux = np.sqrt(((np.diff(Pn, axis=1)) ** 2).sum(0))
    spectral_flux = float(flux.mean())
    flux_cv = float(flux.std() / (flux.mean() + EPS))

    # --- is the modulation itself rhythmic? -----------------------------------
    e = band_n - band_n.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac = ac / (ac[0] + EPS)
    lag1s = min(int(round(1.0 / dt)), len(ac) - 1)
    env_ac_1s = float(ac[lag1s]) if lag1s > 0 else 0.0
    fe, Pe = welch(e, fs=1.0 / dt, nperseg=min(64, len(e)))
    valid = fe > 0
    if valid.any():
        env_rhythm_freq = float(fe[valid][np.argmax(Pe[valid])])
        env_rhythm_strength = float(Pe[valid].max() / (Pe[valid].sum() + EPS))
    else:
        env_rhythm_freq = env_rhythm_strength = 0.0

    # --- drift and stationarity ----------------------------------------------
    tt = np.arange(len(band)) * dt
    slope = float(np.polyfit(tt, band_n, 1)[0]) if len(band) > 2 else 0.0
    h = len(band) // 2
    a, b = band[:h].mean(), band[h:].mean()
    stationarity = float(min(a, b) / (max(a, b) + EPS))

    return {"amp_cv": amp_cv, "burst_fraction": burst_fraction,
            "longest_burst_s": longest, "n_bursts_per_s": n_bursts,
            "freq_wander": freq_wander, "freq_wander_iqr": freq_wander_iqr,
            "spectral_flux": spectral_flux, "flux_cv": flux_cv,
            "env_autocorr_1s": env_ac_1s, "env_rhythm_freq": env_rhythm_freq,
            "env_rhythm_strength": env_rhythm_strength,
            "band_power_slope": slope, "stationarity": stationarity}
