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


# --------------------------------------------------------------------------- #
# Trajectories -- the deep-learning form of the same idea
# --------------------------------------------------------------------------- #
def if_trajectory(x, fs=100.0, f_lo=3.0, f_hi=15.0, bw=2.0, n_out=64,
                  smooth=5, guard_s=0.25):
    """Instantaneous-frequency and envelope TRAJECTORY for a deep model.

    :func:`stability_features` compresses the instantaneous frequency to six
    scalars. That throws away the shape of the fluctuation, which is exactly
    what a sequence model exists to learn.

    This returns the trajectory itself as a ``(2, n_out)`` array -- normalised
    instantaneous frequency and normalised envelope, resampled to a fixed
    length.

    Note on an earlier conclusion in this repo: "a sequence model over the time
    axis is at chance" was measured on RAW SPECTROGRAM FRAMES, a 61-dimensional
    noisy sequence. This is a 1-D physically meaningful trajectory, and is not
    the same test.
    """
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.ndimage import uniform_filter1d

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    out = np.zeros((2, n_out), dtype=np.float32)
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
    inst_f = np.diff(np.unwrap(np.angle(z))) * fs / (2 * np.pi)
    amp = np.abs(z)[1:]
    # Drop the band-pass / Hilbert edge transient. Measured on a synthetic tone
    # the instantaneous frequency is wrong for the first and last 10-16 samples,
    # and because the series is resampled to n_out points the raw ends land
    # exactly on trajectory points 0 and n_out-1: a rock-steady 6 Hz tone read
    # 0.36 Hz of wander at point 0, and a 6 +/- 0.5 Hz FM tone read 2.7 Hz at
    # point 63, against an interior that was correct to 0.06 Hz. A 4 Hz-wide
    # 4th-order filter settles in ~0.25 s, so that is the guard; guard_s=0
    # reproduces the pre-fix behaviour for the paired audit.
    gd = int(round(guard_s * fs))
    if gd > 0 and len(inst_f) > 4 * gd:
        inst_f, amp = inst_f[gd:-gd], amp[gd:-gd]
    ok = (inst_f > f_lo * 0.5) & (inst_f < f_hi * 1.5)
    if ok.sum() < 20:
        return out
    inst_f, amp = inst_f[ok], amp[ok]
    if smooth > 1:
        inst_f = uniform_filter1d(inst_f, smooth)
        amp = uniform_filter1d(amp, smooth)
    g = np.linspace(0, len(inst_f) - 1, n_out)
    src = np.arange(len(inst_f))
    fr = np.interp(g, src, inst_f)
    en = np.interp(g, src, amp)
    # Centre on the patient's own tremor rate so the trajectory encodes
    # FLUCTUATION rather than absolute rate (the spectrum branch already
    # supplies the rate) -- but scale by a FIXED constant, never by the
    # trajectory's own std. Dividing by its own std sets the variance to 1 and
    # destroys the fluctuation magnitude, which is exactly what the Tremor
    # Stability Index measures: a rock-steady tremor and a wildly wandering one
    # come out identical.
    out[0] = np.clip((fr - fr.mean()) / 1.0, -5, 5)      # Hz, centred
    out[1] = np.clip(en / (en.mean() + 1e-12) - 1.0, -5, 5)   # relative envelope
    return out


def _dominant_axis(sig, fs, f_lo, f_hi):
    """Index of the axis carrying the most in-band power."""
    from scipy.signal import welch as _w
    best, bi = -1.0, 0
    for i, c in enumerate(sig):
        f, P = _w(c, fs=fs, nperseg=min(512, len(c)))
        m = (f >= f_lo) & (f <= f_hi)
        p = float(P[m].sum()) if m.any() else 0.0
        if p > best:
            best, bi = p, i
    return bi


def _pca1(sig):
    """Projection onto the dominant oscillation direction of the 3 axes."""
    X = np.asarray(sig, float)
    X = X - X.mean(1, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return Vt[0] @ X if Vt.shape[0] else X.mean(0)
    except Exception:
        return X.mean(0)


def trajectory_table(recs, ch=slice(0, 3), fs=100.0, n_out=64,
                     axis_mode="mean", f_lo=3.0, f_hi=15.0, **kw):
    """(patients, C, n_out) instantaneous-frequency / envelope trajectories.

    ``axis_mode`` decides how the three angular-velocity axes are handled, and
    it matters more than it looks:

    ``mean``      average the three axes' trajectories. **Default, and measured
                  best or tied.**
    ``dominant``  the single axis with the most in-band power.
    ``pca``       project onto the dominant oscillation direction.
    ``stack``     all three axes as 6 channels.

    A note on a correction, because the obvious argument here is wrong.
    Averaging the axes damps the absolute IF fluctuation by 1.61x (close to the
    sqrt(3) expected if the axes were independent), which looks like it must be
    discarding the quantity the Tremor Stability Index measures. It is not: the
    pipeline standardises features per fold, so a uniform scale change never
    reaches the model. The quantity that matters is the standardised effect
    size, and there the difference is small -- Cohen's d for PD vs ET is 1.238
    (mean), 1.405 (dominant), 1.331 (pca), 0.786 (stack), because the
    within-class spread grows along with the gap.

    Measured in the model, 20 splits: macroP 0.660 (mean), 0.662 (dominant),
    0.643 (pca), 0.649 (stack) -- indistinguishable at the top, and ``stack``
    is worse, as its effect size predicts.
    """
    from collections import defaultdict
    rows, lab = defaultdict(list), {}
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        sig = np.atleast_2d(np.asarray(sig))
        if axis_mode == "mean":
            t = np.mean([if_trajectory(c, fs=fs, n_out=n_out, f_lo=f_lo,
                                       f_hi=f_hi, **kw) for c in sig], 0)
        elif axis_mode == "dominant":
            t = if_trajectory(sig[_dominant_axis(sig, fs, f_lo, f_hi)], fs=fs,
                              n_out=n_out, f_lo=f_lo, f_hi=f_hi, **kw)
        elif axis_mode == "pca":
            t = if_trajectory(_pca1(sig), fs=fs, n_out=n_out, f_lo=f_lo,
                              f_hi=f_hi, **kw)
        elif axis_mode == "stack":
            t = np.concatenate([if_trajectory(c, fs=fs, n_out=n_out, f_lo=f_lo,
                                              f_hi=f_hi, **kw) for c in sig], 0)
        else:
            raise ValueError(axis_mode)
        rows[r.subject].append(t)
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.mean(rows[p], axis=0) for p in pats], dtype=np.float32)
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))
