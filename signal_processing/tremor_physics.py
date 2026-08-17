"""The four clinical signal families, implemented properly.

Earlier work here dismissed two of these on evidence that did not actually test
them. This implements all four as described clinically, with the confounds that
killed the first attempt removed.

**1. Frequency & harmonics.** PD rest tremor is often non-sinusoidal, producing
2nd and 3rd harmonics; ET is closer to a single tone. Previously only 2f was
computed. Adds 3f and a harmonic-to-noise ratio.

**2. Orientation & axes.** The clinical distinction is pronation-supination
(rotational) in PD against flexion-extension in ET. Every previous attempt here
used mount-dependent quantities -- log map, body-frame gravity, per-axis
fusion -- and all failed, because **wrist-mount orientation is not recorded** and
varies between patients, so those features partly measure how the watch was
strapped on.

The **eigenvalues of the 3x3 cross-spectral matrix are rotation-invariant**: they
describe whether the oscillation is confined to a line, a plane, or fills three
dimensions, without reference to any coordinate frame. A pronation-supination
tremor is close to a single rotational axis (linear); a multi-axis action tremor
fills more dimensions. That is the confound-free form of the distinction, and it
had never been tested.

**3. Amplitude changes.** PD shows waxing-and-waning bursts; ET a continuous
stationary envelope. Previously only `amp_cv`, a single scalar. Adds the
**modulation spectrum** -- the FFT of the Hilbert envelope, which says at what
RATE the amplitude fluctuates -- and burst statistics.

**4. Raw amplitude.** Included for completeness and expected to be weak: the
literature is explicit that amplitude indexes severity, not diagnosis, and
between-subject variability (331 %) dwarfs within-subject (53.6 %).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch

HARMONIC = ("h2_ratio", "h3_ratio", "harmonic_to_noise")
AXES = ("linearity", "planarity", "sphericity", "axis_entropy")
AMPMOD = ("mod_peak_hz", "mod_low_frac", "mod_entropy", "burst_frac",
          "burst_rate", "env_cv")
AMPLITUDE = ("rms",)
FEATURE_NAMES = HARMONIC + AXES + AMPMOD + AMPLITUDE
FAMILIES = {"harmonic": HARMONIC, "axes": AXES, "ampmod": AMPMOD,
            "amplitude": AMPLITUDE}


def _dominant(f, P, f_lo, f_hi):
    m = (f >= f_lo) & (f <= f_hi)
    if not m.any() or P[m].sum() <= 0:
        return float("nan")
    return float(f[m][np.argmax(P[m])])


def harmonic_features(x, fs=100.0, f_lo=3.0, f_hi=15.0):
    """Power at 2f0 and 3f0 relative to f0, plus harmonic-to-noise ratio."""
    x = np.atleast_2d(np.asarray(x, float))
    n = int(min(512, x.shape[-1]))
    f, P = welch(x, fs=fs, nperseg=n, axis=-1)
    P = P.mean(0)
    f0 = _dominant(f, P, f_lo, f_hi)
    out = {k: float("nan") for k in HARMONIC}
    if not np.isfinite(f0) or f0 <= 0:
        return out

    def band(c, half=0.75):
        m = (f >= c - half) & (f <= c + half)
        return float(P[m].sum()) if m.any() else 0.0

    p1, p2, p3 = band(f0), band(2 * f0), band(3 * f0)
    out["h2_ratio"] = p2 / (p1 + 1e-20)
    out["h3_ratio"] = p3 / (p1 + 1e-20)
    tot = float(P[(f >= 0.5) & (f <= 40)].sum())
    harm = p1 + p2 + p3
    out["harmonic_to_noise"] = harm / (tot - harm + 1e-20)
    return out


def axis_features(x, fs=100.0, f_lo=3.0, f_hi=15.0):
    """Rotation-INVARIANT description of the oscillation's spatial shape.

    Builds the 3x3 cross-spectral matrix averaged over the tremor band and takes
    its eigenvalues. Under any rotation R the matrix becomes R S R^T, which has
    the same eigenvalues -- so these features do not depend on how the sensor was
    mounted, which is what killed every previous orientation attempt here.

    ``linearity``   lam1 / sum   -- 1 means a single oscillation axis
    ``planarity``   (lam1-lam2) / sum
    ``sphericity``  3*lam3 / sum -- 1 means power spread equally over 3 axes
    ``axis_entropy`` entropy of the normalised eigenvalues
    """
    out = {k: float("nan") for k in AXES}
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[0] < 3:
        return out
    x = x[:3] - x[:3].mean(1, keepdims=True)
    n = x.shape[-1]
    win = np.hanning(n)
    X = np.fft.rfft(x * win, axis=-1)
    f = np.fft.rfftfreq(n, 1 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    if not m.any():
        return out
    Xb = X[:, m]
    S = (Xb @ Xb.conj().T).real / m.sum()          # 3x3, rotation-covariant
    w = np.linalg.eigvalsh(S)[::-1]
    w = np.clip(w, 0, None)
    tot = w.sum()
    if tot <= 0:
        return out
    p = w / tot
    out["linearity"] = float(p[0])
    out["planarity"] = float(p[0] - p[1])
    out["sphericity"] = float(3 * p[2])
    out["axis_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(3))
    return out


def ampmod_features(x, fs=100.0, f_lo=3.0, f_hi=15.0, smooth_s=0.15):
    """Waxing-and-waning: the modulation spectrum and burst statistics.

    The Hilbert envelope says how strong the tremor is over time; its own
    spectrum -- the **modulation spectrum** -- says at what RATE that strength
    fluctuates. PD's waxing and waning is a slow modulation; a continuous ET
    envelope has little modulation energy at any rate.
    """
    out = {k: float("nan") for k in AMPMOD}
    x = np.atleast_2d(np.asarray(x, float))
    lo, hi = f_lo / (fs / 2), min(f_hi / (fs / 2), 0.99)
    try:
        b, a = butter(4, [lo, hi], btype="band")
        xb = filtfilt(b, a, x, axis=-1)
    except Exception:
        return out
    env = np.abs(hilbert(xb, axis=-1)).mean(0)
    env = uniform_filter1d(env, max(int(smooth_s * fs), 1))
    if len(env) < int(2 * fs) or env.mean() <= 0:
        return out

    out["env_cv"] = float(env.std() / (env.mean() + 1e-12))

    # modulation spectrum of the (mean-removed) envelope
    e = env - env.mean()
    nper = int(min(len(e), 4 * fs))
    fm, Pm = welch(e, fs=fs, nperseg=nper)
    band = (fm > 0.05) & (fm <= 5.0)
    if band.any() and Pm[band].sum() > 0:
        pm = Pm[band] / Pm[band].sum()
        out["mod_peak_hz"] = float(fm[band][np.argmax(pm)])
        low = (fm[band] >= 0.05) & (fm[band] <= 1.0)
        out["mod_low_frac"] = float(pm[low].sum())
        out["mod_entropy"] = float(-(pm * np.log(pm + 1e-12)).sum()
                                   / np.log(len(pm)))

    # bursts: time spent above the median envelope, and how often it crosses
    thr = np.median(env)
    above = env > thr
    out["burst_frac"] = float(above.mean())
    crossings = int(np.sum(np.diff(above.astype(int)) == 1))
    out["burst_rate"] = float(crossings / (len(env) / fs))
    return out


def amplitude_features(x, fs=100.0, f_lo=3.0, f_hi=15.0):
    """In-band RMS. Expected weak: severity, not diagnosis."""
    x = np.atleast_2d(np.asarray(x, float))
    n = int(min(512, x.shape[-1]))
    f, P = welch(x, fs=fs, nperseg=n, axis=-1)
    P = P.mean(0)
    m = (f >= f_lo) & (f <= f_hi)
    return {"rms": float(np.sqrt(P[m].sum())) if m.any() else float("nan")}


def recording_features(x, fs=100.0, **kw):
    d = {}
    d.update(harmonic_features(x, fs=fs, **kw))
    d.update(axis_features(x, fs=fs, **kw))
    d.update(ampmod_features(x, fs=fs, **kw))
    d.update(amplitude_features(x, fs=fs, **kw))
    return d


def patient_table(recs, ch=slice(0, 3), fs=100.0, **kw):
    """(patients, 14) across all four families, averaged per patient."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        d = recording_features(sig, fs=fs, **kw)
        rows[r.subject].append([d[k] for k in FEATURE_NAMES])
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.nanmean(rows[p], axis=0) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))
