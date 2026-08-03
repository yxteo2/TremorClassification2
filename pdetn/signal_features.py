"""Physiology-informed tremor signal-processing features (the front end).

Focus: features that describe *how the tremor behaves*, not just where its
spectral peak sits — because that is what should separate PD from ET:

  * PD rest tremor: sharp, narrow spectral peak; highly regular/periodic;
    stable frequency over time.
  * ET action tremor: broader peak; more irregular; frequency wanders more.

All features are computed from a **tremor-band-isolated** (3-15 Hz bandpass)
signal, using the vector magnitude of the hand sensor's 3 angular-velocity axes
(rotation-invariant). Torch-free; numpy + scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch, get_window

HAND_CHANNELS = (0, 1, 2)
TREMOR_LO, TREMOR_HI = 3.0, 15.0


def _tremor_magnitude(x: np.ndarray, fs: float, channels=HAND_CHANNELS,
                      mode: str = "magnitude") -> np.ndarray:
    """Bandpass the axes to 3-15 Hz and reduce to one 1-D tremor signal.

    mode:
      * ``magnitude`` — vector magnitude sqrt(sum(ch^2)). NOTE this **rectifies**
        the signal: squaring a 6 Hz oscillation puts energy at 12 Hz, so
        frequency-derived features measure the envelope, not the tremor.
      * ``pc1`` — project onto the first principal component of the bandpassed
        axes, i.e. the dominant axis of oscillation. Keeps the *signed*
        oscillation, so frequencies are the true tremor frequencies.
    """
    sos = butter(4, [TREMOR_LO, TREMOR_HI], btype="band", fs=fs, output="sos")
    ax = x[list(channels)]
    filt = sosfiltfilt(sos, ax, axis=-1)               # (C, T)
    if mode == "magnitude":
        return np.sqrt(np.sum(filt ** 2, axis=0))      # (T,) rectified
    if mode == "pc1":
        X = filt.T - filt.T.mean(axis=0)               # (T, C)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[0]                               # (T,) signed
    raise ValueError(f"mode must be magnitude|pc1, got {mode!r}")


def _hjorth(sig: np.ndarray) -> tuple[float, float]:
    """Hjorth mobility and complexity (form/regularity descriptors)."""
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    v0 = np.var(sig) + 1e-12
    v1 = np.var(d1) + 1e-12
    v2 = np.var(d2) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + 1e-12)
    return float(mob), float(comp)


def _sample_entropy(sig: np.ndarray, m: int = 2, r: float = 0.2,
                    max_len: int = 600) -> float:
    """Sample entropy (regularity: low = periodic/PD-like, high = irregular).

    Signal is decimated to at most ``max_len`` samples for O(N^2) tractability.
    """
    s = sig[:: max(1, len(sig) // max_len)]
    s = (s - s.mean()) / (s.std() + 1e-12)
    n = len(s)
    tol = r
    def _phi(mm):
        templates = np.array([s[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(d <= tol) - 1        # exclude self-match
        return count
    B = _phi(m); A = _phi(m + 1)
    if B == 0 or A == 0:
        return float("nan")
    return float(-np.log(A / B))


def _spectral_shape(sig: np.ndarray, fs: float) -> dict:
    """Peak sharpness (Q), flatness, centroid, spread over the tremor band."""
    nper = int(min(256, len(sig)))
    f, P = welch(sig, fs=fs, nperseg=nper, noverlap=nper // 2)
    band = (f >= TREMOR_LO) & (f < TREMOR_HI)
    f, P = f[band], P[band] + 1e-18
    k = int(np.argmax(P))
    f0, pk = f[k], P[k]
    # Q-factor: peak freq / half-power (-3 dB) bandwidth.
    half = pk / 2.0
    above = np.where(P >= half)[0]
    bw = (f[above[-1]] - f[above[0]]) if len(above) >= 2 else (f[1] - f[0])
    q = f0 / (bw + 1e-9)
    # spectral flatness (geo/arith mean): peaky -> low, noisy -> ~1
    flat = float(np.exp(np.mean(np.log(P))) / np.mean(P))
    centroid = float(np.sum(f * P) / np.sum(P))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * P) / np.sum(P)))
    return {"peak_q_factor": float(q), "spectral_flatness": flat,
            "spectral_centroid": centroid, "spectral_spread": spread,
            "halfpower_bw": float(bw)}


def _freq_stability(sig: np.ndarray, fs: float) -> float:
    """Std (Hz) of the per-frame dominant frequency — low = stable (PD-like)."""
    win = int(min(128, len(sig)))
    if win < 32:
        return 0.0
    step = max(1, win // 2)
    w = get_window("hann", win)
    freqs = np.fft.rfftfreq(win, 1 / fs)
    band = (freqs >= TREMOR_LO) & (freqs < TREMOR_HI)
    peaks = []
    for start in range(0, len(sig) - win + 1, step):
        seg = sig[start:start + win] * w
        mag = np.abs(np.fft.rfft(seg))[band]
        if mag.size and mag.max() > 0:
            peaks.append(freqs[band][int(np.argmax(mag))])
    return float(np.std(peaks)) if len(peaks) >= 2 else 0.0


def _amplitude_modulation(sig: np.ndarray) -> float:
    """Envelope coefficient of variation (tremor amplitude steadiness)."""
    env = np.abs(hilbert(sig))
    return float(np.std(env) / (np.mean(env) + 1e-12))


def advanced_features(x: np.ndarray, fs: float = 100.0,
                      channels=HAND_CHANNELS,
                      mode: str = "magnitude") -> dict[str, float]:
    """All physiology-informed signal features for one recording.

    ``mode='pc1'`` avoids the rectification artifact (see _tremor_magnitude).
    """
    sig = _tremor_magnitude(x, fs, channels, mode=mode)
    feats: dict[str, float] = {}
    feats.update(_spectral_shape(sig, fs))
    mob, comp = _hjorth(sig)
    feats["hjorth_mobility"] = mob
    feats["hjorth_complexity"] = comp
    feats["sample_entropy"] = _sample_entropy(sig)
    feats["freq_stability_std"] = _freq_stability(sig, fs)
    feats["am_depth"] = _amplitude_modulation(sig)
    return feats


ADVANCED_FEATURE_NAMES = (
    "peak_q_factor", "spectral_flatness", "spectral_centroid", "spectral_spread",
    "halfpower_bw", "hjorth_mobility", "hjorth_complexity", "sample_entropy",
    "freq_stability_std", "am_depth",
)
