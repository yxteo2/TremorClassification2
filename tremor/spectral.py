"""Preprocessing for precomputed STFT magnitude matrices.

All ops act on arrays of shape (features, time) where features is
``n_sensors * n_freq_bins``. The block layout is sensor-major: the
first ``n_freq_bins`` rows belong to sensor 0, the next ``n_freq_bins``
to sensor 1, etc. — matching how Drive's ``amptoSTFT`` stacks them
(``hstack((stft_L.T, stft_H.T, stft_U.T))`` then transposed).
"""

from __future__ import annotations

import numpy as np


def log_compress(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """``log(1 + S/eps)`` — bounded, monotonic, zero-preserving."""
    return np.log1p(np.maximum(S, 0.0) / eps).astype(S.dtype, copy=False)


def per_freq_zscore(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each frequency row across time within one recording.

    WARNING: tremor signals are quasi-stationary, so the magnitude of
    the tremor bin barely varies across time. Dividing each row by its
    own time-axis std flattens that bin to zero and erases the
    spectral peak — i.e. erases the very feature that distinguishes
    PD (3-7 Hz) from ET (4-12 Hz). Prefer ``per_recording_zscore``
    unless your signal is genuinely non-stationary across the window.
    """
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + eps
    return ((S - mu) / sd).astype(S.dtype, copy=False)


def per_recording_zscore(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score over all (freq, time) bins of one recording.

    Removes inter-recording amplitude differences (sensor calibration,
    patient-specific gain) while preserving the relative magnitudes
    across freq bins — i.e. the spectral profile that the classifier
    needs. Recommended default for tremor STFT.
    """
    mu = float(S.mean())
    sd = float(S.std()) + eps
    return ((S - mu) / sd).astype(S.dtype, copy=False)


def crop_freq_bins(
    S: np.ndarray,
    n_sensors: int,
    n_freq_bins: int,
    keep_bins: int,
) -> np.ndarray:
    """Keep the lowest ``keep_bins`` frequency rows from each sensor block."""
    if S.shape[0] != n_sensors * n_freq_bins:
        raise ValueError(
            f"Expected {n_sensors * n_freq_bins} feature rows, got {S.shape[0]}."
        )
    keep_bins = min(keep_bins, n_freq_bins)
    parts = [
        S[s * n_freq_bins : s * n_freq_bins + keep_bins] for s in range(n_sensors)
    ]
    return np.concatenate(parts, axis=0)


def freq_bins_for_fmax(
    fs: float, n_freq_bins: int, f_max: float | None
) -> int:
    """How many of the lowest freq bins fall at or below ``f_max``.

    Frequencies are uniformly spaced from 0 to fs/2 over n_freq_bins.
    """
    if f_max is None:
        return n_freq_bins
    nyq = fs / 2.0
    idx = int(np.floor(f_max / nyq * (n_freq_bins - 1))) + 1
    return max(1, min(n_freq_bins, idx))


def spec_augment(
    S: np.ndarray,
    rng: np.random.Generator,
    n_freq_masks: int = 1,
    freq_mask_max: int = 8,
    n_time_masks: int = 1,
    time_mask_max: int = 8,
) -> np.ndarray:
    """SpecAugment-style freq/time masking. Training-only augmentation."""
    out = S.copy()
    f_total, t_total = out.shape
    for _ in range(n_freq_masks):
        f = int(rng.integers(0, freq_mask_max + 1))
        if f and f_total > f:
            f0 = int(rng.integers(0, f_total - f))
            out[f0 : f0 + f] = 0.0
    for _ in range(n_time_masks):
        t = int(rng.integers(0, time_mask_max + 1))
        if t and t_total > t:
            t0 = int(rng.integers(0, t_total - t))
            out[:, t0 : t0 + t] = 0.0
    return out


def time_pad(
    S: np.ndarray, target_T: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Zero-pad / centre-crop along the time axis to ``target_T`` frames.

    If ``rng`` is provided, the padding offset is randomised (training);
    otherwise the signal is centred (val / test).
    """
    F, T = S.shape
    if T >= target_T:
        start = (T - target_T) // 2
        return S[:, start : start + target_T]
    out = np.zeros((F, target_T), dtype=S.dtype)
    if rng is None:
        offset = (target_T - T) // 2
    else:
        offset = int(rng.integers(0, target_T - T + 1))
    out[:, offset : offset + T] = S
    return out
