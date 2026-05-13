"""STFT and random-pad augmentation.

The original MATLAB pipeline applies ``randpadsampling`` (random padding
+ optional Hanning window) and then ``amptostft``. Here we reproduce
the essential operations:

* ``random_pad`` — pad a 2D (channels, time) array up to a target
                   length by inserting it at a random offset.
* ``apply_stft`` — per-channel scipy STFT, magnitude of the non-negative
                   frequency bins, stacked along the channel axis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def random_pad(
    x: np.ndarray,
    target_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    channels, t = x.shape
    if t >= target_length:
        return x[:, :target_length]
    out = np.zeros((channels, target_length), dtype=x.dtype)
    offset = int(rng.integers(0, target_length - t + 1))
    out[:, offset:offset + t] = x
    return out


def center_pad(x: np.ndarray, target_length: int) -> np.ndarray:
    channels, t = x.shape
    if t >= target_length:
        return x[:, :target_length]
    out = np.zeros((channels, target_length), dtype=x.dtype)
    offset = (target_length - t) // 2
    out[:, offset:offset + t] = x
    return out


def apply_stft(x: np.ndarray, fs: float = 60.0, nperseg: int = 64) -> np.ndarray:
    """Compute magnitude STFT of every channel and stack along channel axis.

    Output shape: (channels * n_freq_bins, n_time_bins).
    """
    parts = []
    for ch in range(x.shape[0]):
        _, _, zxx = signal.stft(
            x[ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            return_onesided=True,
        )
        parts.append(2.0 * np.abs(zxx).astype(np.float32))
    return np.concatenate(parts, axis=0)
