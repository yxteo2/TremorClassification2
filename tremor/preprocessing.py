"""Tremor STFT preprocessing.

Data layout convention (matches the original MATLAB project):

    Time-domain recording:
        shape  (channels=3, time=T)
        rows are sensor channels in order (L=lower arm, H=hand, U=upper arm)
        columns are samples at fs = 60 Hz (after the downsize factor of
        0.6 that the MATLAB pipeline applied earlier).

    STFT representation fed to the BiLSTM:
        shape  (channels * n_freq_bins, n_time_bins)
        rows are (L_f0, L_f1, ..., L_fK, H_f0, ..., H_fK, U_f0, ..., U_fK)
        columns are STFT frame indices, i.e. TIME.
        This is the same orientation as MATLAB ``amptostft.m`` — whose
        final ``compiled_data'`` transpose puts (channel x freq) on
        rows and (time) on columns.

Defaults reproduce MATLAB ``stft(x, fs)`` defaults:
    * Window         : Hann (periodic), length 128
    * FFT length     : 256   (so 129 one-sided frequency bins)
    * Overlap        : 96    (75 % overlap; hop = 32 samples)
    * One-sided      : yes   (0 Hz to fs/2)
    * Amplitude      : magnitude multiplied by 2 to restore the energy
                       discarded with the negative-frequency half
                       (matching ``amptostft.m`` exactly).

Tremor band reference:
    Parkinsonian rest : 3 - 6 Hz
    Essential         : 4 - 12 Hz
    Physiological     : 8 - 12 Hz
    Pass ``f_max`` (e.g. 15.0) to crop high-frequency bins that carry
    no tremor information and inflate model input dimensionality.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


# ---------------------------------------------------------------------
# Time-domain operations (shape: channels x time)
# ---------------------------------------------------------------------

def bandpass(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (3.0, 15.0),
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the TIME axis.

    Mirrors ``Preprocessing/bandpass_pad.m``: each channel is pre-padded
    by ~10 % of its length on either side using its endpoint values
    before filtering, then the padding is trimmed.

    The default band of 3 - 15 Hz covers Parkinsonian (3 - 6 Hz),
    essential (4 - 12 Hz), and physiological (8 - 12 Hz) tremors. The
    upper cutoff is clamped strictly below Nyquist to keep
    ``scipy.signal.butter`` happy when ``fs`` is low.

    Parameters
    ----------
    x    : ndarray of shape (channels, time)
    fs   : sampling frequency in Hz
    band : (low, high) cutoffs in Hz
    """
    nyq = 0.5 * fs
    lo, hi = max(band[0], 1e-3), min(band[1], nyq * 0.999)
    if lo >= hi:
        raise ValueError(f"Invalid band {band} for fs={fs} (Nyquist={nyq}).")
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        ch = x[c]
        n = max(1, len(ch) // 10)
        padded = np.concatenate(
            [np.full(n, ch[0], dtype=ch.dtype), ch, np.full(n, ch[-1], dtype=ch.dtype)]
        )
        y = signal.sosfiltfilt(sos, padded)
        out[c] = y[n:-n] if n > 0 else y
    return out.astype(x.dtype, copy=False)


def random_pad(
    x: np.ndarray,
    target_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random-offset zero-padding along the TIME axis (training augmentation)."""
    channels, t = x.shape
    if t >= target_length:
        return x[:, :target_length]
    out = np.zeros((channels, target_length), dtype=x.dtype)
    offset = int(rng.integers(0, target_length - t + 1))
    out[:, offset:offset + t] = x
    return out


def center_pad(x: np.ndarray, target_length: int) -> np.ndarray:
    """Deterministic centered zero-padding (used for val / test)."""
    channels, t = x.shape
    if t >= target_length:
        return x[:, :target_length]
    out = np.zeros((channels, target_length), dtype=x.dtype)
    offset = (target_length - t) // 2
    out[:, offset:offset + t] = x
    return out


# ---------------------------------------------------------------------
# STFT (output orientation: (channels * freq) x time)
# ---------------------------------------------------------------------

def _hann_periodic(n: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Periodic Hann window — matches MATLAB ``hann(n, 'periodic')``."""
    k = np.arange(n, dtype=np.float64)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / n))
    return win.astype(dtype)


def _frame(x: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    """Slice a 1-D signal into overlapping frames.

    Returns shape (n_frames, nperseg). If the signal is shorter than
    one frame the result has shape (1, nperseg) with trailing zeros so
    the downstream STFT shape is well-defined.
    """
    n = len(x)
    if n < nperseg:
        out = np.zeros((1, nperseg), dtype=x.dtype)
        out[0, :n] = x
        return out
    n_frames = 1 + (n - nperseg) // hop
    idx = np.arange(nperseg)[None, :] + (np.arange(n_frames) * hop)[:, None]
    return x[idx]


def stft_magnitude(
    channel: np.ndarray,
    fs: float = 60.0,
    nperseg: int = 128,
    nfft: int = 256,
    noverlap: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel one-sided magnitude STFT.

    Returns
    -------
    f   : ndarray of shape (n_freq_bins,) — frequency in Hz
    S   : ndarray of shape (n_freq_bins, n_time_bins) — magnitude
          scaled by 2 to compensate for the discarded negative-frequency
          half, matching ``amptostft.m``.
    """
    hop = nperseg - noverlap
    win = _hann_periodic(nperseg, dtype=np.float32)
    frames = _frame(channel.astype(np.float32, copy=False), nperseg, hop) * win
    spec = np.fft.rfft(frames, n=nfft, axis=1)        # (n_frames, n_freq)
    mag = (2.0 * np.abs(spec)).astype(np.float32).T   # (n_freq, n_frames)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float32)
    return f, mag


def apply_stft(
    x: np.ndarray,
    fs: float = 60.0,
    nperseg: int = 128,
    nfft: int = 256,
    noverlap: int = 96,
    f_max: float | None = None,
) -> np.ndarray:
    """Stack per-channel STFT magnitudes along the channel-frequency axis.

    Parameters
    ----------
    x       : ndarray of shape (channels, time)
    f_max   : optional frequency upper bound in Hz; bins above this are
              dropped. Useful for restricting to the tremor band
              (~ <= 15 Hz) to reduce input dimensionality.

    Returns
    -------
    ndarray of shape (channels * n_kept_freq_bins, n_time_bins).
    Rows are ordered (ch0_freq0..ch0_freqK, ch1_freq0..ch1_freqK, ...).
    Columns are STFT frame indices (time).
    """
    parts: list[np.ndarray] = []
    for ch in range(x.shape[0]):
        f, mag = stft_magnitude(
            x[ch], fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap
        )
        if f_max is not None:
            mag = mag[f <= f_max]
        parts.append(mag)
    return np.concatenate(parts, axis=0)
