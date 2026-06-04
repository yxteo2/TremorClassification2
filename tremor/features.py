"""Scalar per-recording signal-processing features for tremor.

The TFD spectrograms binned across time and frequency smooth out the
exact discriminator between PD and ET tremor: frequency *regularity*.
PD's pill-rolling tremor is metronomic (low IF variance, narrow IF
range); ET's postural tremor drifts (high IF variance, broader IF
range). This module extracts those scalar fingerprints per recording
so a downstream model can use them as an explicit auxiliary input.

Features (per channel):
    * IF mean  — where the dominant IMF sits in Hz
    * IF std   — frequency drift (PD low, ET high)
    * Amplitude regularity = 1 - std(|a|) / mean(|a|)  (PD high, ET low)
    * IF slope vs time — linear-regression drift over the recording

For ``n_imfs=2`` IMFs and 3 sensor axes, that's 24 features. Plus per-
channel marginal-Hilbert peak frequency and band energy = +6 → 30 in
total. Tunable via :func:`extract_imf_features`'s ``n_imfs``.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from tremor.tfd import _emd_imfs


def _instantaneous_freq_amp(imf: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    analytic = hilbert(imf.astype(np.float64))
    amp = np.abs(analytic).astype(np.float32)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.empty_like(phase, dtype=np.float32)
    inst_f[1:] = np.diff(phase) * fs / (2 * np.pi)
    inst_f[0] = inst_f[1]
    return inst_f, amp


def _imf_stats(imf: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Return [mean_IF, std_IF, amp_regularity, IF_drift_slope] for one IMF.

    All scalars are returned even on degenerate input (zero IMF) — in
    that case mean_IF=0 and the others=0, which BatchNorm later
    centres harmlessly.
    """
    if np.allclose(imf, 0):
        return np.zeros(4, dtype=np.float32)
    inst_f, amp = _instantaneous_freq_amp(imf, fs)
    # weight by amplitude so noisy zero-amplitude phase jumps don't dominate
    w = amp + 1e-12
    mean_if = float(np.sum(inst_f * w) / np.sum(w))
    var_if = float(np.sum((inst_f - mean_if) ** 2 * w) / np.sum(w))
    std_if = float(np.sqrt(var_if))
    # amplitude regularity: 1.0 = perfectly rhythmic, 0.0 = bursty.
    a_mean = float(np.mean(amp))
    a_std = float(np.std(amp))
    amp_reg = 1.0 - a_std / (a_mean + 1e-12)
    amp_reg = float(np.clip(amp_reg, 0.0, 1.0))
    # IF drift slope: linear regression of IF vs time in Hz/s.
    t = np.arange(len(inst_f), dtype=np.float32) / fs
    t_centered = t - t.mean()
    denom = float(np.sum(t_centered ** 2) + 1e-12)
    slope = float(np.sum(t_centered * (inst_f - mean_if)) / denom)
    return np.array([mean_if, std_if, amp_reg, slope], dtype=np.float32)


def _channel_global_stats(channel: np.ndarray, fs: float,
                          band: tuple[float, float]) -> np.ndarray:
    """Per-channel scalars independent of IMF decomposition.

    Returns [marginal_Hilbert_peak_freq, band_energy].
    """
    analytic = hilbert(channel.astype(np.float64))
    amp = np.abs(analytic).astype(np.float32)
    phase = np.unwrap(np.angle(analytic))
    inst_f = np.empty_like(phase, dtype=np.float32)
    inst_f[1:] = np.diff(phase) * fs / (2 * np.pi)
    inst_f[0] = inst_f[1]
    # build a 0.5 Hz-spaced marginal Hilbert spectrum over the tremor band
    f_lo, f_hi = band
    edges = np.arange(f_lo, f_hi + 0.5, 0.5, dtype=np.float32)
    if len(edges) < 2:
        return np.zeros(2, dtype=np.float32)
    bin_idx = np.digitize(inst_f, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < len(edges) - 1)
    hist = np.zeros(len(edges) - 1, dtype=np.float32)
    np.add.at(hist, bin_idx[valid], amp[valid])
    if hist.sum() <= 0:
        return np.zeros(2, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    peak_f = float(centers[int(np.argmax(hist))])
    band_energy = float(hist.sum())
    return np.array([peak_f, band_energy], dtype=np.float32)


def extract_imf_features(
    x: np.ndarray, fs: float = 100.0,
    n_imfs: int = 2, max_imfs: int = 6,
    band: tuple[float, float] = (3.0, 15.0),
) -> np.ndarray:
    """Per-channel IMF + global scalar features for one recording.

    Args:
        x: ``(n_channels, time)`` signal — typically hand-only angular
            velocity, 3 channels.
        fs: sampling rate.
        n_imfs: how many top IMFs (by amplitude) to summarise per channel.
        max_imfs: ceiling on EMD sifting (cost cap).
        band: tremor frequency band used by the global stats (Hz).

    Returns:
        Flat ``np.ndarray`` of length ``n_channels * (n_imfs * 4 + 2)``.
        For default 3-channel hand input + n_imfs=2: length 30.
    """
    n_ch, _T = x.shape
    out: list[np.ndarray] = []
    for c in range(n_ch):
        imfs = _emd_imfs(x[c], max_imfs=max_imfs)
        # rank by mean amplitude so we keep the dominant oscillations
        amps = np.array([float(np.mean(np.abs(imf))) for imf in imfs])
        order = np.argsort(-amps)[:n_imfs]
        for k in range(n_imfs):
            if k < len(order):
                out.append(_imf_stats(imfs[order[k]], fs, band))
            else:
                out.append(np.zeros(4, dtype=np.float32))
        out.append(_channel_global_stats(x[c], fs, band))
    return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def feature_dim(n_channels: int, n_imfs: int = 2) -> int:
    return n_channels * (n_imfs * 4 + 2)
