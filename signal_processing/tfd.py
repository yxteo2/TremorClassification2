"""Time-frequency decompositions used by the pipeline.

Two estimators remain, both consumed through ``signal_processing.transforms``:

* ``apply_multitaper``     -- DPSS multitaper spectrogram (the reported model's
                              spectrum: nw 2.5, K 4, nperseg 256).
* ``apply_wavelet_packet`` -- wavelet-packet band energies (``final_model``'s
                              wavelet_packet arm).

STFT, CWT, SST, HHT/EMD, Welch-as-TFD and the separability helper were removed
on 2026-09-28 with the benchmark estimators that used them; they are in git
history (commit 57fd8a72).
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_multitaper", "apply_wavelet_packet"]


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


def _multitaper_channel(
    channel: np.ndarray, fs: float, nperseg: int, nfft: int,
    noverlap: int, n_tapers: int, nw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper magnitude spectrogram of one channel.

    Averages the power spectra from ``n_tapers`` orthogonal DPSS
    (Slepian) windows, then takes the square root. The taper average
    drives down spectral-estimate variance — valuable on short, noisy
    IMU clips where a single Hann window gives a jittery spectrogram.

    Returns ``(f_hz, mag)`` with ``mag`` shape ``(n_freq_bins, n_frames)``.
    """
    from scipy.signal.windows import dpss

    hop = nperseg - noverlap
    frames = _frame(channel.astype(np.float64, copy=False), nperseg, hop)
    tapers = dpss(nperseg, NW=nw, Kmax=n_tapers)        # (n_tapers, nperseg)
    psd_sum = None
    for w in tapers:
        spec = np.fft.rfft(frames * w, n=nfft, axis=1)  # (n_frames, n_freq)
        p = (np.abs(spec) ** 2)
        psd_sum = p if psd_sum is None else psd_sum + p
    psd = psd_sum / n_tapers
    mag = np.sqrt(psd).astype(np.float32).T            # (n_freq, n_frames)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float32)
    return f, mag


def apply_multitaper(
    x: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 128,
    nfft: int = 128,
    noverlap: int = 96,
    f_max: float | None = None,
    n_tapers: int = 4,
    nw: float = 2.5,
) -> np.ndarray:
    """Stack per-channel multitaper magnitude spectrograms.

    Output layout ``(channels * n_kept_freq_bins, n_time_bins)``. ``nw`` is the time-bandwidth product and
    ``n_tapers`` should be < ``2*nw`` for well-concentrated tapers.
    """
    parts: list[np.ndarray] = []
    for ch in range(x.shape[0]):
        f, mag = _multitaper_channel(
            x[ch], fs=fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap,
            n_tapers=n_tapers, nw=nw,
        )
        if f_max is not None:
            mag = mag[f <= f_max]
        parts.append(mag)
    return np.concatenate(parts, axis=0)


def apply_wavelet_packet(
    x: np.ndarray, fs: float = 100.0, level: int = 5,
    wavelet: str = "db4", f_max: float | None = None,
    log_energy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Wavelet packet decomposition: per-band time-domain energy.

    Builds a full binary tree of length ``level``, giving ``2**level``
    equal-width frequency sub-bands. For each band, returns the
    time-localised energy (squared coefficient magnitude). The output
    shape is ``(channels * n_bands, n_time_frames)`` where
    ``n_time_frames = T // 2**level`` (downsampled by the wavelet
    decimation cascade).

    Args:
        x: ``(channels, time)`` input signal.
        fs: sampling rate; band centres are ``(k + 0.5) * fs / 2**(level+1)``.
        level: decomposition depth; level 5 at fs=100 gives 32 bands
            of ~1.56 Hz width each.
        wavelet: any ``pywt`` wavelet name (``db4``, ``sym8``, ``coif5`` …).
        f_max: optional upper-frequency cap.
        log_energy: if True, return ``log1p(|coeff|**2)``; useful for
            heavy-tailed tremor magnitudes.

    Returns:
        ``(band_centres_Hz, decomposition)`` — frequency vector and
        ``(channels * n_kept_bands, n_time_frames)`` band energy.
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "Wavelet packets require PyWavelets. Install with: pip install pywavelets"
        ) from e

    n_ch, T = x.shape
    n_bands = 2 ** level
    band_width = fs / 2.0 / n_bands
    band_centres = (np.arange(n_bands) + 0.5) * band_width

    keep = np.arange(n_bands)
    if f_max is not None:
        keep = keep[band_centres[keep] <= f_max]
    band_centres = band_centres[keep]

    per_ch = []
    for c in range(n_ch):
        wp = pywt.WaveletPacket(
            data=x[c].astype(np.float64), wavelet=wavelet,
            mode="symmetric", maxlevel=level,
        )
        # Frequency-sorted nodes (grey-code permutation) so band index k
        # corresponds to the k-th frequency band of width fs/2/n_bands.
        nodes = [n.path for n in wp.get_level(level, order="freq")]
        bands = []
        for k in keep:
            coeffs = np.asarray(wp[nodes[int(k)]].data, dtype=np.float32)
            energy = coeffs ** 2
            if log_energy:
                energy = np.log1p(energy)
            bands.append(energy)
        # All band rows have the same length after decomposition
        min_len = min(b.shape[0] for b in bands)
        per_ch.append(np.stack([b[:min_len] for b in bands]))

    # Stack across channels
    min_t = min(p.shape[1] for p in per_ch)
    out = np.concatenate([p[:, :min_t] for p in per_ch], axis=0).astype(np.float32)
    return band_centres.astype(np.float32), out
