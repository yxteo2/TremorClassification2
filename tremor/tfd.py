"""Time-frequency decompositions (TFDs) for tremor classification.

Tremor is essentially a narrow-band oscillation in the limb's angular
velocity signal. The classification model needs a representation that
1) localises that oscillation in frequency (PD ~ 4-6 Hz, ET ~ 6-10 Hz,
2) is robust to short, possibly non-stationary recordings, and
3) keeps feature dimensionality manageable.

Three options provided here:

* ``apply_stft``  : re-exported from ``preprocessing`` for symmetry.
                    Linear-frequency spectrogram with a fixed window.
                    Cheap; works well when the recording is long
                    relative to the slowest tremor period.

* ``apply_cwt``   : Continuous Wavelet Transform with complex Morlet
                    wavelets. Wavelet duration scales as 1/f, so low
                    frequencies get long windows (high freq resolution)
                    and high frequencies get short windows (high time
                    resolution) -- favours short, non-stationary clips.

* ``apply_welch`` : Welch periodogram (time axis collapsed). Returns a
                    1-D PSD per channel; the smoothest estimate but
                    incompatible with sequence models.

Use :func:`compare_tfd_separability` to score how well each method
separates classes on a small labelled set.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from tremor.preprocessing import apply_stft  # noqa: F401 — re-exported


__all__ = [
    "apply_stft", "apply_cwt", "apply_welch",
    "apply_hht", "apply_emd_imfs",
    "marginal_hilbert_spectrum", "dominant_imf_frequencies",
    "apply_wavelet_packet",
    "compare_tfd_separability",
]


def _morlet_wavelet(fs: float, freq: float, w0: float = 6.0) -> np.ndarray:
    """Complex Morlet wavelet centred at ``freq`` Hz.

    Time-domain Gaussian envelope std = ``w0 / (2*pi*freq)`` seconds; a
    higher ``w0`` gives sharper frequency resolution at the cost of
    longer wavelets. ``w0 = 6`` is the standard choice.
    """
    sigma_t = w0 / (2 * np.pi * freq)
    half_width = max(int(4 * sigma_t * fs), 4)
    t = (np.arange(2 * half_width + 1) - half_width) / fs
    psi = np.exp(1j * 2 * np.pi * freq * t) * np.exp(-(t ** 2) / (2 * sigma_t ** 2))
    psi /= np.abs(psi).sum() + 1e-12
    return psi


def apply_cwt(
    x: np.ndarray,
    fs: float = 100.0,
    freqs: np.ndarray | None = None,
    w0: float = 6.0,
    decim: int = 1,
    f_max: float | None = None,
) -> np.ndarray:
    """Magnitude of the complex Morlet CWT, stacked across channels.

    Args:
        x: ``(channels, time)`` input signal.
        fs: sampling rate in Hz.
        freqs: 1-D array of analysis frequencies in Hz. Default = linear
            grid 3 Hz - 15 Hz spaced 0.5 Hz (covers PD / ET / physiological
            tremor bands).
        w0: Morlet central angular frequency parameter.
        decim: temporal decimation factor applied after convolution.
        f_max: optional cap on the analysis frequencies.

    Returns:
        ``(channels * n_kept_freqs, ceil(time/decim))`` magnitude in
        sensor-major, then frequency-minor order to match
        ``apply_stft`` 's layout.
    """
    if freqs is None:
        freqs = np.arange(3.0, 15.01, 0.5)
    freqs = np.asarray(freqs, dtype=np.float64)
    if f_max is not None:
        freqs = freqs[freqs <= f_max]

    n_ch, T = x.shape
    n_f = len(freqs)
    out_T = (T + decim - 1) // decim
    out = np.empty((n_ch * n_f, out_T), dtype=np.float32)

    for i, f in enumerate(freqs):
        psi = _morlet_wavelet(fs, float(f), w0=w0)
        for c in range(n_ch):
            conv = np.convolve(x[c], psi, mode="same")
            mag = np.abs(conv).astype(np.float32)
            out[c * n_f + i] = mag[::decim][:out_T]
    return out


def apply_welch(
    x: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 256,
    noverlap: int = 192,
    f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel, stacked.

    Returns a tuple ``(freqs, psd)`` where ``freqs`` is shape
    ``(n_kept_freqs,)`` and ``psd`` is shape
    ``(channels * n_kept_freqs,)``. The time axis is collapsed, so this
    is a static feature vector — fine for FC / Conv classifiers, not
    for sequence models.
    """
    n_ch = x.shape[0]
    parts = []
    last_freqs = None
    for c in range(n_ch):
        f, psd = welch(x[c], fs=fs, nperseg=min(nperseg, x.shape[1]),
                       noverlap=min(noverlap, x.shape[1] // 2))
        if f_max is not None:
            keep = f <= f_max
            f, psd = f[keep], psd[keep]
        parts.append(psd.astype(np.float32))
        last_freqs = f.astype(np.float32)
    return last_freqs, np.concatenate(parts, axis=0)


def _emd_imfs(signal: np.ndarray, max_imfs: int = 8) -> np.ndarray:
    """Empirical Mode Decomposition of a 1-D signal.

    Returns up to ``max_imfs`` IMFs as rows of shape ``(n_imfs, time)``.
    Pads with zero rows if fewer IMFs are produced so the output shape
    is deterministic.
    """
    try:
        from PyEMD import EMD
    except ImportError as e:
        raise ImportError(
            "EMD / HHT require the 'EMD-signal' package. "
            "Install with: pip install EMD-signal"
        ) from e

    emd = EMD()
    emd.MAX_ITERATION = 500
    imfs = emd(signal.astype(np.float64), max_imf=max_imfs)
    if imfs.shape[0] >= max_imfs:
        imfs = imfs[:max_imfs]
    else:
        pad = np.zeros((max_imfs - imfs.shape[0], signal.shape[0]),
                       dtype=imfs.dtype)
        imfs = np.concatenate([imfs, pad], axis=0)
    return imfs.astype(np.float32)


def apply_emd_imfs(
    x: np.ndarray, max_imfs: int = 8,
) -> np.ndarray:
    """Per-channel EMD; returns IMFs stacked as extra channels.

    Args:
        x: ``(channels, time)`` input.
        max_imfs: keep this many IMFs per channel (zero-padded if fewer).

    Returns:
        ``(channels * max_imfs, time)`` IMF time-series, ready to feed
        into a downstream STFT/CWT or used directly as channels.
    """
    n_ch, T = x.shape
    out = np.empty((n_ch * max_imfs, T), dtype=np.float32)
    for c in range(n_ch):
        imfs = _emd_imfs(x[c], max_imfs=max_imfs)
        out[c * max_imfs : (c + 1) * max_imfs] = imfs
    return out


def apply_hht(
    x: np.ndarray,
    fs: float = 100.0,
    freqs: np.ndarray | None = None,
    max_imfs: int = 8,
    f_max: float | None = None,
    decim: int = 1,
) -> np.ndarray:
    """Hilbert-Huang spectrogram.

    Pipeline:
        1. EMD each channel into IMFs.
        2. Analytic signal per IMF via the Hilbert transform.
        3. Instantaneous frequency = ``(1/2pi) * d/dt(unwrap(angle))``.
        4. Instantaneous amplitude = ``|analytic|``.
        5. For each time step, accumulate amplitude into the frequency
           bin closest to the instantaneous frequency. Repeat for all
           IMFs of all channels and write into a per-channel grid.

    Args:
        x: ``(channels, time)`` input.
        fs: sampling rate in Hz.
        freqs: 1-D array of analysis-frequency bin centres in Hz.
            Default: linear grid 3 - 15 Hz spaced 0.5 Hz.
        max_imfs: how many IMFs per channel to retain.
        f_max: optional cap on the analysis frequencies.
        decim: temporal decimation factor applied at the end.

    Returns:
        ``(channels * n_kept_freqs, ceil(time/decim))`` Hilbert magnitude.
    """
    from scipy.signal import hilbert

    if freqs is None:
        freqs = np.arange(3.0, 15.01, 0.5)
    freqs = np.asarray(freqs, dtype=np.float64)
    if f_max is not None:
        freqs = freqs[freqs <= f_max]

    n_ch, T = x.shape
    n_f = len(freqs)
    out_T = (T + decim - 1) // decim
    out = np.zeros((n_ch * n_f, out_T), dtype=np.float32)

    freq_edges = np.concatenate([
        [-np.inf],
        (freqs[:-1] + freqs[1:]) / 2.0,
        [np.inf],
    ])

    for c in range(n_ch):
        imfs = _emd_imfs(x[c], max_imfs=max_imfs)
        grid = np.zeros((n_f, T), dtype=np.float32)
        for imf in imfs:
            if np.allclose(imf, 0):
                continue
            analytic = hilbert(imf.astype(np.float64))
            inst_amp = np.abs(analytic).astype(np.float32)
            inst_phase = np.unwrap(np.angle(analytic))
            inst_freq = np.empty_like(inst_phase, dtype=np.float32)
            inst_freq[1:] = np.diff(inst_phase) * fs / (2 * np.pi)
            inst_freq[0] = inst_freq[1]
            bin_idx = np.digitize(inst_freq, freq_edges) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_f)
            np.add.at(grid, (bin_idx[valid], np.where(valid)[0]), inst_amp[valid])
        if decim > 1:
            grid = grid[:, ::decim][:, :out_T]
        out[c * n_f : (c + 1) * n_f, : grid.shape[1]] = grid
    return out


def marginal_hilbert_spectrum(
    x: np.ndarray, fs: float = 100.0,
    freqs: np.ndarray | None = None,
    max_imfs: int = 8, f_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal Hilbert spectrum — proper way to read HHT peak frequency.

    The marginal Hilbert spectrum integrates ``amplitude(t, f)`` over time
    to produce a 1-D ``energy(f)`` curve. This is the correct frequency-
    domain summary of an HHT decomposition; using ``argmax`` on the
    binned (freq, time) magnitude as we do in :func:`apply_hht` collapses
    to the lowest bin because of how energy spreads across IMFs.

    Args:
        x: ``(channels, time)`` input signal.
        fs, freqs, max_imfs, f_max: as in :func:`apply_hht`.

    Returns:
        ``(freqs, energy)`` where ``energy[c, f]`` is the time-integrated
        amplitude for channel ``c`` at frequency bin ``f``.
    """
    from scipy.signal import hilbert

    if freqs is None:
        freqs = np.arange(0.5, 15.01, 0.5)
    freqs = np.asarray(freqs, dtype=np.float64)
    if f_max is not None:
        freqs = freqs[freqs <= f_max]

    n_ch, _ = x.shape
    n_f = len(freqs)
    edges = np.concatenate([[-np.inf],
                            (freqs[:-1] + freqs[1:]) / 2.0,
                            [np.inf]])

    energy = np.zeros((n_ch, n_f), dtype=np.float32)
    for c in range(n_ch):
        imfs = _emd_imfs(x[c], max_imfs=max_imfs)
        for imf in imfs:
            if np.allclose(imf, 0):
                continue
            a = hilbert(imf.astype(np.float64))
            amp = np.abs(a).astype(np.float32)
            phase = np.unwrap(np.angle(a))
            f_inst = np.empty_like(phase, dtype=np.float32)
            f_inst[1:] = np.diff(phase) * fs / (2 * np.pi)
            f_inst[0] = f_inst[1]
            bin_idx = np.digitize(f_inst, edges) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_f)
            np.add.at(energy[c], bin_idx[valid], amp[valid])
    return freqs.astype(np.float32), energy


def dominant_imf_frequencies(
    x: np.ndarray, fs: float = 100.0, max_imfs: int = 8,
    f_max: float | None = 15.0,
) -> np.ndarray:
    """Per-IMF amplitude-weighted mean instantaneous frequency.

    Standard HHT readout for picking the dominant tremor frequency:
    decompose the signal into IMFs, then for each IMF report
    ``sum(amp * f_inst) / sum(amp)``. The IMF with the largest total
    amplitude in the tremor band is the dominant one.

    Args:
        x: ``(channels, time)`` input signal.
        fs: sampling rate in Hz.
        max_imfs: number of IMFs per channel to retain.
        f_max: discard IMFs whose dominant frequency exceeds this
            (filters out high-frequency noise IMFs).

    Returns:
        ``(channels, max_imfs)`` array of amplitude-weighted mean
        instantaneous frequency per IMF (NaN for empty / out-of-band IMFs).
    """
    from scipy.signal import hilbert

    n_ch, T = x.shape
    out = np.full((n_ch, max_imfs), np.nan, dtype=np.float32)
    for c in range(n_ch):
        imfs = _emd_imfs(x[c], max_imfs=max_imfs)
        for i, imf in enumerate(imfs):
            if np.allclose(imf, 0):
                continue
            a = hilbert(imf.astype(np.float64))
            amp = np.abs(a)
            phase = np.unwrap(np.angle(a))
            f_inst = np.empty_like(phase)
            f_inst[1:] = np.diff(phase) * fs / (2 * np.pi)
            f_inst[0] = f_inst[1]
            wsum = float((amp * np.abs(f_inst)).sum())
            asum = float(amp.sum() + 1e-12)
            f_mean = wsum / asum
            if f_max is None or 0.0 <= f_mean <= f_max:
                out[c, i] = f_mean
    return out


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


def _flatten_feature(method_output: np.ndarray) -> np.ndarray:
    """Reduce a 2-D (F, T) spectrogram or 1-D PSD to a 1-D feature vector.

    Spectrograms are averaged over time so all three TFDs end up with
    the same shape for an apples-to-apples comparison.
    """
    arr = np.asarray(method_output)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=-1)


def _fisher_ratio(features: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-pair Fisher discriminant ratio (higher = more separable).

    Computed per feature dimension then averaged. With C classes there
    are C*(C-1)/2 ordered pairs; we use the symmetric form

        F_ij = (mu_i - mu_j)**2 / (var_i + var_j + eps)

    averaged across all pairs and feature dims.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    means = {c: features[labels == c].mean(axis=0) for c in classes}
    vars_ = {c: features[labels == c].var(axis=0)  for c in classes}
    pair_scores = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            num = (means[ci] - means[cj]) ** 2
            den = vars_[ci] + vars_[cj] + 1e-9
            pair_scores.append((num / den).mean())
    return float(np.mean(pair_scores))


def _lda_loo_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
    """Leave-one-out cross-validated LDA accuracy.

    A coarse but honest separability score: trains a Linear
    Discriminant Analysis on N-1 samples and tests on the held-out one,
    repeated for each sample.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n = len(features)
    correct = 0
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(features[train_mask], labels[train_mask])
            pred = lda.predict(features[i : i + 1])
            if pred[0] == labels[i]:
                correct += 1
        except Exception:
            pass
    return correct / n


def compare_tfd_separability(
    signals: list[np.ndarray],
    labels: list[int] | np.ndarray,
    fs: float = 100.0,
    f_max: float = 15.0,
) -> dict[str, dict[str, float]]:
    """Score each TFD by how well it separates the given labelled signals.

    Returns one entry per method (``stft``, ``cwt``, ``welch``) with
    keys ``fisher`` (higher = better) and ``lda_loo_acc`` (in [0, 1]).
    """
    feats = {"stft": [], "cwt": [], "welch": []}
    for x in signals:
        feats["stft"].append(_flatten_feature(
            apply_stft(x, fs=fs, nperseg=128, nfft=256, noverlap=96, f_max=f_max)
        ))
        feats["cwt"].append(_flatten_feature(
            apply_cwt(x, fs=fs, f_max=f_max)
        ))
        _, w = apply_welch(x, fs=fs, f_max=f_max)
        feats["welch"].append(_flatten_feature(w))

    labels = np.asarray(labels)
    return {
        name: {
            "fisher": _fisher_ratio(np.stack(arr), labels),
            "lda_loo_acc": _lda_loo_accuracy(np.stack(arr), labels),
            "feature_dim": int(arr[0].shape[0]),
        }
        for name, arr in feats.items()
    }
