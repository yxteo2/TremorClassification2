"""Unified registry of signal-processing / time-frequency methods.

Every method maps a ``(channels, time)`` recording to a **power spectrum over
frequency**, so that the same frequency descriptors (max/mean/median frequency,
...) can be computed from all of them and compared on equal footing.

Two output kinds are supported and both are reduced to a 1-D spectrum:
  * genuine TF matrices (STFT, CWT, HHT, SST, S-transform) -> average over time
  * direct spectra (Welch, multitaper, AR) -> used as-is
  * decompositions (wavelet packet, VMD, EMD) -> band/mode energies, mapped to
    their centre frequencies

Channels are combined **without rectification**: each channel's spectrum is
computed separately and averaged. Taking `sqrt(sum(ch^2))` first would square a
6 Hz oscillation into 12 Hz energy -- a real artifact this repo has already been
bitten by (`reports/signal_processing_summary.md`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, stft

from tremor.tfd import (
    apply_cwt, apply_hht, apply_multitaper, apply_sst, apply_wavelet_packet,
)

FS = 100.0
F_MIN, F_MAX = 3.0, 15.0


def _band(f, P, f_min=F_MIN, f_max=F_MAX):
    k = (f >= f_min) & (f <= f_max)
    return f[k], P[k]


# --------------------------------------------------------------------------- #
# Each method returns (freqs, power_spectrum) averaged over channels and time.
# --------------------------------------------------------------------------- #
def m_welch(x, fs=FS, nperseg=256, **kw):
    f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return _band(f, P.mean(0))


def m_stft(x, fs=FS, nperseg=256, noverlap=192, **kw):
    n = min(nperseg, x.shape[-1])
    f, _, Z = stft(x, fs=fs, nperseg=n, noverlap=min(noverlap, n - 1),
                   nfft=n, axis=-1, boundary=None, padded=False)
    return _band(f, (np.abs(Z) ** 2).mean(axis=(0, 2)))


def m_stft512(x, fs=FS, **kw):
    return m_stft(x, fs=fs, nperseg=512, noverlap=384)


def m_multitaper(x, fs=FS, nperseg=256, **kw):
    n = min(nperseg, x.shape[-1])
    S = apply_multitaper(x, fs=fs, nperseg=n, nfft=n, noverlap=n * 3 // 4,
                         f_max=F_MAX)
    n_ch = np.atleast_2d(x).shape[0]
    n_freq = np.asarray(S).shape[0] // n_ch
    P = _per_freq_mean(S, n_freq, n_ch, square=True)
    # apply_multitaper already cropped to f_max, so the grid runs 0..F_MAX
    return _band(np.linspace(0.0, F_MAX, n_freq), P)


def m_cwt(x, fs=FS, w0=6.0, step=0.25, **kw):
    freqs = np.arange(F_MIN, F_MAX + 1e-9, step)
    S = apply_cwt(x, fs=fs, freqs=freqs, w0=w0, decim=4, f_max=None)
    return freqs, _per_freq_mean(S, len(freqs), np.atleast_2d(x).shape[0], square=True)


def m_hht(x, fs=FS, max_imfs=8, step=0.25, emd_method="emd", **kw):
    freqs = np.arange(F_MIN, F_MAX + 1e-9, step)
    S = apply_hht(x, fs=fs, freqs=freqs, max_imfs=max_imfs, decim=4,
                  emd_method=emd_method)
    return freqs, _per_freq_mean(S, len(freqs), np.atleast_2d(x).shape[0], square=True)


def m_hht_imf2plus(x, fs=FS, max_imfs=8, step=0.25, **kw):
    """HHT excluding IMF1.

    Plain HHT is noise-dominated: EMD puts broadband noise into the first IMF,
    which then dominates the marginal spectrum (verified on synthetic data --
    a clean 6 Hz tone is recovered exactly, but at noise sd=0.3 the peak jumps
    to the top of the band). Dropping IMF1 is the standard remedy and makes the
    comparison against the other methods fair.
    """
    from tremor.tfd import _emd_imfs
    X2 = np.atleast_2d(x)
    freqs = np.arange(F_MIN, F_MAX + 1e-9, step)
    acc = np.zeros(len(freqs))
    for ch in X2:
        imfs = np.asarray(_emd_imfs(ch.astype(float), max_imfs=max_imfs))
        if imfs.ndim == 1 or imfs.shape[0] < 2:
            continue
        resid = imfs[1:].sum(0)              # drop IMF1 (noise), keep the rest
        f, P = welch(resid, fs=fs, nperseg=min(256, len(resid)))
        acc += np.interp(freqs, f, P)
    return freqs, acc / max(X2.shape[0], 1)


def m_sst(x, fs=FS, nperseg=256, **kw):
    n = min(nperseg, x.shape[-1])
    S = apply_sst(x, fs=fs, nperseg=n, nfft=n, noverlap=n * 3 // 4, f_max=F_MAX)
    n_ch = np.atleast_2d(x).shape[0]
    n_freq = np.asarray(S).shape[0] // n_ch
    P = _per_freq_mean(S, n_freq, n_ch, square=True)
    return _band(np.linspace(0.0, F_MAX, n_freq), P)


def m_wavelet_packet(x, fs=FS, level=5, wavelet="db4", **kw):
    out = apply_wavelet_packet(x, fs=fs, level=level, wavelet=wavelet,
                               f_max=None, log_energy=False)
    # returns (band_centres, S) -- use the transform's OWN centres rather than
    # assuming a uniform grid, since the packet ordering is not simply linear
    centres, S = (out if isinstance(out, tuple) else (None, out))
    S = np.asarray(S)
    n_ch = np.atleast_2d(x).shape[0]
    nb = S.shape[0] // n_ch
    P = _per_freq_mean(S, nb, n_ch)
    centres = (np.asarray(centres)[:nb] if centres is not None
               else (np.arange(nb) + 0.5) * (fs / 2) / nb)
    o = np.argsort(centres)
    return _band(np.asarray(centres)[o], P[o])


def m_stransform(x, fs=FS, n_freq=128, **kw):
    """Stockwell S-transform spectrum, computed directly.

    ``pdetn.extra_transforms.stransform_recording`` returns a flattened feature
    vector, not a (freq, time) matrix, so it cannot be reduced to a spectrum;
    computed here instead via the standard frequency-domain formulation.
    """
    X2 = np.atleast_2d(x)
    T = X2.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    keep = np.where((freqs >= F_MIN) & (freqs <= F_MAX))[0]
    if len(keep) > n_freq:                       # thin to keep the cost sane
        keep = keep[np.linspace(0, len(keep) - 1, n_freq).astype(int)]
    acc = np.zeros(len(keep))
    fft_freqs = np.fft.fftfreq(T, d=1.0 / fs)
    for ch in X2:
        F = np.fft.fft(ch - ch.mean())
        for i, k in enumerate(keep):
            f0 = freqs[k]
            if f0 <= 0:
                continue
            # Gaussian window in the frequency domain, width 1/f0
            g = np.exp(-2.0 * (np.pi ** 2) * (fft_freqs ** 2) / (f0 ** 2))
            row = np.fft.ifft(np.roll(F, -k) * g)
            acc[i] += float(np.mean(np.abs(row) ** 2))
    return freqs[keep], acc / X2.shape[0]


def m_vmd(x, fs=FS, K=6, **kw):
    """VMD mode energies placed at each mode's OWN centre frequency.

    Assigning modes to a fixed grid would throw away exactly what VMD estimates,
    so each mode's centre frequency is measured from its own spectrum.
    """
    from vmdpy import VMD
    X2 = np.atleast_2d(x)
    centres, powers = [], []
    for ch in X2:
        sig = ch.astype(float)
        if len(sig) % 2:                     # VMD requires even length
            sig = sig[:-1]
        u, _, omega = VMD(sig, alpha=2000.0, tau=0.0, K=K, DC=0, init=1, tol=1e-6)
        c = omega[-1] * fs                   # each mode's own centre freq, Hz
        e = np.var(u, axis=1)
        centres.append(c)
        powers.append(e)
    c = np.concatenate(centres); e = np.concatenate(powers)
    keep = (c >= F_MIN) & (c <= F_MAX)
    if not keep.any():
        return np.array([F_MIN]), np.array([0.0])
    o = np.argsort(c[keep])
    return c[keep][o], e[keep][o]


def m_ar(x, fs=FS, order=16, nfreq=257, **kw):
    """Parametric (autoregressive) spectrum -- a genuinely different estimator."""
    f = np.linspace(0, fs / 2, nfreq)
    acc = np.zeros(nfreq)
    for ch in np.atleast_2d(x):
        ch = ch - ch.mean()
        r = np.correlate(ch, ch, "full")[len(ch) - 1:len(ch) + order]
        R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
        try:
            a = np.linalg.solve(R + 1e-10 * np.eye(order), r[1:order + 1])
        except np.linalg.LinAlgError:
            continue
        w = 2 * np.pi * f / fs
        denom = 1 - (a[None, :] * np.exp(-1j * w[:, None] * np.arange(1, order + 1))).sum(1)
        # innovation variance: without this gain the AR spectrum is pure SHAPE and
        # completely scale-invariant (doubling the signal leaves it unchanged),
        # so total_power carries no amplitude information at all.
        sigma2 = max(float(r[0] - a @ r[1:order + 1]) / len(ch), 1e-20)
        acc += sigma2 / (np.abs(denom) ** 2 + 1e-20)
    return _band(f, acc / max(len(np.atleast_2d(x)), 1))


# --------------------------------------------------------------------------- #
def _per_freq_mean(S, n_freq, n_ch=None, square=False):
    """Stacked ``(n_ch*n_freq, T)`` -> mean power per frequency across channels.

    ``n_ch`` must be passed explicitly. Inferring it from the array shape is how
    the first version of this file silently mis-mapped SST and multitaper onto
    the wrong frequency grid.

    ``square=True`` for transforms that return |S| rather than |S|^2. Getting
    this wrong is not cosmetic: every power-weighted descriptor (mean_freq,
    median_freq, spread, entropy) uses P as the weight, so an amplitude-valued
    P weights low-power bins more heavily than a power-valued one and the
    "same" descriptor means a different quantity per method.
    """
    S = np.abs(np.asarray(S))
    if square:                    # transform returned AMPLITUDE -> make it POWER
        S = S ** 2
    if S.ndim == 1:
        S = S[:, None]
    if n_ch is None:
        n_ch = max(S.shape[0] // n_freq, 1)
    return S[:n_ch * n_freq].reshape(n_ch, n_freq, -1).mean(axis=(0, 2))


#: name -> callable(x, fs, **kw) -> (freqs, power). Add new methods here only.
METHODS = {
    "welch": m_welch,
    "stft256": m_stft,
    "stft512": m_stft512,
    "multitaper": m_multitaper,
    "cwt": m_cwt,
    "hht": m_hht,
    "hht_imf2plus": m_hht_imf2plus,
    "sst": m_sst,
    "wavelet_packet": m_wavelet_packet,
    "stransform": m_stransform,
    "vmd": m_vmd,
    "ar16": m_ar,
}
