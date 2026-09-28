"""Unified registry of signal-processing / time-frequency methods.

Every method maps a ``(channels, time)`` recording to a **power spectrum over
frequency**, so that the same frequency descriptors (max/mean/median frequency,
...) can be computed from all of them and compared on equal footing.

Kept estimators (the ones the pipeline uses): Welch and multitaper spectra,
STFT-512 (the descriptor table), and wavelet-packet band energies mapped to
their centre frequencies. The benchmark-only estimators (CWT, HHT, SST,
S-transform, VMD, AR) were removed on 2026-09-28; they are in git history
(commit 57fd8a72); none was adopted -- HHT measured worst and AR16
significantly worse, the rest a plateau (skill `references/closed_families.md`).

Channels are combined **without rectification**: each channel's spectrum is
computed separately and averaged. Taking `sqrt(sum(ch^2))` first would square a
6 Hz oscillation into 12 Hz energy -- a real artifact this repo has already been
bitten by (`reports/signal_processing_summary.md`).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch, stft

from signal_processing.tfd import apply_multitaper, apply_wavelet_packet

FS = 100.0
F_MIN, F_MAX = 3.0, 15.0


def _band(f, P, f_min=F_MIN, f_max=F_MAX):
    k = (f >= f_min) & (f <= f_max)
    return f[k], P[k]


def _kept_rfftfreq(nfft, fs, f_max=F_MAX):
    """The frequency axis ``apply_multitaper`` actually kept.

    It builds their grid with ``np.fft.rfftfreq(nfft, 1/fs)`` and then crop to
    ``f <= f_max``, returning magnitudes only. Reconstructing that axis as
    ``linspace(0, f_max, n_freq)`` -- which is what this module used to do -- is
    **wrong**, because the last kept bin is the largest multiple of ``fs/nfft``
    that does not exceed ``f_max``, not ``f_max`` itself.

    At ``nfft=256, fs=100`` the kept bins run 0 to 14.8438 Hz in steps of
    0.390625, while the linspace ran 0 to 15.0 in steps of 0.394737 -- a **1.05 %
    stretch of the whole frequency axis**, reaching +0.156 Hz at the top of the
    band. That is 14 % of the N-vs-ET mean-frequency gap (8.16 vs 7.04 Hz).

    The error was uniform across patients and cohorts, so it biased no class and
    no site, which is why it went unnoticed. It did put the multitaper spectrum
    on a different frequency scale from every other branch of the pipeline --
    ``m_welch`` and ``m_stft`` take their axes from SciPy and were always
    correct, and the descriptor table is built from ``stft512``.
    """
    f = np.fft.rfftfreq(int(nfft), d=1.0 / fs)
    return f[f <= f_max]


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
    f = _kept_rfftfreq(n, fs)
    assert len(f) == n_freq, f"axis {len(f)} != spectrum {n_freq}"
    return _band(f, P)


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


# --------------------------------------------------------------------------- #
def _per_freq_mean(S, n_freq, n_ch=None, square=False):
    """Stacked ``(n_ch*n_freq, T)`` -> mean power per frequency across channels.

    ``n_ch`` must be passed explicitly. Inferring it from the array shape is how
    the first version of this file silently mis-mapped multitaper onto
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
    "stft512": m_stft512,
    "multitaper": m_multitaper,
    "wavelet_packet": m_wavelet_packet,
}
