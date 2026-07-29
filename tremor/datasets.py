"""PyTorch datasets for tremor classification.

Two datasets share the same loader/model interface:

* :class:`STFTDataset` consumes precomputed STFT magnitude matrices.
* :class:`TremorDataset` consumes raw amplitude or quaternion
  recordings and computes the TFD (STFT / CWT / HHT / wavelet packet)
  on the fly per item.

Both yield ``(channels, time)`` tensors and an integer label, and
share :func:`_oversample_per_class` for train-time class balancing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from tremor.data import Recording
from tremor.preprocessing import apply_stft
from tremor.spectral import (
    crop_freq_bins,
    fit_length,
    log_compress,
    per_freq_zscore,
    per_recording_zscore,
    spec_augment,
)
from tremor.stft_data import STFTRecording
from tremor.tfd import (
    apply_cwt,
    apply_hht,
    apply_multitaper,
    apply_sst,
    apply_wavelet_packet,
)


__all__ = [
    "STFTDataset",
    "TremorDataset",
    "_oversample_per_class",
]


def _oversample_per_class(recs, oversample_to, rng):
    """Oversample each class to ``oversample_to`` items.

    Returns the original list unchanged when ``oversample_to`` is None.
    Empty classes are skipped (they cannot be sampled from) instead of
    raising — set up that case with a clear message at the call site.
    """
    if oversample_to is None or oversample_to <= 0:
        return list(recs)
    by_class: dict[int, list] = {}
    for r in recs:
        by_class.setdefault(r.y, []).append(r)
    balanced = []
    for cls, items in by_class.items():
        if not items:
            continue
        idx = rng.integers(0, len(items), size=oversample_to)
        balanced.extend(items[i] for i in idx)
    return balanced


class STFTDataset(Dataset):
    """Dataset over precomputed STFT magnitude matrices.

    Per-item pipeline:
        1. (optional) crop the lowest freq bins per sensor (``--f-max``).
        2. (optional) ``log1p(S/eps)`` to compress dynamic range.
        3. (optional) per-frequency z-score across time within recording.
        4. random/centred zero-pad in TIME to ``target_T`` frames.
        5. (training only, optional) SpecAugment.
    """

    def __init__(
        self,
        recs: list[STFTRecording],
        target_T: int,
        rng_seed: int,
        n_sensors: int,
        n_freq_bins: int,
        keep_bins: int,
        log_compress_on: bool,
        normalize: str,  # 'none' | 'per_freq' | 'per_recording'
        spec_augment_on: bool,
        length_mode: str = "truncate",  # 'truncate' | 'pad'
        pad_mode: str = "random",       # 'center' | 'random' (only used in pad mode)
        oversample_to: int | None = None,
        augment: bool = False,
    ) -> None:
        self.target_T = target_T
        self.n_sensors = n_sensors
        self.n_freq_bins = n_freq_bins
        self.keep_bins = keep_bins
        self.log_compress_on = log_compress_on
        self.normalize = normalize
        self.spec_augment_on = spec_augment_on
        self.length_mode = length_mode
        self.pad_mode = pad_mode
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)
       
        self.recs = _oversample_per_class(recs, oversample_to, self.rng)

    def __len__(self) -> int:
        return len(self.recs)

    def _fit_mode(self) -> str:
        """Resolve crop/pad offset strategy from length_mode + augment + pad_mode."""
        if not self.augment:
            return "center"
        if self.length_mode == "truncate":
            return "random"      # random crop -> train-time augmentation
        return self.pad_mode      # 'random' or 'center' as the user chose

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.recs[i]
        x = r.x
        if self.keep_bins < self.n_freq_bins:
            x = crop_freq_bins(x, self.n_sensors, self.n_freq_bins, self.keep_bins)
        if self.log_compress_on:
            x = log_compress(x)
        if self.normalize == "per_freq":
            x = per_freq_zscore(x)
        elif self.normalize == "per_recording":
            x = per_recording_zscore(x)
        x = fit_length(x, self.target_T, mode=self._fit_mode(), rng=self.rng)
        if self.augment and self.spec_augment_on:
            x = spec_augment(x, self.rng)
        return torch.from_numpy(x), r.y


class TremorDataset(Dataset):
    def __init__(
        self,
        recs: list[Recording],
        target_length: int,
        fs: float = 100,
        nperseg: int = 128,
        nfft: int = 128,
        noverlap: int = 96,
        rng_seed: int = 42,
        f_max: float = 30.0,
        oversample_to: int | None = None,
        augment: bool = False,
        tfd_method: str = "stft",
        cwt_w0: float = 6.0,
        cwt_decim: int = 8,
        cwt_freq_step: float = 0.5,
        hht_max_imfs: int = 8,
        hht_emd_method: str = "emd",
        wp_level: int = 5,
        wp_wavelet: str = "db4",
        log_compress_on: bool = True,
        normalize: str = "per_recording",
        spec_augment_on: bool = False,
        length_mode: str = "truncate",
        pad_mode: str = "random",
    ) -> None:
        self.target_length = target_length
        self.fs = fs
        self.nperseg = nperseg
        self.nfft = nfft
        self.noverlap = noverlap
        self.f_max = f_max
        self.augment = augment
        self.tfd_method = tfd_method
        self.cwt_w0 = cwt_w0
        self.cwt_decim = cwt_decim
        self.cwt_freq_step = cwt_freq_step
        self.hht_max_imfs = hht_max_imfs
        self.hht_emd_method = hht_emd_method
        self.wp_level = wp_level
        self.wp_wavelet = wp_wavelet
        self.log_compress_on = log_compress_on
        self.normalize = normalize
        self.spec_augment_on = spec_augment_on
        self.length_mode = length_mode
        self.pad_mode = pad_mode
        self.rng = np.random.default_rng(rng_seed)
        self.recs = _oversample_per_class(recs, oversample_to, self.rng)

    def __len__(self) -> int:
        return len(self.recs)

    def _fit_mode(self) -> str:
        if not self.augment:
            return "center"
        if self.length_mode == "truncate":
            return "random"
        return self.pad_mode

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.recs[i]
        x = fit_length(r.x, self.target_length,
                        mode=self._fit_mode(), rng=self.rng)
        if self.tfd_method == "cwt":
            f_high = self.f_max if self.f_max else 15.0
            freqs = np.arange(3.0, f_high + 1e-9, self.cwt_freq_step)
            x = apply_cwt(
                x, fs=self.fs, freqs=freqs, w0=self.cwt_w0, decim=self.cwt_decim,
                f_max=self.f_max,
            )
        elif self.tfd_method == "hht":
            f_high = self.f_max if self.f_max else 15.0
            freqs = np.arange(3.0, f_high + 1e-9, self.cwt_freq_step)
            x = apply_hht(
                x, fs=self.fs, freqs=freqs, max_imfs=self.hht_max_imfs,
                decim=self.cwt_decim, f_max=self.f_max,
                emd_method=self.hht_emd_method,
            )
        elif self.tfd_method == "wavelet_packet":
            _bc, x = apply_wavelet_packet(
                x, fs=self.fs, level=self.wp_level,
                wavelet=self.wp_wavelet, f_max=self.f_max,
                log_energy=True,
            )
        elif self.tfd_method == "multitaper":
            x = apply_multitaper(
                x, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft,
                noverlap=self.noverlap, f_max=self.f_max,
            )
        elif self.tfd_method == "sst":
            x = apply_sst(
                x, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft,
                noverlap=self.noverlap, f_max=self.f_max,
            )
        else:
            x = apply_stft(
                x, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft,
                noverlap=self.noverlap, f_max=self.f_max,
            )

        if self.log_compress_on:
            x = log_compress(x)
        if self.normalize == "per_recording":
            x = per_recording_zscore(x)
        elif self.normalize == "per_freq":
            x = per_freq_zscore(x)
        if self.augment and self.spec_augment_on:
            x = spec_augment(x, self.rng)
        return torch.from_numpy(x), r.y
