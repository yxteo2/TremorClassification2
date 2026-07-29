"""Interpretable tremor biomarkers from angular-velocity recordings.

A transparent, clinically-motivated feature set to sit alongside the deep model
for the PD-vs-ET differential. The physiology it encodes:

  * PD is a **rest** tremor, classically ~4-6 Hz.
  * ET is an **action/postural** tremor, classically ~6-12 Hz, often with
    visible harmonics.
  * Normal has little tremor-band power.

So the discriminative structure is (a) *where* the spectral peak sits, (b) *how
much* power is in the PD vs ET sub-bands, (c) the harmonic structure, and — the
key one — (d) how power/frequency **change between rest and action conditions**.

Everything here is torch-free (numpy + scipy). Features are computed from the
**hand** sensor (distal, carries the most tremor): the first three
angular-velocity channels.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

# Tremor sub-bands (Hz). Deliberately clinically named and non-overlapping.
BANDS = {
    "b_3_5": (3.0, 5.0),      # low / early PD
    "b_5_7": (5.0, 7.0),      # PD core
    "b_7_10": (7.0, 10.0),    # ET core
    "b_10_15": (10.0, 15.0),  # high / ET harmonics
}
TREMOR_LO, TREMOR_HI = 3.0, 15.0
HAND_CHANNELS = (0, 1, 2)     # angular_velocity: hand sensor is the first 3 ch


def welch_psd(x: np.ndarray, fs: float = 100.0, nperseg: int = 256):
    """PSD averaged over the given channels. x: (channels, time)."""
    T = x.shape[1]
    nper = int(min(nperseg, T))
    f, P = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    return f, P.mean(axis=0)      # average across channels -> (n_freq,)


def _bandpower(f, psd, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapz(psd[m], f[m])) if m.any() else 0.0


def recording_features(x: np.ndarray, fs: float = 100.0,
                       channels=HAND_CHANNELS) -> dict[str, float]:
    """Interpretable spectral features for one recording (one sensor)."""
    x = x[list(channels)]
    f, psd = welch_psd(x, fs=fs)
    band = (f >= TREMOR_LO) & (f < TREMOR_HI)
    fb, pb = f[band], psd[band]
    total = float(np.trapz(pb, fb)) + 1e-12

    feats: dict[str, float] = {}
    for name, (lo, hi) in BANDS.items():
        feats[f"pow_{name}"] = _bandpower(f, psd, lo, hi)
        feats[f"rel_{name}"] = feats[f"pow_{name}"] / total   # fraction of tremor power
    feats["pow_total"] = total

    # Dominant tremor frequency and its sharpness.
    k = int(np.argmax(pb))
    dom = float(fb[k])
    feats["dom_freq"] = dom
    feats["peak_power"] = float(pb[k])

    # Harmonic ratio: power at ~2x dominant vs at dominant (ET tends higher).
    def _near(centre, half=0.6):
        return _bandpower(f, psd, centre - half, centre + half)
    p1 = _near(dom) + 1e-12
    feats["harmonic_ratio"] = _near(2 * dom) / p1 if 2 * dom <= TREMOR_HI else 0.0

    # ET-band / PD-band power ratio.
    pd_pow = feats["pow_b_3_5"] + feats["pow_b_5_7"] + 1e-12
    et_pow = feats["pow_b_7_10"] + feats["pow_b_10_15"]
    feats["et_pd_ratio"] = et_pow / pd_pow

    # Spectral entropy over the tremor band (flat -> high, peaky -> low).
    p_norm = pb / pb.sum() if pb.sum() > 0 else np.ones_like(pb) / len(pb)
    feats["spec_entropy"] = float(-(p_norm * np.log(p_norm + 1e-12)).sum())
    return feats


def mean_band_psd(x: np.ndarray, fs: float = 100.0, channels=HAND_CHANNELS,
                  grid: np.ndarray | None = None):
    """PSD resampled onto a common frequency grid (for by-class averaging)."""
    if grid is None:
        grid = np.arange(TREMOR_LO, TREMOR_HI + 1e-9, 0.25)
    f, psd = welch_psd(x[list(channels)], fs=fs)
    return grid, np.interp(grid, f, psd)


FEATURE_NAMES = (
    [f"pow_{b}" for b in BANDS] + [f"rel_{b}" for b in BANDS] +
    ["pow_total", "dom_freq", "peak_power", "harmonic_ratio",
     "et_pd_ratio", "spec_entropy"]
)
