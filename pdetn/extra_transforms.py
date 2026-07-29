"""Extra time-frequency methods: VMD and the S-transform (Stockwell).

Added to complete the TF comparison. Both reduce a recording to a per-recording
feature vector comparable in size to the STFT/CWT/HHT features in
``pdetn.separability``, so separability and two-stage scores are apples-to-apples.

* **S-transform** — frequency-dependent Gaussian windows (STFT/CWT hybrid). Per
  channel: |S| over 3-15 Hz, reduced to mean+std spectral profile on a fixed
  grid.
* **VMD** — Variational Mode Decomposition. Per channel: K modes with compact
  centre frequencies; features are the sorted mode centre frequencies and their
  log energies (VMD's strength is exactly resolving tremor's centre frequency,
  which separates PD from ET).
"""

from __future__ import annotations

import numpy as np

from tremor.data import CLASS_NAMES

TREMOR_LO, TREMOR_HI = 3.0, 15.0
N_GRID = 24               # freq bins per channel for the S-transform profile


def _fit(sig: np.ndarray, target_length: int) -> np.ndarray:
    if len(sig) >= target_length:
        start = (len(sig) - target_length) // 2
        return sig[start:start + target_length]
    return np.pad(sig, (0, target_length - len(sig)))


def stransform_recording(x: np.ndarray, fs: float, target_length: int) -> np.ndarray:
    """Per-channel S-transform -> mean+std spectral profile on a fixed grid."""
    from stockwell import st
    grid = np.linspace(TREMOR_LO, TREMOR_HI, N_GRID)
    feats = []
    for ch in x:
        sig = _fit(ch.astype(float), target_length)
        n = len(sig)
        lo, hi = int(TREMOR_LO * n / fs), int(TREMOR_HI * n / fs)
        S = np.abs(st.st(sig, lo, hi))                 # (F, T)
        f = np.linspace(TREMOR_LO, TREMOR_HI, S.shape[0])
        mean_prof = np.interp(grid, f, S.mean(axis=1))
        std_prof = np.interp(grid, f, S.std(axis=1))
        feats.append(np.concatenate([mean_prof, std_prof]))
    return np.concatenate(feats)


def vmd_recording(x: np.ndarray, fs: float, K: int = 6,
                  alpha: float = 2000.0) -> np.ndarray:
    """Per-channel VMD -> sorted mode centre freqs (Hz) and log energies."""
    from vmdpy import VMD
    feats = []
    for ch in x:
        sig = ch.astype(float)
        if len(sig) % 2:                      # VMD wants even length
            sig = sig[:-1]
        u, _, omega = VMD(sig, alpha=alpha, tau=0.0, K=K, DC=0, init=1, tol=1e-6)
        centre = omega[-1] * fs               # Hz
        energy = np.log10(np.var(u, axis=1) + 1e-12)
        order = np.argsort(centre)            # sort modes by centre frequency
        feats.append(np.concatenate([centre[order], energy[order]]))
    return np.concatenate(feats)


def extra_method_features(recs, which: str, fs: float = 100.0, **kw):
    """(X, y, subjects) for 'stransform' or 'vmd', recording-level."""
    target_length = int(min(r.x.shape[1] for r in recs))
    rows = []
    for r in recs:
        if which == "stransform":
            rows.append(stransform_recording(r.x, fs, target_length))
        elif which == "vmd":
            rows.append(vmd_recording(r.x, fs, **kw))
        else:
            raise ValueError(which)
    X = np.stack(rows)
    y = np.array([r.y for r in recs])
    subjects = np.array([r.subject for r in recs])
    return X, y, subjects
