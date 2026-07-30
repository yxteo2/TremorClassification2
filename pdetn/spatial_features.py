"""Temporal-spatial features: exploit the 3-sensor arm geometry.

The recordings carry three IMUs down the arm — hand (distal), lower_arm,
upper_arm (proximal). Tremor *propagation* and *spatial distribution* along the
arm are a discriminative axis we had not used (prior features were hand-only or
channel-stacked without spatial structure).

Physiology this targets:
  * Tremor is usually distal-dominant, but the distal→proximal power gradient
    and how coherently tremor propagates up the arm can differ between PD (rest,
    focal) and ET (postural, can involve more proximal/whole-limb).
  * Cross-sensor coherence/phase in the tremor band captures rigid vs
    distributed oscillation.

Per recording, from a 3-15 Hz bandpassed vector-magnitude signal per sensor:
per-sensor power, distal→proximal gradients, distal concentration, pairwise
coherence & phase, and cross-sensor dominant-frequency consistency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, coherence, csd, sosfiltfilt, welch

SENSORS = {"hand": (0, 1, 2), "lower": (3, 4, 5), "upper": (6, 7, 8)}
TREMOR_LO, TREMOR_HI = 3.0, 15.0


def _sensor_mag(x: np.ndarray, fs: float, chans) -> np.ndarray:
    sos = butter(4, [TREMOR_LO, TREMOR_HI], btype="band", fs=fs, output="sos")
    filt = sosfiltfilt(sos, x[list(chans)], axis=-1)
    return np.sqrt(np.sum(filt ** 2, axis=0))


def spatial_features(x: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    sig = {s: _sensor_mag(x, fs, ch) for s, ch in SENSORS.items()}
    P = {s: float(np.var(v)) + 1e-12 for s, v in sig.items()}
    feats: dict[str, float] = {}

    for s in SENSORS:
        feats[f"logpow_{s}"] = float(np.log10(P[s]))
    # distal -> proximal gradients (tremor spread up the arm)
    feats["logratio_hand_upper"] = float(np.log10(P["hand"] / P["upper"]))
    feats["logratio_hand_lower"] = float(np.log10(P["hand"] / P["lower"]))
    feats["logratio_lower_upper"] = float(np.log10(P["lower"] / P["upper"]))
    feats["distal_concentration"] = float(P["hand"] / sum(P.values()))

    nper = int(min(256, len(sig["hand"])))
    for a, b in (("hand", "lower"), ("hand", "upper"), ("lower", "upper")):
        f, Cxy = coherence(sig[a], sig[b], fs=fs, nperseg=nper)
        band = (f >= TREMOR_LO) & (f < TREMOR_HI)
        feats[f"coh_{a}_{b}"] = float(np.mean(Cxy[band]))
        # phase of the cross-spectrum at the peak-coherence frequency
        fb, Cb = f[band], Cxy[band]
        f2, Pxy = csd(sig[a], sig[b], fs=fs, nperseg=nper)
        idx = int(np.argmin(np.abs(f2 - fb[int(np.argmax(Cb))])))
        feats[f"absphase_{a}_{b}"] = float(abs(np.angle(Pxy[idx])))

    doms = {}
    for s, v in sig.items():
        f, Pw = welch(v, fs=fs, nperseg=nper)
        band = (f >= TREMOR_LO) & (f < TREMOR_HI)
        doms[s] = float(f[band][int(np.argmax(Pw[band]))])
    feats["domfreq_std_sensors"] = float(np.std(list(doms.values())))
    feats["domfreq_hand"] = doms["hand"]
    return feats


SPATIAL_FEATURE_NAMES = (
    "logpow_hand", "logpow_lower", "logpow_upper",
    "logratio_hand_upper", "logratio_hand_lower", "logratio_lower_upper",
    "distal_concentration", "coh_hand_lower", "coh_hand_upper", "coh_lower_upper",
    "absphase_hand_lower", "absphase_hand_upper", "absphase_lower_upper",
    "domfreq_std_sensors", "domfreq_hand",
)
