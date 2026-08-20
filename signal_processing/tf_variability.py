"""Keep how the spectrum MOVES, not just where it sits.

Every transform in `signal_processing/transforms.py` computes a time-frequency
surface and then collapses it to its frequency marginal before returning:

    f, P = welch(x, ...);  return _band(f, P.mean(0))        # m_welch
    P = _per_freq_mean(S, n_freq, n_ch, square=True)          # m_multitaper

So a project whose stated line of work is *time-frequency* processing has been
using only the **frequency** axis. The time axis is averaged away at the first
step, and every downstream family — descriptors, harmonics, axis shape — inherits
that loss.

The obvious fix, feeding the 2-D surface to a network, has already failed here:
ImageNet backbones on spectrograms measured at chance, and
`time_domain_deep.md` showed learned time-axis models lose to fixed formulas at
this sample size. So this keeps the time axis **as fixed summary statistics per
frequency bin** rather than as something to be learned:

    median per bin      (16)   what the current pipeline already has
    IQR per bin         (16)   how much that bin's share fluctuates
    spectral flux        (1)   mean L1 change between consecutive frames
    peak wander          (1)   sd of the per-frame peak frequency, in Hz

34 numbers, comparable to the 10 descriptors and 16 spectrum bins already in use,
so the dimensionality constraint that has killed thirteen feature unions is
respected.

**Why variability specifically.** Two independent results this session point at
it. catch22's best single feature for PD vs ET is `SB_MotifThree_quantile_hh`
(Cohen's d 1.235), a state-transition statistic; and the instantaneous-frequency
stability channel orders the classes N 2.705 > PD 2.322 > ET 1.946 Hz — the
direction Häring's mechanism predicts, where ET is one stable pacemaker and PD is
several switching oscillators. A spectrum that is *steady* over time is the
signature of a single pacemaker; one that fluctuates is the signature of several.
The frequency marginal cannot express that distinction at all.

**Frame normalisation.** Each frame is normalised to sum 1 before binning, so the
features describe the *shape* of the spectrum and its movement, not amplitude.
This matches the scale-invariance every other family here has, and it means
overall loudness fluctuation does not masquerade as spectral instability.

**Window length is the real knob and has never been swept.** Every transform in
the repo is fixed at nperseg 256 or 512 — 2.6 s or 5.1 s at 100 Hz, which is 20
to 60 cycles of a 4-12 Hz tremor. Resolving *state switching* needs short frames;
resolving *frequency* needs long ones. That trade-off is the defining choice in
time-frequency analysis and this project has never tested it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import butter, filtfilt, stft

FS, F_LO, F_HI, NBIN = 100.0, 3.0, 15.0, 16

# Every frame is interpolated onto this fixed grid before binning. Without it the
# usable window lengths are silently restricted: at nperseg 64 the resolution is
# 1.56 Hz, so 3-15 Hz holds only ~8 native bins and 16 log-bins cannot be formed
# at all. Interpolating decouples the number of output bins from the window
# length, which is the whole point of sweeping the window.
GRID = np.linspace(F_LO, F_HI, 64)

BLOCK_NAMES = ("median", "iqr", "flux", "wander")


def _logbin_frames(P, f, nb=NBIN):
    """(T, F) band-limited power on grid f -> (T, nb) log-power, all bins used."""
    P = np.stack([np.interp(GRID, f, row) for row in P])
    L = np.log(np.clip(P, 0, None) + 1e-12)
    e = np.linspace(0, L.shape[1], nb + 1).round().astype(int)
    return np.stack([L[:, e[i]:e[i + 1]].mean(1) for i in range(nb)], 1)


def tf_features(x, fs=FS, nperseg=128, f_lo=F_LO, f_hi=F_HI, nb=NBIN,
                stat="median"):
    """(3, T) -> 2*nb + 2 features describing the spectrum and its movement.

    Returns ``None`` if the recording cannot supply at least four frames, since
    a variability statistic over fewer than that is meaningless.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:3]
    # 75 % overlap, so four frames need only nperseg + 3*(nperseg//4) samples.
    # At 50 % overlap a 1024-sample PADS recording yields 3 frames at nperseg
    # 512 and was silently dropped.
    if x.shape[-1] < nperseg + 3 * (nperseg // 4):
        return None
    x = x - x.mean(-1, keepdims=True)

    nyq = fs / 2.0
    try:
        b, a = butter(4, [f_lo / nyq, min(f_hi / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x, axis=-1)
    except Exception:
        return None

    n = min(nperseg, x.shape[-1])
    f, _, Z = stft(x, fs=fs, nperseg=n, noverlap=3 * n // 4, axis=-1,
                   boundary=None, padded=False)
    if Z.ndim < 3 or Z.shape[-1] < 4:
        return None
    P = (np.abs(Z) ** 2).mean(0)                    # axes averaged -> (F, T)
    k = (f >= f_lo) & (f <= f_hi)
    if k.sum() < 4:
        return None
    P, fb = P[k].T, f[k]                            # (T, F)

    tot = P.sum(1, keepdims=True)
    good = tot[:, 0] > 0
    if good.sum() < 4:
        return None
    P, tot = P[good], tot[good]
    Pn = P / tot                                    # per-frame shape

    B = _logbin_frames(Pn, fb, nb)                  # (T, nb) log-shape
    # ``stat`` exists to separate two explanations of the short-window gain:
    # a median over many frames is ROBUST to transients, while a mean is not.
    # If mean and median score the same, the gain is the window, not the
    # estimator. See `experiments/tf_window_control.py`.
    med = np.median(B, 0) if stat == "median" else B.mean(0)
    iqr = np.percentile(B, 75, 0) - np.percentile(B, 25, 0)
    flux = float(np.abs(np.diff(Pn, axis=0)).sum(1).mean())
    peak = GRID[np.argmax(np.stack([np.interp(GRID, fb, r) for r in Pn]), 1)]
    wander = float(peak.std())

    out = np.concatenate([med, iqr, [flux, wander]])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def blocks(nb=NBIN):
    """Index slices for each named block of the feature vector."""
    return {"median": slice(0, nb), "iqr": slice(nb, 2 * nb),
            "flux": slice(2 * nb, 2 * nb + 1),
            "wander": slice(2 * nb + 1, 2 * nb + 2),
            "variability": slice(nb, 2 * nb + 2)}


def patient_table(recs, ch=slice(3, 6), nperseg=128, fs=FS, nb=NBIN,
                  stat="median"):
    """(patients, 2*nb + 2) averaged over each patient's recordings."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        v = tf_features(x, fs=fs, nperseg=nperseg, nb=nb, stat=stat)
        if v is None:
            continue
        rows[r.subject].append(v)
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = (np.array([np.mean(rows[p], 0) for p in pats]) if pats
         else np.zeros((0, 2 * nb + 2)))
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]),
            np.array(pats))
