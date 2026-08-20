"""Fixed-length waveform tensors: the input no deep model here has ever seen.

Every neural network in this project reads **frequency** as its sequence axis.
`ResidualTCN` is "a TCN with actual residual blocks, over the frequency axis";
`SpectrumBiLSTM` reads "the power spectrum as a sequence over FREQUENCY";
`Spectrum1DCNN` is "a 1-D CNN over the frequency axis". The only time-domain
stream is `TrajectoryEncoder`, and it sees an instantaneous-frequency trajectory
already summarised to 64 points.

**No model has read the waveform itself.** That gap is now motivated rather than
merely present: `catch22_waveform_features.md` measured six temporal statistics
matching ten tuned spectral descriptors on PADS PD-vs-ET (AUC 0.798 vs 0.794) at
half the fold variance. Temporal structure carries comparable information, and a
convolution over time can extract it adaptively where catch22 uses 22 fixed
formulas.

The processing chain, each step for a reason:

``band-pass 3-15 Hz``     the band every other family uses; removes gravity,
                          drift and voluntary movement below the tremor band.
``principal-axis``        rotation-invariant reduction of 3 axes to 1. NOT the
                          magnitude: for a linear oscillation ||w(t)|| has
                          fundamental 2f (verified -- 11.91 Hz against 6.05 Hz on
                          a 6 Hz synthetic). Tremor is near-linear (linearity
                          0.997) so the projection keeps the waveform intact.
``sign fix``              the eigenvector sign is arbitrary under rotation; fixed
                          by forcing non-negative skewness so the pipeline is
                          deterministic.
``decimate to 40 Hz``     Nyquist 20 Hz still clears the 15 Hz band edge, and it
                          shortens the sequence 2.5x. Sequence length is the
                          binding cost for a TCN at 404 patients.
``z-score``               amplitude is not comparable across cohorts; every other
                          family here is scale-invariant and this must match.
``centre crop``           one fixed length so the flat-matrix training harness
                          can carry it, chosen so nothing is ever padded -- the
                          amount of padding would be a cohort signature.

Absolute phase is meaningless, which a convolution plus global pooling already
handles: the encoder is translation-equivariant and the pooling makes it
translation-invariant.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

FS_IN, FS_OUT = 100.0, 40.0
F_LO, F_HI = 3.0, 15.0
# 9.6 s at 40 Hz. Chosen so EVERY recording is cropped and none is padded:
# the shortest is 1000 samples at 100 Hz -> 400 at 40 Hz, and 2015's shortest is
# 1046 -> 418. Padding would otherwise be a cohort signature, since PADS is
# always 1024 samples, NewData always 1000, and 2015 varies.
LENGTH = 384


def waveform(x, fs_in=FS_IN, fs_out=FS_OUT, f_lo=F_LO, f_hi=F_HI,
             length=LENGTH):
    """(3, T) raw -> (length,) band-passed, rotation-invariant, z-scored.

    Returns ``None`` for recordings too short or degenerate to process.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:3]
    if x.shape[-1] < 128:
        return None
    x = x - x.mean(-1, keepdims=True)

    nyq = fs_in / 2.0
    try:
        b, a = butter(4, [f_lo / nyq, min(f_hi / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x, axis=-1)
    except Exception:
        return None

    if x.shape[0] >= 2:
        C = np.cov(x)
        if not np.isfinite(C).all():
            return None
        v = np.linalg.eigh(C)[1][:, -1]
        s = v @ x
    else:
        s = x[0]

    # decimate 100 -> 40 Hz; the anti-alias filter is inside resample_poly
    g = np.gcd(int(fs_out), int(fs_in))
    s = resample_poly(s, int(fs_out) // g, int(fs_in) // g)

    sd = s.std()
    if not np.isfinite(sd) or sd <= 1e-12:
        return None
    s = (s - s.mean()) / sd
    if float((s ** 3).mean()) < 0:          # deterministic sign
        s = -s

    if len(s) >= length:                    # centre crop
        i = (len(s) - length) // 2
        s = s[i:i + length]
    else:                                   # centre pad with zeros
        pad = length - len(s)
        s = np.pad(s, (pad // 2, pad - pad // 2))
    return np.nan_to_num(s.astype(np.float32))


def patient_tensor(recs, ch=slice(3, 6), n_rec=2, length=LENGTH):
    """(patients, n_rec, length) — up to n_rec recordings per patient.

    A patient's recordings cannot be averaged in the time domain (they have
    independent phase), so they are stacked and the model pools over them.
    Missing slots are zero-filled and marked in the returned mask.
    """
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        w = waveform(x, length=length)
        if w is None:
            continue
        rows[r.subject].append(w)
        lab[r.subject] = r.y

    pats = sorted(rows)
    X = np.zeros((len(pats), n_rec, length), np.float32)
    M = np.zeros((len(pats), n_rec), np.float32)
    for i, p in enumerate(pats):
        for j, w in enumerate(rows[p][:n_rec]):
            X[i, j], M[i, j] = w, 1.0
    return X, M, np.array([lab[p] for p in pats]), np.array(pats)


def analytic_channels(x, fs_in=FS_IN, fs_out=FS_OUT, f_lo=F_LO, f_hi=F_HI,
                      length=LENGTH, centre_if=True):
    """(3, T) raw -> (2, length): log envelope and instantaneous frequency.

    The raw waveform contains amplitude and phase entangled, so a network reading
    it must first learn to demodulate before it can see anything about *states* —
    and demodulation costs capacity this cohort cannot spare. The analytic signal
    does that step in closed form, handing the model the two quantities the
    Häring mechanism is actually about:

        channel 0   log envelope, z-scored  -- amplitude modulation over time
        channel 1   instantaneous frequency in Hz, centred on its own median
                    -- "several discrete oscillator states in PD" should appear
                    as switching between levels; "a singular pacemaker in ET" as
                    a flat line.

    Centring the IF on the patient's own median makes it a *stability* channel
    rather than a frequency channel, so it carries no absolute-frequency
    information the spectrum already has, and stays comparable across cohorts.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None, :]
    x = x[:3]
    if x.shape[-1] < 128:
        return None
    x = x - x.mean(-1, keepdims=True)

    nyq = fs_in / 2.0
    try:
        b, a = butter(4, [f_lo / nyq, min(f_hi / nyq, 0.99)], btype="band")
        x = filtfilt(b, a, x, axis=-1)
    except Exception:
        return None

    if x.shape[0] >= 2:
        C = np.cov(x)
        if not np.isfinite(C).all():
            return None
        s = np.linalg.eigh(C)[1][:, -1] @ x
    else:
        s = x[0]

    z = hilbert(s)
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst = np.diff(phase) * fs_in / (2 * np.pi)          # Hz, length T-1
    env = env[:len(inst)]

    keep = np.isfinite(inst) & np.isfinite(env) & (env > 0)
    if keep.sum() < 128:
        return None
    inst = np.clip(inst, f_lo, f_hi)
    le = np.log(np.clip(env, 1e-12, None))

    g = np.gcd(int(fs_out), int(fs_in))
    up, down = int(fs_out) // g, int(fs_in) // g
    le = resample_poly(le, up, down)
    inst = resample_poly(inst, up, down)

    sd = le.std()
    le = (le - le.mean()) / (sd + 1e-12) if sd > 1e-12 else le * 0.0
    # centre_if=True makes channel 1 a STABILITY channel (deviation from the
    # patient's own median); False leaves it as ABSOLUTE instantaneous frequency.
    # The choice is not cosmetic: absolute tremor frequency is the most
    # discriminative single quantity available (max+mean frequency give AUC 0.786
    # on PADS), so centring removes it deliberately. See `analytic_if_control.py`.
    if centre_if:
        inst = inst - np.median(inst)

    out = np.vstack([le, inst])
    if out.shape[-1] >= length:
        i = (out.shape[-1] - length) // 2
        out = out[:, i:i + length]
    else:
        pad = length - out.shape[-1]
        out = np.pad(out, ((0, 0), (pad // 2, pad - pad // 2)))
    return np.nan_to_num(out.astype(np.float32))


def patient_analytic(recs, ch=slice(3, 6), n_rec=2, length=LENGTH,
                     centre_if=True):
    """(patients, n_rec, 2, length) envelope+IF, plus mask, labels, ids."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        w = analytic_channels(x, length=length, centre_if=centre_if)
        if w is None:
            continue
        rows[r.subject].append(w)
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.zeros((len(pats), n_rec, 2, length), np.float32)
    M = np.zeros((len(pats), n_rec), np.float32)
    for i, p in enumerate(pats):
        for j, w in enumerate(rows[p][:n_rec]):
            X[i, j], M[i, j] = w, 1.0
    return X, M, np.array([lab[p] for p in pats]), np.array(pats)
