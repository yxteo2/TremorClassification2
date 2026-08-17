"""Re-emergent tremor: does the tremor appear immediately, or after a delay?

Parkinsonian rest tremor characteristically **disappears when the arms are first
held out and re-emerges after a latency** of roughly 2-10 s. Essential tremor is
postural by nature and is present from the moment the posture is adopted. That
is a purely temporal signature, and nothing in this project could see it:

* the spectrum is averaged over the whole recording, so time is gone;
* ``select_task_epoch`` (NewData) picks the window with the MOST tremor power,
  which deliberately skips the latency period;
* the IF trajectory normalises the envelope by its own mean and resamples to a
  fixed length, so relative shape survives but absolute onset timing does not.

This module measures it directly, in absolute time from the start of the
recording.

**Feasibility caveats, measured rather than assumed.** Recording durations are
2015 OUT median 15.5 s (10.5-30.3), PADS StretchHold uniformly 10.24 s, NewData
~38 s but only if loaded with ``segment=False``. A 10 s window can show an
early-versus-late contrast but cannot resolve a latency longer than itself, and
none of the cohorts documents whether the recording starts exactly at task
onset. A null result here is therefore weak evidence against the phenomenon --
it may mean the recordings are not aligned to the posture being adopted.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, hilbert

FEATURE_NAMES = ("onset_latency", "early_late_ratio", "env_slope",
                 "late_energy_frac", "env_rise_range")


def tremor_envelope(x, fs=100.0, f_lo=3.0, f_hi=15.0, smooth_s=0.5):
    """Tremor-band amplitude envelope in ABSOLUTE time from recording start."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    lo, hi = f_lo / (fs / 2), min(f_hi / (fs / 2), 0.99)
    try:
        b, a = butter(4, [lo, hi], btype="band")
        xb = filtfilt(b, a, x, axis=-1)
    except Exception:
        return np.zeros(x.shape[-1])
    env = np.abs(hilbert(xb, axis=-1)).mean(0)      # average axes: rotation-safe
    k = max(int(smooth_s * fs), 1)
    return uniform_filter1d(env, k)


def reemergence_features(x, fs=100.0, **kw):
    """Timing features of the envelope, relative to the start of the recording."""
    out = {k: float("nan") for k in FEATURE_NAMES}
    env = tremor_envelope(x, fs=fs, **kw)
    n = len(env)
    if n < int(4 * fs) or not np.isfinite(env).all() or env.max() <= 0:
        return out
    t = np.arange(n) / fs
    med = float(np.median(env))

    # time at which the envelope first reaches half its median level --
    # small for a tremor present from the start, large for a delayed one
    above = np.flatnonzero(env >= 0.5 * med)
    out["onset_latency"] = float(t[above[0]]) if above.size else float(t[-1])

    third = n // 3
    early, late = env[:third].mean(), env[-third:].mean()
    out["early_late_ratio"] = float(early / (late + 1e-12))

    # slope of the envelope over the recording, scaled by its own level so the
    # feature is amplitude-invariant (raw amplitude is severity, not diagnosis)
    out["env_slope"] = float(np.polyfit(t, env, 1)[0] / (med + 1e-12))

    half = n // 2
    e1, e2 = float((env[:half] ** 2).sum()), float((env[half:] ** 2).sum())
    out["late_energy_frac"] = float(e2 / (e1 + e2 + 1e-12))

    # how much the envelope climbs, normalised -- a re-emerging tremor rises
    out["env_rise_range"] = float((env.max() - env[:third].mean()) / (med + 1e-12))
    return out


def patient_table(recs, ch=slice(0, 3), fs=100.0, **kw):
    """(patients, 5) re-emergence features, averaged over a patient's recordings."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        d = reemergence_features(sig, fs=fs, **kw)
        rows[r.subject].append([d[k] for k in FEATURE_NAMES])
        lab[r.subject] = r.y
    pats = sorted(rows)
    X = np.array([np.nanmean(rows[p], axis=0) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))
