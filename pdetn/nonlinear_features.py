"""Nonlinear-dynamics, higher-order-spectral, and parametric tremor features.

New signal-processing axes, chosen for the PD-vs-ET physiology (PD tremor is
more regular / deterministic; ET more variable, often with coupled harmonics):

  * DFA (detrended fluctuation analysis) — long-range temporal correlation.
  * RQA determinism / recurrence rate / entropy — periodicity of the attractor.
  * Poincaré SD1, SD2, ratio — short- vs long-term variability.
  * Higuchi fractal dimension — signal complexity.
  * Harmonic bicoherence — quadratic phase coupling between f0 and 2*f0.
  * AR pole — parametric dominant resonance frequency & bandwidth (sharpness).

Computed on the 3-15 Hz bandpassed hand-sensor magnitude. Torch-free.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from pdetn.signal_features import _tremor_magnitude   # bandpassed hand magnitude

TREMOR_LO, TREMOR_HI = 3.0, 15.0


def _dfa(sig, scales=(4, 8, 16, 32, 64)):
    x = np.cumsum(sig - sig.mean())
    F = []
    good = []
    for s in scales:
        if s >= len(x):
            continue
        n = len(x) // s
        if n < 2:
            continue
        segs = x[:n * s].reshape(n, s)
        t = np.arange(s)
        rms = []
        for seg in segs:
            c = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(c, t)) ** 2)))
        F.append(np.mean(rms)); good.append(s)
    if len(good) < 2:
        return float("nan")
    a = np.polyfit(np.log(good), np.log(np.array(F) + 1e-12), 1)[0]
    return float(a)


def _poincare(sig):
    x1, x2 = sig[:-1], sig[1:]
    sd1 = np.std((x1 - x2) / np.sqrt(2))
    sd2 = np.std((x1 + x2) / np.sqrt(2))
    return float(sd1), float(sd2), float(sd1 / (sd2 + 1e-12))


def _higuchi_fd(sig, kmax=8):
    N = len(sig); L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            ll = np.sum(np.abs(np.diff(sig[idx]))) * (N - 1) / (len(idx) * k)
            Lk.append(ll)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return float("nan")
    k = np.arange(1, len(L) + 1)
    return float(-np.polyfit(np.log(k), np.log(np.array(L) + 1e-12), 1)[0])


def _rqa(sig, m=3, tau=4, eps_q=0.2, lmin=2, max_len=350):
    s = sig[:: max(1, len(sig) // max_len)]
    s = (s - s.mean()) / (s.std() + 1e-12)
    n = len(s) - (m - 1) * tau
    if n < 10:
        return dict(rr=float("nan"), det=float("nan"), entr=float("nan"))
    emb = np.array([s[i:i + n] for i in range(0, m * tau, tau)]).T   # (n, m)
    D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
    eps = eps_q * D.max()
    R = (D <= eps).astype(int)
    rr = R.mean()
    # diagonal line lengths (excluding main diagonal)
    lengths = []
    for d in range(1, n):
        diag = np.diag(R, k=d)
        c = 0
        for v in diag:
            if v:
                c += 1
            elif c:
                lengths.append(c); c = 0
        if c:
            lengths.append(c)
    lengths = [l for l in lengths if l >= lmin]
    npts = R.sum() - np.trace(R)
    det = (sum(lengths) / npts) if npts > 0 else 0.0
    if lengths:
        vals, cnt = np.unique(lengths, return_counts=True)
        p = cnt / cnt.sum(); entr = float(-(p * np.log(p)).sum())
    else:
        entr = 0.0
    return dict(rr=float(rr), det=float(det), entr=entr)


def _harmonic_bicoherence(sig, fs):
    """Bicoherence-like quadratic phase coupling at (f0, f0)->2f0."""
    n = len(sig)
    w = np.hanning(n)
    X = np.fft.rfft((sig - sig.mean()) * w)
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= TREMOR_LO) & (f < TREMOR_HI)
    if not band.any():
        return float("nan")
    k0 = np.where(band)[0][np.argmax(np.abs(X[band]))]
    k2 = min(2 * k0, len(X) - 1)
    num = abs(X[k0] * X[k0] * np.conj(X[k2]))
    den = np.sqrt((abs(X[k0]) ** 2 * abs(X[k0]) ** 2) * abs(X[k2]) ** 2) + 1e-12
    return float(num / den)


def _ar_pole(sig, order=8, fs=100.0):
    """Dominant AR pole: resonance frequency and bandwidth (Yule-Walker)."""
    s = sig - sig.mean()
    r = np.correlate(s, s, mode="full")[len(s) - 1:len(s) + order]
    R = np.array([[r[abs(i - j)] for j in range(order)] for i in range(order)])
    try:
        a = np.linalg.solve(R + 1e-9 * np.eye(order), r[1:order + 1])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    poles = np.roots(np.concatenate([[1.0], -a]))
    poles = poles[np.imag(poles) > 0]
    if len(poles) == 0:
        return float("nan"), float("nan")
    freqs = np.angle(poles) * fs / (2 * np.pi)
    band = (freqs >= TREMOR_LO) & (freqs < TREMOR_HI)
    poles, freqs = (poles[band], freqs[band]) if band.any() else (poles, freqs)
    k = int(np.argmax(np.abs(poles)))
    bw = -np.log(np.abs(poles[k]) + 1e-12) * fs / np.pi
    return float(freqs[k]), float(abs(bw))


def nonlinear_features(x: np.ndarray, fs: float = 100.0) -> dict[str, float]:
    sig = _tremor_magnitude(x, fs)
    feats: dict[str, float] = {}
    feats["dfa_alpha"] = _dfa(sig)
    sd1, sd2, ratio = _poincare(sig)
    feats["poincare_sd1"] = sd1
    feats["poincare_sd2"] = sd2
    feats["poincare_ratio"] = ratio
    feats["higuchi_fd"] = _higuchi_fd(sig)
    rqa = _rqa(sig)
    feats["rqa_rr"] = rqa["rr"]
    feats["rqa_det"] = rqa["det"]
    feats["rqa_entr"] = rqa["entr"]
    feats["harmonic_bicoh"] = _harmonic_bicoherence(sig, fs)
    pf, pbw = _ar_pole(sig, fs=fs)
    feats["ar_pole_freq"] = pf
    feats["ar_pole_bw"] = pbw
    return feats


NONLINEAR_FEATURE_NAMES = (
    "dfa_alpha", "poincare_sd1", "poincare_sd2", "poincare_ratio", "higuchi_fd",
    "rqa_rr", "rqa_det", "rqa_entr", "harmonic_bicoh", "ar_pole_freq", "ar_pole_bw",
)
