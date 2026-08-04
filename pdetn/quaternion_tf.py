"""Quaternion-aware time-frequency representations for PD/ET discrimination.

Motivation. Unit quaternions live on S^3, so a component-wise STFT of
(w, x, y, z) is not a well-defined spectral operation -- it mixes the scalar
and vector parts and breaks the norm constraint. Worse for our problem, the
scalar reductions we have been using (per-axis power, vector magnitude) throw
away the one thing three axes give you that one axis does not: the **relative
phase between axes**, i.e. the shape and handedness of the 3-D orbit the limb
segment traces at each tremor frequency.

That matters here specifically. Our own results say the classes overlap ~70 %
along the frequency axis in two independent cohorts, and the only features that
ever moved ET-F1 were the *cross-sensor phase/coherence* ones. Cross-**axis**
phase is the same kind of information one level down, and we have never used it.

Three representations, cheapest to richest:

``polarization_stft``
    The workhorse. At each (t, f) cell the three axes give a complex 3-vector
    Z = [Zx, Zy, Zz]. Its power |Z|^2 is the usual (rotation-invariant)
    spectrogram, but Z also carries the orbit geometry:

      * **circularity** ||Im(conj(Z) x Z)|| / |Z|^2 in [0, 1] -- 0 = the motion
        is a straight line at that frequency (flexion-extension), 1 = a perfect
        circle (pronation-supination / pill-rolling).
      * **planarity** from the eigenvalues of the spectral covariance -- is the
        orbit confined to a plane or is it 3-D?
      * **linearity direction stability** -- how much the dominant axis wanders.

    All three are invariant to how the sensor was strapped on, because they are
    built from rotation-invariant contractions of Z. That is the property the
    raw log map does *not* have.

``qstft``
    True hypercomplex STFT of the pure quaternion f(t) = i*x + j*y + k*z, via
    the symplectic decomposition f = A + B*mu_perp with A, B complex w.r.t. a
    chosen pure unit quaternion mu. Returns the simplex/perplex magnitude pair,
    whose ratio is the hypercomplex statement of the same chirality idea.

``dual_stream``
    Tremor spectrogram stacked with the slowly-varying gravity (posture) vector,
    so static limb pose survives as explicit context instead of being
    differentiated away.

Everything takes ``(channels, time)`` with channels grouped as 3 axes per
sensor, matching :func:`tremor.quaternion.process_quaternion_data`.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

__all__ = [
    "polarization_stft",
    "polarization_features",
    "gravity_referenced_chirality",
    "gravity_chirality_features",
    "qstft",
    "qstft_features",
    "POLARIZATION_FEATURE_NAMES",
    "GRAV_CHIRALITY_FEATURE_NAMES",
    "QSTFT_FEATURE_NAMES",
]

_EPS = 1e-12


def _axis_stft(x, fs, nperseg, noverlap, nfft, f_max):
    """STFT of a ``(n, T)`` stack -> freqs, times, Z of shape ``(n, F, Tt)``.

    A complex input keeps its **two-sided** spectrum: for a complex signal the
    +f and -f halves are independent, and their asymmetry *is* the rotation
    direction. Forcing one-sided (or casting to real) would silently discard
    exactly the chirality this module exists to measure.
    """
    x = np.asarray(x)
    onesided = not np.iscomplexobj(x)
    f, t, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                   axis=-1, boundary=None, padded=False,
                   return_onesided=onesided)
    if not onesided:
        order = np.argsort(f)
        f, Z = f[order], Z[..., order, :]
    # scipy puts the frequency axis at -2 and time at -1 for axis=-1 input
    if f_max:
        keep = np.abs(f) <= f_max
        f, Z = f[keep], Z[..., keep, :]
    return f, t, Z


def polarization_stft(x3, fs=100.0, nperseg=256, noverlap=192, nfft=None,
                      f_max=15.0, f_min=3.0):
    """Time-frequency polarization analysis of one 3-axis sensor.

    Args:
        x3: ``(3, T)`` angular velocity or so(3) rotation vector.

    Returns a dict of ``(F, Tt)`` maps:
        ``power``       trace |Z|^2 (rotation-invariant spectrogram)
        ``circularity`` in [0, 1], 0 = linear orbit, 1 = circular orbit
        ``planarity``   in [0, 1], 1 = orbit confined to a plane
        plus ``freqs`` and ``times``.
    """
    x3 = np.asarray(x3, dtype=np.float64)
    if x3.shape[0] != 3:
        raise ValueError(f"expected (3, T) axis triple, got {x3.shape}")
    nfft = nfft or nperseg
    f, t, Z = _axis_stft(x3, fs, nperseg, noverlap, nfft, f_max)
    if f_min:
        keep = f >= f_min
        f, Z = f[keep], Z[:, keep, :]

    power = np.sum(np.abs(Z) ** 2, axis=0)                      # (F, Tt)

    # Circularity: the imaginary part of conj(Z) x Z is the "spin vector" of the
    # analytic orbit; its norm relative to the power is the degree of circular
    # polarization. Both numerator and denominator are rotation-invariant.
    Zc = np.conj(Z)
    spin = np.stack([
        Zc[1] * Z[2] - Zc[2] * Z[1],
        Zc[2] * Z[0] - Zc[0] * Z[2],
        Zc[0] * Z[1] - Zc[1] * Z[0],
    ], axis=0)
    circularity = 2.0 * np.linalg.norm(np.imag(spin), axis=0) / (2.0 * power + _EPS)

    # Planarity from the eigenvalues of the (3, 3) Hermitian spectral covariance
    # S = Z Z^H at each cell. lambda ordered descending; a purely planar orbit
    # has lambda_3 = 0.
    S = np.einsum("aft,bft->ftab", Z, np.conj(Z))
    ev = np.linalg.eigvalsh(S)                                  # ascending
    ev = np.clip(ev, 0.0, None)
    tot = ev.sum(axis=-1) + _EPS
    planarity = 1.0 - 3.0 * ev[..., 0] / tot                    # smallest / total

    out = {"freqs": f, "times": t, "power": power,
           "circularity": np.clip(circularity, 0.0, 1.0),
           "planarity": np.clip(planarity, 0.0, 1.0),
           "spin": np.imag(spin)}
    return out


def gravity_referenced_chirality(x3, g3, fs=100.0, **kw):
    """Signed orbit handedness, measured against gravity -- mount-invariant.

    ``circularity`` says *how round* the orbit is but throws the sign away, and
    :func:`qstft`'s chirality keeps the sign but measures it against an
    arbitrary axis ``mu`` fixed in the sensor frame -- so it changes if the
    sensor is strapped on rotated, and flips between left and right limbs.

    The fix is to reference the handedness to something anatomical. The spin
    pseudovector ``s = Im(conj(Z) x Z)`` rotates *with* the sensor, and so does
    the body-frame gravity direction ``g``; their dot product therefore does
    not, giving a signed handedness in [-1, 1] that is invariant to mounting:

        chi(f, t) = (s . g_hat) / |Z|^2

    +1 = the segment orbits counter-clockwise seen from below (gravity-down),
    -1 = clockwise, 0 = linear motion.

    Args:
        x3: ``(3, T)`` angular velocity for one sensor.
        g3: ``(3, T)`` body-frame gravity for the SAME sensor, from
            :func:`tremor.quaternion.gravity_from_quaternions`.
    """
    m = polarization_stft(x3, fs=fs, **kw)
    g = np.asarray(g3, dtype=np.float64)
    if g.shape[0] != 3:
        raise ValueError(f"expected (3, T) gravity, got {g.shape}")
    g_hat = g.mean(axis=1)
    g_hat = g_hat / (np.linalg.norm(g_hat) + _EPS)
    chi = np.einsum("a,aft->ft", g_hat, m["spin"]) / (m["power"] + _EPS)
    return {"freqs": m["freqs"], "times": m["times"], "power": m["power"],
            "chirality": np.clip(chi, -1.0, 1.0)}


GRAV_CHIRALITY_FEATURE_NAMES = ["gchir_peak", "gchir_weighted", "gchir_mean",
                                "gchir_std", "gchir_sign_frac"]


def gravity_chirality_features(x3, g3, fs=100.0, **kw):
    """Scalar summary of the gravity-referenced (mount-invariant) handedness."""
    m = gravity_referenced_chirality(x3, g3, fs=fs, **kw)
    P, C, f = m["power"], m["chirality"], m["freqs"]
    if P.size == 0:
        return dict.fromkeys(GRAV_CHIRALITY_FEATURE_NAMES, 0.0)
    spec = P.mean(axis=1)
    pk = int(np.argmax(spec))
    chi_f = (C * P).sum(axis=1) / (P.sum(axis=1) + _EPS)     # power-weighted
    w = spec / (spec.sum() + _EPS)
    return {"gchir_peak": float(chi_f[pk]),
            "gchir_weighted": float((chi_f * w).sum()),
            "gchir_mean": float(chi_f.mean()),
            "gchir_std": float(chi_f.std()),
            # fraction of tremor-band power that orbits in the positive sense
            "gchir_sign_frac": float(w[chi_f > 0].sum())}


POLARIZATION_FEATURE_NAMES = [
    "circ_peak", "circ_bandmean", "circ_std", "circ_lowband", "circ_highband",
    "circ_weighted", "plan_peak", "plan_bandmean", "plan_weighted",
    "circ_slope", "peak_freq", "peak_share",
]


def polarization_features(x3, fs=100.0, **kw):
    """Scalar summary of :func:`polarization_stft` for one sensor.

    The orbit-geometry numbers are read **at and around the tremor peak**, since
    circularity is only meaningful where there is power to be polarized.
    """
    m = polarization_stft(x3, fs=fs, **kw)
    f, P, C, PL = m["freqs"], m["power"], m["circularity"], m["planarity"]
    if P.size == 0:
        return dict.fromkeys(POLARIZATION_FEATURE_NAMES, 0.0)

    spec = P.mean(axis=1)                       # (F,) time-averaged power
    pk = int(np.argmax(spec))
    w = spec / (spec.sum() + _EPS)              # power weights over frequency

    circ_f = (C * P).sum(axis=1) / (P.sum(axis=1) + _EPS)   # power-weighted in time
    plan_f = (PL * P).sum(axis=1) / (P.sum(axis=1) + _EPS)
    mid = f[len(f) // 2]
    lo, hi = f <= mid, f > mid

    out = {
        "circ_peak": float(circ_f[pk]),
        "circ_bandmean": float(circ_f.mean()),
        "circ_std": float(circ_f.std()),
        "circ_lowband": float(circ_f[lo].mean()) if lo.any() else 0.0,
        "circ_highband": float(circ_f[hi].mean()) if hi.any() else 0.0,
        "circ_weighted": float((circ_f * w).sum()),
        "plan_peak": float(plan_f[pk]),
        "plan_bandmean": float(plan_f.mean()),
        "plan_weighted": float((plan_f * w).sum()),
        # does the orbit get rounder or flatter as frequency rises?
        "circ_slope": float(np.polyfit(f, circ_f, 1)[0]) if len(f) > 2 else 0.0,
        "peak_freq": float(f[pk]),
        "peak_share": float(spec[pk] / (spec.sum() + _EPS)),
    }
    return out


# --------------------------------------------------------------------------- #
# True hypercomplex STFT (symplectic decomposition)
# --------------------------------------------------------------------------- #
def qstft(x3, fs=100.0, nperseg=256, noverlap=192, nfft=None, f_max=15.0,
          f_min=3.0, mu=None):
    """Quaternion STFT of the pure quaternion signal f = i*x + j*y + k*z.

    Uses the symplectic (Cayley-Dickson) decomposition: pick a pure unit
    quaternion ``mu`` as the transform axis and an orthogonal pure unit ``nu``;
    then ``f = A + B*nu`` where A, B are ordinary complex signals in the
    subfield spanned by {1, mu}. The quaternion Fourier transform is
    ``QFT(f) = FFT(A) + FFT(B)*nu``, so a standard complex STFT of A and B is
    exact -- no approximation.

    ``|A|`` (simplex) and ``|B|`` (perplex) split the energy relative to the
    chosen axis; because A and B are complex, their spectra are **two-sided**,
    and the +f / -f asymmetry of the perplex part is the direction the limb
    orbits in the plane normal to ``mu`` -- the hypercomplex form of chirality.

    Caveat, stated plainly: unlike :func:`polarization_stft`, this depends on
    the choice of ``mu``, so it is *not* invariant to how the sensor was
    mounted. Prefer the polarization route for cross-subject features; this one
    is for inspecting a single axis convention.

    Returns dict with ``freqs`` (two-sided, sorted), ``times``, ``simplex``,
    ``perplex``, ``magnitude`` (= sqrt(|A|^2 + |B|^2)).
    """
    x3 = np.asarray(x3, dtype=np.float64)
    if x3.shape[0] != 3:
        raise ValueError(f"expected (3, T) axis triple, got {x3.shape}")
    nfft = nfft or nperseg

    mu = np.asarray(mu if mu is not None else [1.0, 1.0, 1.0], dtype=np.float64)
    mu = mu / (np.linalg.norm(mu) + _EPS)
    # any pure unit quaternion orthogonal to mu completes the basis
    seed = np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    nu = seed - mu * float(seed @ mu)
    nu = nu / (np.linalg.norm(nu) + _EPS)
    lam = np.cross(mu, nu)                       # mu * nu, completes {mu, nu, lam}

    # f = i x + j y + k z, with vector part v. Project onto the basis:
    #   A = 0 + mu * (v . mu)            (simplex part, complex in {1, mu})
    #   B = (v . nu) + mu * (v . lam)    (perplex coefficients)
    v = x3
    a_im = mu @ v                                # scalar part of A is 0 (pure)
    b_re, b_im = nu @ v, lam @ v

    A = (1j * a_im)[None, :]
    B = (b_re + 1j * b_im)[None, :]
    f, t, ZA = _axis_stft(A, fs, nperseg, noverlap, nfft, f_max)
    _, _, ZB = _axis_stft(B, fs, nperseg, noverlap, nfft, f_max)
    if f_min:
        keep = np.abs(f) >= f_min
        f, ZA, ZB = f[keep], ZA[..., keep, :], ZB[..., keep, :]

    simplex, perplex = np.abs(ZA[0]), np.abs(ZB[0])
    return {"freqs": f, "times": t, "simplex": simplex, "perplex": perplex,
            "magnitude": np.sqrt(simplex ** 2 + perplex ** 2)}


QSTFT_FEATURE_NAMES = ["q_balance", "q_balance_peak", "q_chirality",
                       "q_chirality_peak", "q_peak_freq"]


def qstft_features(x3, fs=100.0, **kw):
    """Scalar summary of the hypercomplex spectrum.

    ``q_balance``   simplex-vs-perplex energy split (which plane the motion is in)
    ``q_chirality`` (+f - -f) / (+f + -f) of the perplex part: orbit handedness
    """
    m = qstft(x3, fs=fs, **kw)
    S, P, f = m["simplex"], m["perplex"], m["freqs"]
    if S.size == 0:
        return dict.fromkeys(QSTFT_FEATURE_NAMES, 0.0)
    tot = S ** 2 + P ** 2
    spec = tot.mean(axis=1)
    pk = int(np.argmax(spec))
    bal_f = (S ** 2 - P ** 2).sum(axis=1) / (tot.sum(axis=1) + _EPS)

    # pair each +f bin with its -f partner to read the rotation direction
    pos, neg = f > 0, f < 0
    fp = f[pos]
    Ppos = (P ** 2).sum(axis=1)[pos]
    # -f bins in ascending order are the mirror of +f in descending order
    Pneg_sorted = (P ** 2).sum(axis=1)[neg][::-1]
    n = min(len(fp), len(Pneg_sorted))
    fp, Ppos, Pneg = fp[:n], Ppos[:n], Pneg_sorted[:n]
    chir_f = (Ppos - Pneg) / (Ppos + Pneg + _EPS)
    w = (Ppos + Pneg) / (Ppos.sum() + Pneg.sum() + _EPS)
    pk_pos = int(np.argmax(Ppos + Pneg)) if n else 0

    return {"q_balance": float(bal_f.mean()),
            "q_balance_peak": float(bal_f[pk]),
            "q_chirality": float((chir_f * w).sum()) if n else 0.0,
            "q_chirality_peak": float(chir_f[pk_pos]) if n else 0.0,
            "q_peak_freq": float(abs(f[pk]))}
