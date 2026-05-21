"""Quaternion -> angular velocity for IMU tremor data.

The raw_quaternion folder stores orientations as 12-column CSVs:
3 sensors x 4 components per row, sampled at fs=100 Hz. The
convention is scalar-last ``(x, y, z, w)``: a near-identity
orientation has its largest absolute component in the final column.

For tremor classification we want the SIGNAL of orientation change,
not orientation itself. Two ways to extract it:

  1. ``angular_velocity`` — the body-frame angular velocity
     ``omega = 2 * (dq/dt) * q^(-1)`` per sensor, giving a 9-channel
     time series (3 sensors x 3 axes). Theoretically clean: omega
     is a Lie-algebra-valued vector that the STFT can analyse
     directly. This is the recommended mode.

  2. ``components`` — keep the raw 12-channel quaternion stream and
     rely on a downstream bandpass (3-15 Hz) to strip the
     baseline orientation, leaving only tremor oscillations.
     Matches the Drive ``bp_filter`` path; cheaper but mixes the
     scalar and vector parts.
"""

from __future__ import annotations

import numpy as np


def _quat_components(q: np.ndarray, convention: str):
    """Split an (..., 4) quaternion into its scalar (w) and vector (x, y, z) parts."""
    if convention == "xyzw":
        return q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    if convention == "wxyz":
        return q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    raise ValueError(f"unknown convention {convention!r}; expected 'xyzw' or 'wxyz'")


def _stack_quat(w, x, y, z, convention: str) -> np.ndarray:
    if convention == "xyzw":
        return np.stack([x, y, z, w], axis=-1)
    return np.stack([w, x, y, z], axis=-1)


def quat_conjugate(q: np.ndarray, convention: str = "xyzw") -> np.ndarray:
    """Conjugate (= inverse for unit quaternions): (w, -x, -y, -z)."""
    w, x, y, z = _quat_components(q, convention)
    return _stack_quat(w, -x, -y, -z, convention)


def quat_multiply(p: np.ndarray, q: np.ndarray, convention: str = "xyzw") -> np.ndarray:
    """Hamilton product p * q. Vectorised over leading dims."""
    pw, px, py, pz = _quat_components(p, convention)
    qw, qx, qy, qz = _quat_components(q, convention)
    rw = pw * qw - px * qx - py * qy - pz * qz
    rx = pw * qx + px * qw + py * qz - pz * qy
    ry = pw * qy - px * qz + py * qw + pz * qx
    rz = pw * qz + px * qy - py * qx + pz * qw
    return _stack_quat(rw, rx, ry, rz, convention)


def _normalize_quaternions(Q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise each row to unit norm.

    Real IMU streams sometimes store quaternions that drift slightly off
    the unit sphere (typically by < 1 %). ``angular_velocity_from_quaternions``
    treats the conjugate as the inverse, which is only valid when
    ``|q| = 1``; without this step a 5 % deviation in |q| produces a
    ~10 % error in omega.
    """
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    return (Q / np.maximum(norms, eps)).astype(Q.dtype, copy=False)


def _enforce_sign_continuity(Q: np.ndarray) -> np.ndarray:
    """Flip ``q -> -q`` wherever it would otherwise discontinuously sign-flip.

    Quaternion-based pose estimators occasionally emit ``-q`` instead of
    ``q`` for the same rotation. The vector part of our angular-velocity
    formula is already insensitive to this (``vec(q * q^*) = 0``), but
    continuity makes the recovered scalar part meaningful and removes
    any ambiguity downstream.
    """
    out = Q.copy()
    flips = np.einsum("ij,ij->i", out[1:], out[:-1]) < 0
    # propagate cumulative flips
    cum = np.concatenate([[False], np.logical_xor.accumulate(flips)])
    out[cum] *= -1
    return out


def angular_velocity_from_quaternions(
    Q: np.ndarray, fs: float, convention: str = "xyzw",
    normalize: bool = True, fix_signs: bool = True,
) -> np.ndarray:
    """Body-frame angular velocity from a unit-quaternion sequence.

    Uses a central-difference approximation, which is O(dt^2) accurate
    and phase-shift-free:
        omega(t) = 2 * (q[t+1] - q[t-1]) * (fs/2) * q[t]^(-1)

    For unit q, ``q^(-1) = conj(q)``, which is what we use. The
    pre-processing step ``_normalize_quaternions`` ensures the unit
    assumption holds even if the input drifts slightly off the
    unit sphere; ``_enforce_sign_continuity`` removes spurious antipodal
    flips between frames.

    The scalar part of the resulting product is approximately zero (omega
    is a pure quaternion); we return only the vector part. The first and
    last samples are dropped, so output length is ``T - 2``.

    Note on frame:
        ``ω = 2 (dq/dt) q^{-1}`` is the SPACE-FRAME angular velocity.
        ``ω = 2 q^{-1} (dq/dt)`` would give body-frame. For small tremor
        rotations (~0.05 rad) the two agree to within a few percent and
        their spectral content is identical, so the choice does not
        affect tremor classification.

    Args:
        Q: (T, 4) quaternion sequence; need not be unit-norm.
        fs: sampling rate in Hz.
        convention: 'xyzw' (scalar last) or 'wxyz' (scalar first).
        normalize: pre-normalise each row to unit norm (default True).
        fix_signs: flip sign discontinuities frame-to-frame (default True).

    Returns:
        (T-2, 3) angular velocity in rad/s, ordered (omega_x, omega_y, omega_z).
    """
    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError(f"expected (T, 4) quaternion, got {Q.shape}")
    if Q.shape[0] < 3:
        raise ValueError("need at least 3 timesteps for central-difference omega")

    if normalize:
        Q = _normalize_quaternions(Q)
    if fix_signs:
        Q = _enforce_sign_continuity(Q)

    dq_dt = (Q[2:] - Q[:-2]) * (fs / 2.0)
    q_inv = quat_conjugate(Q[1:-1], convention)
    omega_q = 2.0 * quat_multiply(dq_dt, q_inv, convention)
    _, x_part, y_part, z_part = _quat_components(omega_q, convention)
    return np.stack([x_part, y_part, z_part], axis=-1).astype(np.float32, copy=False)


def process_quaternion_data(
    Q12: np.ndarray,
    fs: float = 100.0,
    mode: str = "angular_velocity",
    convention: str = "xyzw",
    n_sensors: int = 3,
) -> np.ndarray:
    """Convert a (T, 12) quaternion CSV to a (channels, time) signal.

    mode='angular_velocity'  -> (n_sensors * 3, T-2) in rad/s
    mode='components'        -> (n_sensors * 4, T)   raw transposed
    """
    if Q12.ndim != 2 or Q12.shape[1] != n_sensors * 4:
        raise ValueError(
            f"expected (T, {n_sensors * 4}) quaternion data, got {Q12.shape}"
        )
    if mode == "components":
        return Q12.T.astype(np.float32, copy=False)
    if mode != "angular_velocity":
        raise ValueError(
            f"unknown mode {mode!r}; expected 'angular_velocity' or 'components'"
        )

    T = Q12.shape[0]
    Q = Q12.reshape(T, n_sensors, 4)
    omegas = [
        angular_velocity_from_quaternions(Q[:, s, :], fs=fs, convention=convention)
        for s in range(n_sensors)
    ]
    return np.concatenate(omegas, axis=1).T.astype(np.float32, copy=False)
