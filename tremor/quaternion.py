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


# Sensor labels by index — distal-to-proximal along the arm.
SENSOR_NAMES = ("hand", "lower_arm", "upper_arm")


def select_sensor_channels(
    x: np.ndarray, sensors: list[str] | tuple[str, ...],
    mode: str = "angular_velocity",
) -> np.ndarray:
    """Keep only the rows of ``x`` belonging to the requested sensors.

    The tremor literature (Deuschl 1998; Elble 2009) focuses on the
    distal limb because tremor amplitude is largest there; selecting
    ``sensors=['hand']`` reduces 9-channel angular-velocity (or
    12-channel quaternion) input to 3 (or 4) hand-only channels.

    Args:
        x: ``(n_channels, time)`` array from :func:`process_quaternion_data`.
        sensors: subset of ``{'hand', 'lower_arm', 'upper_arm'}``.
        mode: ``'angular_velocity'`` (3 ch/sensor) or ``'components'`` (4).

    Returns:
        ``(len(sensors) * channels_per_sensor, time)`` slice of ``x``.
    """
    per_sensor = 3 if mode == "angular_velocity" else 4
    name_to_idx = {n: i for i, n in enumerate(SENSOR_NAMES)}
    keep: list[int] = []
    for s in sensors:
        if s not in name_to_idx:
            raise ValueError(f"unknown sensor {s!r}; must be one of {SENSOR_NAMES}")
        i = name_to_idx[s]
        keep.extend(range(i * per_sensor, (i + 1) * per_sensor))
    return x[keep]


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


def log_map_from_quaternions(
    Q: np.ndarray, convention: str = "xyzw", reference: str = "median",
    normalize: bool = True, fix_signs: bool = True, eps: float = 1e-8,
) -> np.ndarray:
    """Lie-algebra (so(3)) rotation vector for a unit-quaternion sequence.

    Maps each orientation off the S^3 manifold into R^3 via the quaternion
    logarithm, ``theta = 2 * ln(q) = 2 * arccos(w) * v / |v|``, so the result is
    a genuine Euclidean 3-vector whose norm is the rotation angle and whose
    direction is the instantaneous axis of rotation. Component-wise spectral
    analysis is then well defined, which it is not on the raw (w, x, y, z).

    ``reference`` controls what the rotation is measured *relative to*, which
    matters because an absolute log map encodes the sensor's mounting pose --
    subject-specific nuisance that can be learned instead of tremor:

    * ``'median'``  (default) -- relative to the recording's median orientation,
      i.e. ``theta(t) = 2 ln(q_ref^* * q(t))``. Mount-invariant: rotating the
      whole recording by a fixed R leaves theta unchanged. Use this.
    * ``'first'``   -- relative to the first sample (drift-sensitive).
    * ``'none'``    -- absolute pose. Keeps mounting orientation; only sensible
      if you deliberately want the posture stream.

    Returns ``(T, 3)`` in radians.
    """
    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError(f"expected (T, 4) quaternion, got {Q.shape}")
    if normalize:
        Q = _normalize_quaternions(Q)
    if fix_signs:
        Q = _enforce_sign_continuity(Q)

    if reference == "median":
        ref = _normalize_quaternions(np.median(Q, axis=0, keepdims=True))
        Q = quat_multiply(quat_conjugate(ref, convention), Q, convention)
    elif reference == "first":
        Q = quat_multiply(quat_conjugate(Q[:1], convention), Q, convention)
    elif reference != "none":
        raise ValueError(f"unknown reference {reference!r}")

    w, x, y, z = _quat_components(Q, convention)
    # antipodal fix: ln is defined on the w >= 0 hemisphere
    flip = w < 0
    w = np.where(flip, -w, w)
    v = np.stack([x, y, z], axis=-1)
    v = np.where(flip[:, None], -v, v)

    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))[:, None]
    # near identity, theta -> 2 * v (the arccos/|v| ratio is 2 in the limit)
    axis = np.where(v_norm > eps, v / np.maximum(v_norm, eps), 0.0)
    theta = np.where(v_norm > eps, angle * axis, 2.0 * v)
    return theta.astype(np.float32, copy=False)


def gravity_from_quaternions(
    Q: np.ndarray, convention: str = "xyzw", normalize: bool = True,
) -> np.ndarray:
    """Body-frame gravity direction ``g_local = q^* * g_global * q``.

    Encodes the static posture of the segment (which way the limb points) with
    no dependence on tremor dynamics. Useful as an explicit low-frequency
    context stream alongside a tremor-band spectrogram; PD (rest) and ET
    (postural/action) tremor are elicited in different limb postures.

    Returns ``(T, 3)``, unit-norm rows.
    """
    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError(f"expected (T, 4) quaternion, got {Q.shape}")
    if normalize:
        Q = _normalize_quaternions(Q)
    g = np.zeros((Q.shape[0], 4), dtype=Q.dtype)
    gw_idx = 3 if convention == "xyzw" else 0
    zi = 2 if convention == "xyzw" else 3
    g[:, gw_idx] = 0.0
    g[:, zi] = -1.0                      # gravity along -Z in the world frame
    rotated = quat_multiply(
        quat_multiply(quat_conjugate(Q, convention), g, convention), Q, convention
    )
    _, x, y, z = _quat_components(rotated, convention)
    return np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)


#: Channels produced per sensor by each :func:`process_quaternion_data` mode.
MODE_CHANNELS = {
    "angular_velocity": 3, "components": 4, "log_map": 3, "gravity": 3,
    "log_map_gravity": 6,
}


def process_quaternion_data(
    Q12: np.ndarray,
    fs: float = 100.0,
    mode: str = "angular_velocity",
    convention: str = "xyzw",
    n_sensors: int = 3,
    log_map_reference: str = "median",
) -> np.ndarray:
    """Convert a (T, 12) quaternion CSV to a (channels, time) signal.

    mode='angular_velocity'  -> (n_sensors * 3, T-2) in rad/s
    mode='components'        -> (n_sensors * 4, T)   raw transposed
    mode='log_map'           -> (n_sensors * 3, T)   so(3) rotation vector (rad)
    mode='gravity'           -> (n_sensors * 3, T)   body-frame gravity (posture)
    mode='log_map_gravity'   -> (n_sensors * 6, T)   log map stacked over gravity
    """
    if Q12.ndim != 2 or Q12.shape[1] != n_sensors * 4:
        raise ValueError(
            f"expected (T, {n_sensors * 4}) quaternion data, got {Q12.shape}"
        )
    if mode == "components":
        return Q12.T.astype(np.float32, copy=False)
    if mode not in MODE_CHANNELS:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(MODE_CHANNELS)}"
        )

    T = Q12.shape[0]
    Q = Q12.reshape(T, n_sensors, 4)

    def per_sensor(s):
        q = Q[:, s, :]
        if mode == "angular_velocity":
            return angular_velocity_from_quaternions(q, fs=fs, convention=convention)
        if mode == "log_map":
            return log_map_from_quaternions(q, convention=convention,
                                            reference=log_map_reference)
        if mode == "gravity":
            return gravity_from_quaternions(q, convention=convention)
        return np.concatenate([
            log_map_from_quaternions(q, convention=convention,
                                     reference=log_map_reference),
            gravity_from_quaternions(q, convention=convention),
        ], axis=1)

    return np.concatenate([per_sensor(s) for s in range(n_sensors)],
                          axis=1).T.astype(np.float32, copy=False)
