"""Compare quaternion-aware representations against the scalar TF baseline.

Question this answers: does keeping the *orientation / cross-axis phase*
structure of the quaternion stream -- rather than reducing each sensor to a
scalar spectrum -- buy anything on the PD-vs-ET axis, which we have shown is
NOT separable along frequency alone?

Representations compared (all evaluated with the same patient-level LOSO,
tuned ET threshold and subject bootstrap CI as everything else in the repo):

  ``omega``      angular velocity, scalar TF summary       (the current baseline)
  ``logmap``     so(3) rotation vector, scalar TF summary   (Lie-algebra route)
  ``polar``      orbit geometry: circularity / planarity    (NEW, cross-axis phase)
  ``qstft``      hypercomplex simplex/perplex + chirality   (NEW, axis-dependent)
  ``omega+polar``  baseline plus orbit geometry             (the interesting one)

The polarization features are rotation-invariant by construction, so unlike a
raw log map they cannot encode how the sensor happened to be strapped on.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tremor.quaternion_data import load_quaternion_recordings
from pdetn.quaternion_tf import (
    GRAV_CHIRALITY_FEATURE_NAMES,
    POLARIZATION_FEATURE_NAMES,
    QSTFT_FEATURE_NAMES,
    gravity_chirality_features,
    polarization_features,
    qstft_features,
)
from pdetn.spatial_features import SENSORS

FS = 100.0


# --------------------------------------------------------------------------- #
# Per-recording feature blocks
# --------------------------------------------------------------------------- #
def polar_features(x, fs=FS, sensors=None, **kw):
    """Orbit-geometry features for every sensor of a ``(9, T)`` recording."""
    out = {}
    for name, ch in (sensors or SENSORS).items():
        for k, v in polarization_features(x[list(ch)], fs=fs, **kw).items():
            out[f"{name}_{k}"] = v
    return out


def qstft_feature_block(x, fs=FS, sensors=None, **kw):
    out = {}
    for name, ch in (sensors or SENSORS).items():
        for k, v in qstft_features(x[list(ch)], fs=fs, **kw).items():
            out[f"{name}_{k}"] = v
    return out


def gchir_features(x, g, fs=FS, sensors=None, **kw):
    """Mount-invariant signed handedness per sensor.

    ``x`` is the ``(9, T-2)`` angular velocity, ``g`` the ``(9, T)`` body-frame
    gravity for the same recording; gravity is trimmed to match the
    central-difference omega.
    """
    g = g[:, 1:-1] if g.shape[1] == x.shape[1] + 2 else g[:, :x.shape[1]]
    out = {}
    for name, ch in (sensors or SENSORS).items():
        idx = list(ch)
        for k, v in gravity_chirality_features(x[idx], g[idx], fs=fs, **kw).items():
            out[f"{name}_{k}"] = v
    return out


def gchir_feature_names(sensors=None):
    return [f"{s}_{f}" for s in (sensors or SENSORS)
            for f in GRAV_CHIRALITY_FEATURE_NAMES]


def patient_gchir_table(omega_recs, grav_recs, sensors=None, **kw):
    """(X, y, patients, cols) for the gravity-referenced handedness block."""
    by_path = {r.path: r for r in grav_recs}
    cols = gchir_feature_names(sensors)
    per, lab = defaultdict(list), {}
    for r in omega_recs:
        g = by_path.get(r.path)
        if g is None:
            raise KeyError(f"no gravity recording matching {r.path}")
        d = gchir_features(r.x, g.x, sensors=sensors, **kw)
        per[r.subject].append([d[c] for c in cols])
        lab[r.subject] = r.y
    X, y, pats = _aggregate(per, lab)
    return X, y, pats, cols


def polar_feature_names(sensors=None):
    return [f"{s}_{f}" for s in (sensors or SENSORS)
            for f in POLARIZATION_FEATURE_NAMES]


def qstft_feature_names(sensors=None):
    return [f"{s}_{f}" for s in (sensors or SENSORS)
            for f in QSTFT_FEATURE_NAMES]


# --------------------------------------------------------------------------- #
# Patient-level tables
# --------------------------------------------------------------------------- #
def _aggregate(per, lab):
    pats = sorted(per)
    X = np.array([np.mean(per[p], axis=0) for p in pats], dtype=np.float64)
    return np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats)


def patient_table(recs, block="polar", sensors=None, **kw):
    """(X, y, patients) for one feature block, averaged over each patient's
    recordings."""
    fn, names = ((polar_features, polar_feature_names)
                 if block == "polar" else (qstft_feature_block, qstft_feature_names))
    cols = names(sensors)
    per, lab = defaultdict(list), {}
    for r in recs:
        d = fn(r.x, sensors=sensors, **kw)
        per[r.subject].append([d[c] for c in cols])
        lab[r.subject] = r.y
    X, y, pats = _aggregate(per, lab)
    return X, y, pats, cols


def load_repr(data_root="Data", action="OUT", mode="angular_velocity", **kw):
    """Recordings under a given quaternion->signal mode ('angular_velocity',
    'log_map', 'gravity', 'log_map_gravity')."""
    return load_quaternion_recordings(data_root, action=action, mode=mode, **kw)


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def align(*tables):
    """Column-stack feature tables that share a patient index."""
    pats = tables[0][2]
    for t in tables[1:]:
        if not (t[2] == pats).all():
            raise ValueError("patient indices differ between feature tables")
    return np.hstack([t[0] for t in tables]), tables[0][1], pats


def compare(data_root="Data", action="OUT", n_boot=1000, n_perm=1000,
            stft_kw=None, verbose=True):
    """Run every representation through the same two-stage LOSO evaluation."""
    from pdetn.model import TwoStageClassifier
    from pdetn.evaluate import evaluate
    from pdetn.separability import patient_decomp_features

    stft_kw = stft_kw or dict(nperseg=256, nfft=256, noverlap=192)

    omega = load_repr(data_root, action, mode="angular_velocity")
    logmap = load_repr(data_root, action, mode="log_map")
    grav = load_repr(data_root, action, mode="gravity")

    tf_omega = patient_decomp_features(omega, "stft", **stft_kw)
    tf_logmap = patient_decomp_features(logmap, "stft", **stft_kw)
    pol = patient_table(omega, block="polar")
    qst = patient_table(omega, block="qstft")
    gch = patient_gchir_table(omega, grav)
    base = (tf_omega[0], tf_omega[1], tf_omega[2])

    sets = {
        "omega_stft (baseline)": base,
        "logmap_stft": (tf_logmap[0], tf_logmap[1], tf_logmap[2]),
        "polarization": (pol[0], pol[1], pol[2]),
        "qstft": (qst[0], qst[1], qst[2]),
        "grav_chirality": (gch[0], gch[1], gch[2]),
        "polar + gchir": align(pol, gch),
        "omega_stft + polarization": align(base, pol),
        "omega_stft + gchir": align(base, gch),
        "omega_stft + polar + qstft": align(base, pol, qst),
        "omega_stft + polar + gchir": align(base, pol, gch),
    }

    results = {}
    for name, (X, y, pats) in sets.items():
        r = evaluate(
            lambda: TwoStageClassifier("logreg", "logreg", tune_et_threshold=True),
            X, y, pats, n_boot=n_boot, n_perm=n_perm)
        results[name] = r
        if verbose:
            ci = r["ci"]["ET"]
            print(f"{name:>28}  n_feat {X.shape[1]:>4}  macroF1 {r['macro_f1']:.3f}  "
                  f"ET-F1 {r['per_class_f1']['ET']:.3f} "
                  f"[{ci['lo']:.2f},{ci['hi']:.2f}]  "
                  f"N-vs-T {r['n_vs_tremor_acc']:.3f}  "
                  f"PD-vs-ET {r['pd_vs_et_acc']:.3f}  p={r['permutation_p']:.4f}")
    return results


def univariate_screen(data_root="Data", action="OUT", top=15):
    """Which individual orbit-geometry features separate PD from ET?

    Mann-Whitney with rank-biserial effect size, PD vs ET only (the hard axis).
    Screening, not selection -- these p-values are uncorrected and the features
    are NOT chosen from this for the classifier.
    """
    from scipy.stats import mannwhitneyu

    recs = load_repr(data_root, action, mode="angular_velocity")
    grav = load_repr(data_root, action, mode="gravity")
    tables = [patient_table(recs, block=b) for b in ("polar", "qstft")]
    tables.append(patient_gchir_table(recs, grav))
    rows = []
    for X, y, _, cols in tables:
        pd_m, et_m = y == 1, y == 2
        for j, c in enumerate(cols):
            a, b = X[pd_m, j], X[et_m, j]
            if len(a) < 3 or len(b) < 3 or np.allclose(a.std() + b.std(), 0):
                continue
            u, p = mannwhitneyu(a, b, alternative="two-sided")
            eff = 2.0 * u / (len(a) * len(b)) - 1.0      # rank-biserial
            rows.append((c, eff, p, float(np.median(a)), float(np.median(b))))
    rows.sort(key=lambda r: r[2])
    print(f"{'feature':>26}{'effect':>9}{'p':>10}{'PD med':>10}{'ET med':>10}")
    for c, eff, p, ma, mb in rows[:top]:
        print(f"{c:>26}{eff:>9.3f}{p:>10.4f}{ma:>10.3f}{mb:>10.3f}")
    return rows
