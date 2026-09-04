"""Does the *orientation* of the tremor carry class information? Measured without a model.

Supporting evidence for a Riemannian tangent-space arm, run before any deep
fits (invariant 10). It asks whether the thing that method would add actually
exists in this data.

## The gap, and why the closed rows do not cover it

`method_table` computes a power spectrum **per gyroscope axis and then averages
the axes** (`_per_freq_mean`). Everything about how the three axes relate — the
direction the limb oscillates in — is discarded before the model sees anything.

`spectral_representation.md` closed the two obvious repairs, and its reason
matters here:

  principal eigenvalue lambda_1 of S(f)   macroP -0.000
  polarisation lambda_1/trace            macroP -0.020

> an SNR improvement that lives in absolute amplitude is invisible to a
> scale-invariant pipeline

Both of those are **rotation-invariant scalars**: they collapse the 3x3
cross-spectral matrix to one number per frequency and therefore throw away the
orientation of the oscillation, keeping only its strength. The untested object
is the **full covariance matrix**, whose Riemannian tangent-space vector
(Barachant et al., *IEEE TBME* 59(4) 2012) is 6 numbers for a 3x3 and is
**scale-free once the matrix is trace-normalised** — so it survives the
sum-normalisation that deleted lambda_1's gain.

The physiological reason to care: PD rest tremor is classically
pronation-supination, a rotation about the forearm's long axis, while ET
postural tremor is predominantly flexion-extension, about an axis roughly
perpendicular to it. If that holds in these recordings and the sensor is mounted
consistently, **the axis of rotation separates PD from ET and the current
representation cannot see it.**

## What would have to be true

That story needs four things, and each is cheap to measure:

  1. the tremor is anisotropic  -- otherwise there is no dominant axis at all
  2. its dominant axis is reproducible within a patient, better than between
     patients -- otherwise the orientation is noise
  3. the orientation differs by class *within a cohort* -- the between-cohort
     comparison is confounded by mounting (PADS wrist wearable vs the 2015 /
     NewData lower-arm strap), so it is measured within cohort only
  4. a 6-feature tangent-space vector beats chance on PD-vs-ET under a
     permutation null

Failing 1 or 2 kills the method outright. Failing 3 and 4 means the covariance
carries no more than the closed lambda_1 arm did, and the fits should not be
spent.

Uses log-Euclidean tangent space -- ``logm`` of the trace-normalised covariance,
upper triangle, 6 features -- rather than the AIRM tangent space at a Frechet
mean. The two agree closely for well-conditioned 3x3 matrices and the
log-Euclidean version has no reference point to fit, so nothing leaks.

Run: ``python -m experiments._axis_orientation_diagnostic``
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy.linalg import logm
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.estimator_smoothing import load_cohorts

warnings.filterwarnings("ignore")

FS = 100.0
BAND = (3.0, 15.0)
CLS = ("N", "PD", "ET")
N_PERM = 2000
SEED = 0


def _band_cov(x):
    """3x3 covariance of the 3-15 Hz band-passed 3-axis angular velocity."""
    sos = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype="band",
                 output="sos")
    xb = sosfiltfilt(sos, np.asarray(x, float), axis=-1)
    xb = xb - xb.mean(-1, keepdims=True)
    return (xb @ xb.T) / xb.shape[-1]


def _tangent(C):
    """Log-Euclidean tangent vector of the trace-normalised covariance."""
    C = C / (np.trace(C) + 1e-20)
    L = np.real(logm(C + 1e-9 * np.eye(3)))
    iu = np.triu_indices(3)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[iu] * w


def _recordings(rs, ch):
    """(subject, label, covariance) per recording."""
    out = []
    for r in rs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        out.append((r.subject, int(r.y), _band_cov(x)))
    return out


def _axis_angle(u, v):
    """Angle in degrees between two sign-ambiguous axes (0 = same axis)."""
    c = abs(float(np.dot(u, v))) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-20)
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def _perm_auc(X, y, rng, groups=None):
    """5-fold CV AUC of a 6-feature logistic regression, and its permutation p."""
    def auc(yy):
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        s = np.zeros(len(yy))
        for tr, te in cv.split(X, yy):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"))
            m.fit(X[tr], yy[tr])
            s[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(yy, s)

    a = auc(y)
    null = np.array([auc(rng.permutation(y)) for _ in range(N_PERM // 100)])
    return a, float(np.mean(null >= a)), float(np.quantile(null, 0.95))


def main():
    rng = np.random.default_rng(SEED)
    (rA, rB, rC), keep = load_cohorts()
    cohorts = {
        "2015": _recordings(rA, slice(3, 6)),
        "NewData": _recordings(rB, slice(3, 6)),
        "PADS": _recordings(rC, slice(0, 3)),
    }

    # ---- 1. anisotropy: is there a dominant axis at all? -------------------
    print("1. ANISOTROPY  lambda_1 / trace of the 3-15 Hz covariance")
    print("   (1/3 = isotropic, 1.0 = a perfectly linear oscillation)\n")
    print(f"   {'cohort':<9}{'n rec':>7}" + "".join(f"{c:>9}" for c in CLS))
    for name, recs in cohorts.items():
        by = defaultdict(list)
        for _, y, C in recs:
            w = np.linalg.eigvalsh(C)
            by[y].append(w[-1] / (w.sum() + 1e-20))
        print(f"   {name:<9}{len(recs):>7}"
              + "".join(f"{np.mean(by[c]):>9.3f}" if by[c] else f"{'-':>9}"
                        for c in range(3)))

    # ---- 2. reproducibility: same patient vs different patients -----------
    # A patient's two recordings mean different things per cohort: 2015 and
    # NewData repeat the SAME arm, so their within-patient angle is a genuine
    # test-retest reliability. PADS's pair is LeftWrist and RightWrist -- a
    # mirrored mounting on a different limb -- so its number is a bilateral
    # comparison and NOT a reliability measure. Labelled accordingly.
    print("\n2. REPRODUCIBILITY of the dominant axis, degrees between a")
    print("   patient's two recordings (0 = identical axis, 90 = orthogonal,")
    print("   ~57 = uniformly random pair of axes)\n")
    print(f"   {'cohort':<9}{'pair is':<14}{'within patient':>16}"
          f"{'between patients':>18}{'n pairs':>9}")
    kind = {"2015": "repeat trial", "NewData": "repeat trial",
            "PADS": "L vs R wrist"}
    for name, recs in cohorts.items():
        v1 = [(s, np.linalg.eigh(C)[1][:, -1]) for s, _, C in recs]
        bysub = defaultdict(list)
        for s, v in v1:
            bysub[s].append(v)
        within = [_axis_angle(vs[i], vs[j]) for vs in bysub.values()
                  for i in range(len(vs)) for j in range(i + 1, len(vs))]
        idx = rng.integers(0, len(v1), (2000, 2))
        between = [_axis_angle(v1[i][1], v1[j][1]) for i, j in idx
                   if v1[i][0] != v1[j][0]]
        w = f"{np.mean(within):>16.1f}" if within else f"{'-':>16}"
        print(f"   {name:<9}{kind[name]:<14}{w}{np.mean(between):>18.1f}"
              f"{len(within):>9}")
    print("\n   Only the 'repeat trial' rows test reliability. PADS's row says")
    print("   how alike the two WRISTS are, which is a bilateral-symmetry")
    print("   measurement -- a known PD-vs-ET contrast in its own right.")

    # ---- 3 & 4. class information in the orientation, within cohort -------
    print("\n3. CLASS INFORMATION in the 6-feature tangent vector, PATIENT level,")
    print("   within cohort (mounting differs between cohorts, so pooling would")
    print("   confound orientation with cohort). AUC with a permutation null.\n")
    print(f"   {'cohort':<9}{'contrast':<12}{'n':>5}{'AUC':>8}{'null p95':>10}"
          f"{'p':>8}")
    for name, recs in cohorts.items():
        bysub, lab = defaultdict(list), {}
        for s, y, C in recs:
            bysub[s].append(_tangent(C))
            lab[s] = y
        subs = sorted(bysub)
        X = np.array([np.mean(bysub[s], 0) for s in subs])
        y = np.array([lab[s] for s in subs])
        for tag, pos, neg in (("PD vs ET", 2, 1), ("N vs tremor", 1, 0)):
            if tag == "PD vs ET":
                m = np.isin(y, (1, 2))
                yy = (y[m] == 2).astype(int)
            else:
                m = np.ones(len(y), bool)
                yy = (y > 0).astype(int)
            if yy.sum() < 5 or (1 - yy).sum() < 5:
                print(f"   {name:<9}{tag:<12}{m.sum():>5}   too few")
                continue
            a, p, q95 = _perm_auc(X[m], yy, rng)
            star = " *" if p < 0.05 else ""
            print(f"   {name:<9}{tag:<12}{m.sum():>5}{a:>8.3f}{q95:>10.3f}"
                  f"{p:>8.3f}{star}")

    print("\nRead: anisotropy near 1/3 kills the method (no dominant axis);")
    print("within-patient angle no better than between-patient kills it (the")
    print("orientation is noise); AUC inside the permutation null means the")
    print("covariance carries no more than the closed lambda_1 arm did.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
