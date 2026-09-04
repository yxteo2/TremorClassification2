"""Euclidean Alignment at two levels — measured before any fits are spent.

**Euclidean Alignment** (He & Wu, *IEEE TBME* 67(2) 2020) is the standard
unsupervised domain adaptation in EEG brain-computer interfaces. For each
subject it takes the arithmetic mean of their trials' spatial covariances and
whitens by its inverse square root:

    R_bar = mean_i (X_i X_i^T / n)        X_i' = R_bar^(-1/2) X_i

Every subject's mean spatial covariance then equals the identity, which removes
per-subject sensor gain, mounting and impedance differences. It is cheap,
label-free, and it is the closest published method to the one item
`SKILL.md` lists as genuinely open — *feature-level cohort harmonisation, fitted
on train only*.

## Why the level matters, and why this runs before the fits

EA works in BCI because **each subject supplies trials of every class**. The
subject's mean covariance is then a pure subject effect, and whitening by it
leaves the between-class differences untouched.

**Here each patient has exactly one label.** A patient's mean covariance is
therefore their class signature as much as their subject signature, and
whitening by it should remove exactly what
`_axis_orientation_diagnostic.py` just measured to be worth PD-vs-ET AUC
0.702 / 0.713. That is the PCEN failure again in a different costume: dividing
a unit by its own average destroys what varies *across* the units being
classified.

**Cohort-level EA is the version that survives that argument**: whitening by the
cohort mean removes the mounting and scaling differences between 2015, NewData
and PADS while leaving each patient's deviation from their cohort intact.

Both are measured here at zero model cost, on the same 6-feature tangent vector
and the same permutation null as the orientation diagnostic.

## Prediction, recorded before the run

**Patient-level EA collapses PD-vs-ET AUC to chance; cohort-level EA does not.**
If patient-level EA instead preserves the AUC, the argument above is wrong and
the method deserves a real arm.

Run: ``python -m experiments._euclidean_alignment_diagnostic``
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy.linalg import fractional_matrix_power

from experiments._axis_orientation_diagnostic import _perm_auc
from experiments.estimator_smoothing import load_cohorts
from experiments.riemann_axes import band_cov, tangent

warnings.filterwarnings("ignore")
SEED = 0


def _whiten(R):
    """R^(-1/2), the EA transform for a mean covariance R."""
    return np.real(fractional_matrix_power(R + 1e-12 * np.eye(3), -0.5))


def main():
    rng = np.random.default_rng(SEED)
    (rA, rB, rC), _ = load_cohorts()
    cohorts = {"2015": (rA, slice(3, 6)), "NewData": (rB, slice(3, 6)),
               "PADS": (rC, slice(0, 3))}

    print("PD-vs-ET AUC from the 6-feature tangent vector, patient level,")
    print("within cohort, with a permutation null. Three alignments.\n")
    print(f"{'cohort':<9}{'alignment':<18}{'n':>5}{'AUC':>8}{'null p95':>10}"
          f"{'p':>8}")

    for name, (recs, ch) in cohorts.items():
        # per-recording covariances, grouped by patient
        bysub, lab = defaultdict(list), {}
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            bysub[r.subject].append(band_cov(x))
            lab[r.subject] = int(r.y)
        subs = sorted(bysub)
        y = np.array([lab[s] for s in subs])

        # the three alignments, each applied to the covariance directly:
        # whitening x by W maps C -> W C W^T
        Wc = _whiten(np.mean([C for s in subs for C in bysub[s]], 0))
        arms = {
            "none": {s: bysub[s] for s in subs},
            "patient-level EA": {
                s: [(W := _whiten(np.mean(bysub[s], 0))) @ C @ W.T
                    for C in bysub[s]] for s in subs},
            "cohort-level EA": {
                s: [Wc @ C @ Wc.T for C in bysub[s]] for s in subs},
        }

        m = np.isin(y, (1, 2))
        yy = (y[m] == 2).astype(int)
        for tag, cov in arms.items():
            X = np.array([np.mean([tangent(C) for C in cov[s]], 0)
                          for s in subs])
            if yy.sum() < 5 or (1 - yy).sum() < 5:
                print(f"{name:<9}{tag:<18}{m.sum():>5}   too few")
                continue
            a, p, q95 = _perm_auc(X[m], yy, rng)
            star = " *" if p < 0.05 else ""
            print(f"{name:<9}{tag:<18}{m.sum():>5}{a:>8.3f}{q95:>10.3f}"
                  f"{p:>8.3f}{star}")
        print()

    print("Read: patient-level EA falling to the null confirms that whitening a")
    print("patient by their own mean covariance removes their class signature,")
    print("because unlike in BCI each patient here supplies only one class.")
    print("Cohort-level EA holding its AUC is what makes it the version worth")
    print("a real arm.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
