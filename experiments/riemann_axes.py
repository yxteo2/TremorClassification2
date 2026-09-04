"""Riemannian tangent-space features: the inter-axis structure the pipeline throws away.

`method_table` computes a power spectrum **per gyroscope axis and then averages
the axes**. Everything about how the three axes relate — the direction the limb
oscillates in — is discarded before any model sees it.

## Why the closed rows do not cover this

`spectral_representation.md` closed the two obvious repairs and gave the reason:

    principal eigenvalue lambda_1 of S(f)   macroP -0.000
    polarisation lambda_1/trace            macroP -0.020

> an SNR improvement that lives in absolute amplitude is invisible to a
> scale-invariant pipeline

Both are **rotation-invariant scalars**: they collapse the 3x3 cross-spectral
matrix to one number per frequency, keeping the oscillation's *strength* and
discarding its *orientation*. The untested object is the full covariance matrix,
whose Riemannian tangent-space vector (Barachant et al., *IEEE TBME* 59(4) 2012
— the canonical small-n method in EEG brain-computer interfaces) is 6 numbers
for a 3x3 and is **scale-free once the matrix is trace-normalised**. It
therefore survives the sum-normalisation that deleted lambda_1's gain, and it
satisfies the repaired time-domain rule in `closed_families.md`: *few features,
selected for classification*.

It is also close to orthogonal to the spectrum. One band-limited covariance has
no frequency content at all, so it cannot be re-reading peak location or
bandwidth.

The physiology: PD rest tremor is classically pronation-supination, a rotation
about the forearm's long axis; ET postural tremor is predominantly
flexion-extension, about a roughly perpendicular axis.

## The model-free diagnostic, run first (invariant 10)

`_axis_orientation_diagnostic.py`, no model and no deep fits:

    anisotropy lambda_1/trace      0.65-0.81   (1/3 would be isotropic)
    axis reliability, 2015         15.8 deg within patient vs 40.2 between
    PD-vs-ET AUC, 6 features       2015 0.702 (null p95 0.578) *
                                   PADS 0.713 (null p95 0.604) *

So the tremor has a dominant axis, that axis is reproducible within a patient
far better than between patients, and six numbers describing it separate PD
from ET beyond a permutation null in both cohorts large enough to test. The
information exists. Whether it composes to the 3-class model is what this
measures.

## Arms

    reported                 must reproduce build() bit-exactly
    + tangent (6)            the 6 features appended to the descriptor stream
    + tangent SHUFFLED       attribution control: the same six features
                             permuted across patients WITHIN cohort, redrawn
                             per split. Anything this arm reproduces is
                             dimensionality, not orientation.
    tangent alone            how much the six features hold on their own

The shuffle is redrawn every split because a single fixed draw was itself
class-associated at p = 0.051 once in this project and manufactured a
precET gain of +0.090.

## Predictions, recorded before the run

1. **`tangent alone` beats its own permutation null but loses to the reported
   model.** Six band-limited numbers cannot match a 16-bin spectrum.
2. **The fusion arm's gain, if any, is larger on precPD and precET than on
   precN**, because the diagnostic separated PD from ET (AUC 0.702 / 0.713)
   better than it separated N from tremor on PADS (0.579). This is the claim
   worth checking — it is what makes the mechanism testable rather than just
   the adoption question.
3. **Direction on macroP: leaning positive, small, and genuinely uncertain.**
   The case against is standing rule #5 — descriptor-level gains have failed to
   compose three times — and the fact that 16 feature unions here have
   underperformed their best member. The case for is that this is a
   *measurement-derived* prediction, the category that has held ten times out of
   ten, and the feature is orthogonal to the spectrum rather than a restatement
   of it.

A live alternative account, stated in advance so it is not invented afterwards:
on PADS the two recordings are LeftWrist and RightWrist, and their axes differ
as much as two random patients' do (39.5 deg vs 38.3). Averaging them therefore
shrinks toward isotropy by an amount that depends on **bilateral symmetry** — a
known PD-vs-ET contrast. The reported model already carries `ASYM` features, so
any gain here is over and above them, but "it is really asymmetry" would remain
the explanation to beat.

20 splits, paired, checkpointed. Run: ``python -m experiments.riemann_axes``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.linalg import logm
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pooling_rules import fit_members

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
FS = 100.0
BAND = (3.0, 15.0)
ARMS = ("reported", "+ tangent (6)", "+ tangent SHUFFLED", "tangent alone")


def score(pt, off, yte):
    """recET and nETpred are carried so a degenerate one-prediction precision
    (MiniRocket once read precET 1.000 off a single ET prediction) is visible."""
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2], float((pred == 2).sum())]


def band_cov(x):
    """3x3 covariance of the 3-15 Hz band-passed 3-axis angular velocity."""
    sos = butter(4, [BAND[0] / (FS / 2), BAND[1] / (FS / 2)], btype="band",
                 output="sos")
    xb = sosfiltfilt(sos, np.asarray(x, float), axis=-1)
    xb = xb - xb.mean(-1, keepdims=True)
    return (xb @ xb.T) / xb.shape[-1]


def tangent(C):
    """Log-Euclidean tangent vector of the trace-normalised covariance.

    Trace normalisation is what makes this scale-free, and therefore what makes
    it survive the per-patient sum-normalisation that deleted the lambda_1 gain.
    The off-diagonals carry sqrt(2) so the vector's Euclidean norm equals the
    matrix's Frobenius norm.
    """
    C = C / (np.trace(C) + 1e-20)
    L = np.real(logm(C + 1e-9 * np.eye(3)))
    iu = np.triu_indices(3)
    return L[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def tangent_table(recs, ch):
    """Per-patient mean tangent vector, patients sorted by subject id."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        rows[r.subject].append(tangent(band_cov(x)))
        lab[r.subject] = r.y
    p = sorted(rows)
    return np.array([np.mean(rows[k], 0) for k in p]), np.array([lab[k]
                                                                 for k in p])


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC, TR = d["SPEC"]["multitaper"], d["TRAJ"]   # the reported spectrum
    A = np.hstack([d["ASYM"], d["HAVE"]])
    D = np.hstack([d["DESC"], A])

    (rA, rB, rC), keep = load_cohorts()
    TAN = np.vstack([tangent_table(rA, slice(3, 6))[0],
                     tangent_table(rB, slice(3, 6))[0],
                     tangent_table(rC, slice(0, 3))[0][keep]])
    assert len(TAN) == len(y), f"{len(TAN)} tangent rows vs {len(y)} patients"
    yc = tangent_table(rC, slice(0, 3))[1][keep]
    assert (yc == y[len(y) - len(keep):]).all(), "PADS row order does not match"

    coh = np.array([k.split("_")[0] for k in key])
    print(f"n={len(y)}  tangent dims={TAN.shape[1]}  splits={SPLITS}")
    print("predictions on record: (1) tangent alone loses to reported;")
    print("(2) any fusion gain is larger on precPD/precET than precN;")
    print("(3) macroP leaning positive, small, uncertain\n", flush=True)

    res, done = resume_load("riemann_axes", ARMS)
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        # per-split redraw, permuted WITHIN cohort so the control keeps the
        # cohort structure and destroys only the patient-to-feature link
        rng = np.random.default_rng(1000 + sp)
        SH = TAN.copy()
        for c in np.unique(coh):
            i = np.flatnonzero(coh == c)
            SH[i] = TAN[rng.permutation(i)]

        for arm in ARMS:
            if arm == "reported":
                desc = D
            elif arm == "+ tangent (6)":
                desc = np.hstack([D, TAN])
            elif arm == "+ tangent SHUFFLED":
                desc = np.hstack([D, SH])
            else:
                desc = np.hstack([TAN, A])
            V, T = fit_members(SPEC, desc, TR, y, tr, va, te)
            res[arm].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                  y[te]))
        resume_save("riemann_axes", res, sp)
        print(f"  split {sp + 1}/{SPLITS} done", flush=True)

    R = {a: np.array(res[a]) for a in ARMS}
    print(f"\n{'arm':>22}" + "".join(f"{n:>9}" for n in NM))
    for a in ARMS:
        print(f"{a:>22}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))

    for base in ("reported", "+ tangent SHUFFLED"):
        print(f"\npaired vs {base!r}:")
        for a in ARMS:
            if a == base:
                continue
            print(f"  {a}:")
            for (dd, lo, hi), c in zip(paired(R[a], R[base]), NM):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nPREDICTION 2 -- gain larger on precPD/precET than precN:")
    g = R["+ tangent (6)"].mean(0) - R["reported"].mean(0)
    print("    precN {:+.3f}   precPD {:+.3f}   precET {:+.3f}".format(*g[:3]))

    print("\nsplit-level win rate vs the reported model:")
    for a in ARMS[1:]:
        print(f"  {a:>22}: " + "  ".join(
            f"{c} {float((R[a][:, i] > R['reported'][:, i]).mean()):.2f}"
            for i, c in enumerate(NM[:5])))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
