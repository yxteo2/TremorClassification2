"""Does the 2015 REST PD-vs-ET signal replicate outside 2015?

`01_tremor_characteristics.ipynb` found PD vs ET separable at REST on 2015 (six
frequency characteristics, AUC ~0.65-0.70, above a repeated-CV shuffled-label
null on all three sensors, 16 ET), with PD slower than ET, and the reverse
ordering in PADS. `inhouse_rest.md` then showed the deep model cannot use it at
this n. For the paper the question is simpler: **is it a property of in-house
REST, or of the 2015 cohort?** A single cohort at 16 ET is a lead, not a result.

## Tests, fixed in advance

Features: the six characteristics of `frequency.characteristics`, on the
**lower-arm sensor** -- the pipeline's sensor, fixed before looking, not the hand
sensor that happened to score best in the notebook.

    A  2015 REST, 5-fold CV x 20 repeats, 500-permutation null   (re-statement)
    B  in-house pooled REST (2015 + NewData), features z-scored within cohort,
       same CV and null                                           (21 ET)
    C  fit on all of 2015 REST, apply unchanged to NewData REST   (6 ET)
    D  fit on all of 2015 REST, apply unchanged to PADS Relaxed   (28 ET)
    E  zero-parameter: max_freq alone, "ET higher", on NewData and PADS Relaxed

C, D and E have no CV: the model or the rule is frozen before it sees the test
cohort, and the p-value is the exact one-sided Mann-Whitney test of the AUC.

## What was already seen -- this is NOT a blind replication

The follow-up audit printed NewData REST medians (max_freq PD 5.27, ET 5.52;
univariate AUCs 0.51-0.60) and PADS Relaxed (ET slower, AUC 0.30). E is therefore
known before running; it is included so C and D can be read against it. C (the
multivariate model's transfer) had not been computed.

## Prediction, recorded before the run

**C is above 0.5 but not significant** -- 6 ET cannot confirm an effect of this
size (a true AUC of 0.65 needs roughly 15+ ET per side for p < 0.05 one-sided).
**D is significantly below 0.5** -- the frozen 2015 rule is reversed in PADS.
**B stays above its null** but is weaker than A, because NewData adds 6 ET with
a smaller effect.

Run: ``python -m experiments.rest_replication``
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from frequency.characteristics import FEATURES, patient_table

REPEATS, NPERM = 20, 500


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def pd_et(X, y):
    m = y != 0
    return X[m], (y[m] == 2).astype(int)


def cv_auc(X, y, seed0=0):
    return float(np.mean([roc_auc_score(y, cross_val_predict(
        model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed0 + r),
        method="predict_proba")[:, 1]) for r in range(REPEATS)]))


def cv_with_null(tag, X, y):
    real = cv_auc(X, y)
    rng = np.random.default_rng(0)
    null = np.array([cv_auc(X, rng.permutation(y), 1000 + i * REPEATS)
                     for i in range(NPERM)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    p = (1 + np.sum(null >= real)) / (1 + NPERM)
    print(f"  {tag:<44} n={len(y):>3} ET={y.sum():>2}  AUC {real:.3f}   "
          f"null 95% [{lo:.3f}, {hi:.3f}]   p={p:.3f}")
    return real, p


def frozen(tag, score, y):
    auc = roc_auc_score(y, score)
    p_hi = mannwhitneyu(score[y == 1], score[y == 0], alternative="greater").pvalue
    p_lo = mannwhitneyu(score[y == 1], score[y == 0], alternative="less").pvalue
    print(f"  {tag:<44} n={len(y):>3} ET={y.sum():>2}  AUC {auc:.3f}   "
          f"p(ET higher)={p_hi:.3f}   p(ET lower)={p_lo:.3f}")
    return auc, p_hi, p_lo


def main():
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    ch = slice(3, 6)
    X15, y15, _ = patient_table(load_quaternion_recordings(
        "Data", action="REST", mode="angular_velocity"), ch=ch)
    Xnd, ynd, _ = patient_table(load_2025_all(conditions=("REST",)), ch=ch)
    Xpa, ypa, _ = patient_table(load_pads_extracted("pads_relaxed"),
                                ch=slice(0, 3))
    A = pd_et(X15, y15)
    N = pd_et(Xnd, ynd)
    P = pd_et(Xpa, ypa)
    print(f"REST PD vs ET, six characteristics {FEATURES}, lower-arm sensor\n")

    print("cross-validated, with shuffled-label null")
    cv_with_null("A  2015 REST", *A)
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-12)
    cv_with_null("B  in-house pooled (z-scored within cohort)",
                 np.vstack([z(A[0]), z(N[0])]), np.concatenate([A[1], N[1]]))

    print("\nfrozen on 2015 REST, applied unchanged")
    m = model().fit(*A)
    frozen("C  -> NewData REST", m.predict_proba(N[0])[:, 1], N[1])
    frozen("D  -> PADS Relaxed", m.predict_proba(P[0])[:, 1], P[1])

    print("\nzero-parameter rule: max_freq, ET higher")
    j = FEATURES.index("max_freq")
    frozen("E  2015 REST (where the rule came from)", A[0][:, j], A[1])
    frozen("E  NewData REST", N[0][:, j], N[1])
    frozen("E  PADS Relaxed", P[0][:, j], P[1])

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
