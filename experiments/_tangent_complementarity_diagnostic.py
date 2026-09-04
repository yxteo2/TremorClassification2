"""Is the tangent vector complementary to the spectrum, or a restatement of it?

`_axis_orientation_diagnostic.py` measured that six tangent-space numbers
separate PD from ET beyond a permutation null (AUC 0.702 on 2015, 0.713 on
PADS). That says the information exists. It does **not** say the information is
new — the reported model already consumes ten spectral descriptors, and a
feature that merely re-encodes peak location would add nothing while measuring
well on its own.

This asks the complementarity question directly, at the cost of a logistic
regression rather than 480 deep fits:

    descriptors alone        the 10 the reported model already has
    tangent alone            the 6 new ones
    descriptors + tangent    does the union beat its best member?

Run **after** `riemann_axes.py` rather than before it, deliberately. It is a
sub-component measurement, and standing rule #5 in this project is that
sub-component gains do not compose to the 3-class model — three instances. So
this cannot license the deep arm; it can only sharpen how the deep arm's result
is read:

  * union > best member here, deep arm null  -> rule #5, fourth instance
  * union = best member here, deep arm null  -> the feature was redundant, and
                                               the deep null needs no appeal to
                                               rule #5 at all
  * union > best member here, deep arm positive -> the one case where a
                                               descriptor-level gain composed

Within cohort throughout: mounting differs between cohorts, so pooling would let
the classifier read cohort instead of class. NewData is reported but carries
nothing at n = 29.

Run: ``python -m experiments._tangent_complementarity_diagnostic``
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

from common.cohorts import desc_table
from experiments._axis_orientation_diagnostic import _perm_auc
from experiments.estimator_smoothing import load_cohorts
from experiments.riemann_axes import band_cov, tangent

warnings.filterwarnings("ignore")
SEED = 0


def _tangent_rows(recs, ch):
    """Per-patient mean tangent vector, keyed by subject."""
    rows = defaultdict(list)
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        rows[r.subject].append(tangent(band_cov(x)))
    return {s: np.mean(v, 0) for s, v in rows.items()}


def main():
    rng = np.random.default_rng(SEED)
    (rA, rB, rC), _ = load_cohorts()
    cohorts = {"2015": (rA, slice(3, 6)), "NewData": (rB, slice(3, 6)),
               "PADS": (rC, slice(0, 3))}

    print("AUC from a 5-fold logistic regression, patient level, within cohort,")
    print("with a permutation null. 'union' is descriptors + tangent.\n")
    print(f"{'cohort':<9}{'contrast':<13}{'features':<14}{'n':>5}{'dim':>5}"
          f"{'AUC':>8}{'null p95':>10}{'p':>8}")

    for name, (recs, ch) in cohorts.items():
        # `desc_table` returns one row per patient in sorted(subject) order and
        # `_tangent_rows` keys on subject, so both are put in that same order.
        D = desc_table(recs, ch)
        tan = _tangent_rows(recs, ch)
        subs = sorted(tan)
        assert len(D) == len(subs), f"{len(D)} descriptor rows vs {len(subs)}"
        lab = {r.subject: int(r.y) for r in recs}
        y = np.array([lab[s] for s in subs])
        T = np.array([tan[s] for s in subs])

        for tag, pos_neg in (("PD vs ET", (1, 2)), ("N vs tremor", None)):
            if pos_neg is None:
                m, yy = np.ones(len(y), bool), (y > 0).astype(int)
            else:
                m = np.isin(y, pos_neg)
                yy = (y[m] == 2).astype(int)
            if yy.sum() < 5 or (1 - yy).sum() < 5:
                print(f"{name:<9}{tag:<13}{'-':<14}{int(m.sum()):>5}   too few")
                continue
            best = -1.0
            for fname, X in (("descriptors", D), ("tangent", T),
                             ("union", np.hstack([D, T]))):
                a, p, q95 = _perm_auc(np.asarray(X)[m], yy, rng)
                star = " *" if p < 0.05 else ""
                mark = ""
                if fname == "union":
                    mark = "  <- beats best member" if a > best + 1e-9 else \
                           "  <- no better than its best member"
                best = max(best, a) if fname != "union" else best
                print(f"{name:<9}{tag:<13}{fname:<14}{int(m.sum()):>5}"
                      f"{np.shape(X)[1]:>5}{a:>8.3f}{q95:>10.3f}{p:>8.3f}"
                      f"{star}{mark}")
            print()

    print("Read with riemann_axes.md, not instead of it. A union that beats its")
    print("best member here is a descriptor-level gain, and descriptor-level")
    print("gains have failed to compose to the 3-class model three times.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
