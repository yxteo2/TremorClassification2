"""How many ET patients would it take to measure the effects we keep chasing?

Every negative in this project is reported against a detection floor set by
**49 ET patients**, ~10 of which land in a test fold. This computes that floor
as a function of cohort size, so a data-collection plan can be written against
numbers rather than intuition. No model is fitted; these are properties of the
evaluation protocol alone.

Three quantities:

1. **The PD-vs-ET permutation null.** With random scores, how high does AUC
   reach by chance at a given ET count? `permutation_null.md` reports the 95th
   percentile at 0.655 for 21 in-house ET. Anything below that is not
   distinguishable from chance, which is why in-house PD-vs-ET is unmeasurable
   today.

2. **precET granularity.** Precision is a ratio over the ET predictions actually
   made. With ~9 of them, precision can only take ~10 distinct values, so it
   moves in jumps of ~0.11 and no amount of averaging over splits makes a
   smaller effect visible in a single fold.

3. **Minimum detectable paired difference** on precET, from this project's own
   measured per-split sd (0.178), at 20 and 40 splits — and what that sd would
   become with more ET patients, since precision's variance falls roughly as
   1/n_ET.

Run: ``python -m experiments._sample_size_planning``
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
N_PERM = 4000
TEST_FRAC = 0.20          # common/protocol.py (verified, not assumed)
SD_PRECET_NOW = 0.178     # measured, window_training.py / riemann_axes.py
N_ET_NOW = 49


def auc_null_p95(n_pos, n_neg, n=N_PERM):
    """95th percentile of AUC under random scores — the chance ceiling."""
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
    return float(np.quantile([roc_auc_score(y, RNG.permutation(len(y)))
                              for _ in range(n)], 0.95))


def main():
    print("1. PD-vs-ET PERMUTATION NULL — the chance ceiling on AUC")
    print("   (an effect must clear p95 to be distinguishable from nothing)\n")
    print(f"   {'ET total':>9}{'ET in test':>12}{'PD in test':>12}"
          f"{'null p95':>10}{'verdict vs AUC 0.71':>22}")
    for n_et in (21, 49, 75, 100, 150, 200, 300):
        et_te = max(int(round(n_et * TEST_FRAC)), 2)
        pd_te = max(int(round(188 * TEST_FRAC)), 2)   # PD total in the merge
        p95 = auc_null_p95(et_te, pd_te)
        v = "detectable" if p95 < 0.71 else "INSIDE THE NULL"
        print(f"   {n_et:>9}{et_te:>12}{pd_te:>12}{p95:>10.3f}{v:>22}")

    print("\n2. precET GRANULARITY — precision is a ratio over ET predictions")
    print("   made, so few predictions means a coarse instrument\n")
    print(f"   {'ET total':>9}{'ET predicted/fold':>19}{'step size':>11}")
    for n_et in (21, 49, 75, 100, 150, 200, 300):
        npred = max(n_et * TEST_FRAC * 0.75, 1)     # ~0.75 recall, measured
        print(f"   {n_et:>9}{npred:>19.1f}{1.0 / npred:>11.3f}")

    print("\n3. MINIMUM DETECTABLE PAIRED DIFFERENCE on precET")
    print("   sd scales as sqrt(n_ET_now / n_ET); CI half-width = 1.96*sd/sqrt(splits)\n")
    print(f"   {'ET total':>9}{'sd(precET)':>12}{'20 splits':>12}{'40 splits':>12}"
          f"{'80 splits':>12}")
    for n_et in (21, 49, 75, 100, 150, 200, 300):
        sd = SD_PRECET_NOW * np.sqrt(N_ET_NOW / n_et)
        row = "".join(f"{1.96 * sd / np.sqrt(s):>12.3f}" for s in (20, 40, 80))
        print(f"   {n_et:>9}{sd:>12.3f}{row}")

    print("\n   The reported headline gain is precET +0.104. The largest")
    print("   *candidate* improvements measured since are +0.02 to +0.04 —")
    print("   below what 40 splits resolves at 49 ET (0.078). Doubling ET to")
    print("   ~100 buys more resolution than quadrupling the splits, and does")
    print("   it by adding information rather than re-cutting the same data.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
