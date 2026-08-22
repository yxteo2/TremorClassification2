"""The short-window spectrum on in-house patients, scored by per-class precision.

`tf_window_length.md` established that a spectrum built as the mean of 0.64 s
STFT frames beats the current multitaper representation on PD-vs-ET — paired,
30 repeats, both model families:

    PADS    logreg AUC +0.034 * , precET +0.033 * ; CNN AUC +0.016 *
    MERGED  logreg AUC +0.030 * , precET +0.030 * ; CNN precET +0.013 *

Both of those cohorts contain PADS. The representation has **never been tested on
in-house patients alone**, which is the cohort the clinic actually has and the one
where every method so far has failed.

Two protocols, because they answer different questions and the second is the only
one that can say whether anything is real:

**1. The in-house 3-class protocol** (`own_data_10et.py`): 10 ET fixed in every
test set, N and PD at the cohorts' natural ratio, so ET prevalence is ~0.10 and
per-class precision is comparable across arms. 20 repeats, paired, the reported
two-stream model with only the spectrum swapped.

**2. In-house PD-vs-ET with a permutation null.** Invariant 6 exists because at
21 ET a bootstrap will call chance results significant: the null there spans
[0.298, 0.655], so **nothing below AUC ≈ 0.66 in-house is distinguishable from
chance**. The current best is 0.629. If the short-window spectrum clears that
floor it would be the first in-house PD-vs-ET result in the project to do so; if
it does not, the precision numbers from protocol 1 must be read as differences
between two models that both sit inside the null.

The honest prior is that it will not clear the floor. Every PADS gain measured
this session has failed to transfer in-house (`pd_vs_et_transfer.md`: descriptors
fall from AUC 0.794 within PADS to 0.519 in-house), and the short-window gain was
measured on cohorts containing PADS.

Run: ``python -m experiments.inhouse_shortwindow``
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.own_data_10et import ET_TEST, REPEATS, build, fit_eval
from experiments.shortwindow_deep import short_window_spectrum
from experiments.tf_variability_screen import NPERM, oof, perm_p, scores
from frequency.tables import spectrum_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(NM))]


def main():
    torch.set_num_threads(1)
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings

    A, B, C = build()
    own_spec_mt = np.vstack([A[0], B[0]])
    own_desc = np.vstack([A[1], B[1]])
    own_traj = np.vstack([A[2], B[2]])
    own_y = np.concatenate([A[3], B[3]])
    n_own = len(own_y)

    # in-house patient order, matching build()'s concatenation of 2015 then NewData
    pA = spectrum_table(load_quaternion_recordings("Data", action="OUT",
                                                   mode="angular_velocity"),
                        ch=slice(3, 6))
    pB = spectrum_table(load_2025_all(conditions=("OUT",)), ch=slice(3, 6))
    order = np.concatenate([pA[2], pB[2]])
    assert np.array_equal(np.concatenate([pA[1], pB[1]]), own_y), \
        "in-house patient order does not match build()"

    print("building the short-window spectrum ...", flush=True)
    own_spec_sw = np.nan_to_num(short_window_spectrum(order))
    assert own_spec_sw.shape == own_spec_mt.shape

    print(f"in-house: n={n_own}  N={int((own_y==0).sum())} "
          f"PD={int((own_y==1).sum())} ET={int((own_y==2).sum())}")

    # ---------------------------------------------------------------- #
    # 1. PD vs ET with a permutation null -- can anything be measured?
    # ---------------------------------------------------------------- #
    m = own_y != 0
    yb = (own_y[m] == 2).astype(int)
    print(f"\n{'='*82}")
    print(f"in-house PD vs ET   n={len(yb)}  ET={int(yb.sum())}  "
          f"prevalence {yb.mean():.3f}   logreg, {NPERM} permutations")
    print(f"{'='*82}")
    print(f"{'representation':>26}{'AUC':>9}{'precPD':>9}{'precET':>9}"
          f"{'macroP':>9}{'null 95%':>20}{'p':>8}")
    for lab, S in (("multitaper 16 (current)", own_spec_mt),
                   ("short-window mean 16", own_spec_sw)):
        X = np.nan_to_num(S[m])
        p = oof(X, yb, 3)
        s = scores(yb, p)
        lo, hi, pv = perm_p(X, yb, 3, s[0])
        print(f"{lab:>26}{s[0]:>9.3f}{s[1]:>9.3f}{s[2]:>9.3f}{s[3]:>9.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}{pv:>7.3f}"
              + ("*" if pv < 0.05 else " "), flush=True)
    print("  detection floor: nothing below AUC ~0.66 in-house is "
          "distinguishable from chance")

    # ---------------------------------------------------------------- #
    # 2. The in-house 3-class protocol, 10 ET in every test set
    # ---------------------------------------------------------------- #
    frac = ET_TEST / int((own_y == 2).sum())
    n_te = {0: int(round(frac * (own_y == 0).sum())),
            1: int(round(frac * (own_y == 1).sum())), 2: ET_TEST}
    prev = n_te[2] / sum(n_te.values())
    print(f"\n{'='*82}")
    print(f"in-house 3-class, {ET_TEST} ET per test set "
          f"(N={n_te[0]} PD={n_te[1]} ET={n_te[2]}, ET prevalence {prev:.3f})")
    print(f"{REPEATS} repeats, paired; only the spectrum differs")
    print(f"{'='*82}")

    res = {"multitaper (current)": [], "short-window": []}
    for rep in range(REPEATS):
        rng = np.random.default_rng(rep)
        te, rest = [], []
        for c in (0, 1, 2):
            idx = np.flatnonzero(own_y == c)
            rng.shuffle(idx)
            te.extend(idx[:n_te[c]]); rest.extend(idx[n_te[c]:])
        te = np.array(sorted(te)); rest = np.array(sorted(rest))
        rng.shuffle(rest)
        n_va = max(int(0.25 * len(rest)), 12)
        va, tr = np.sort(rest[:n_va]), np.sort(rest[n_va:])
        res["multitaper (current)"].append(
            fit_eval(own_spec_mt, own_desc, own_traj, own_y, tr, va, te))
        res["short-window"].append(
            fit_eval(own_spec_sw, own_desc, own_traj, own_y, tr, va, te))
        print(f"  repeat {rep+1}/{REPEATS}", flush=True)

    for k in res:
        res[k] = np.array(res[k])
    print(f"\n{'spectrum':>26}" + "".join(f"{c:>9}" for c in NM)
          + "   sd(macroP)")
    for k in res:
        print(f"{k:>26}" + "".join(f"{v:>9.3f}" for v in res[k].mean(0))
              + f"{res[k][:, 3].std():>12.3f}")

    print("\npaired short-window - multitaper, same test patients:")
    for (dd, lo, hi), c in zip(paired(res["short-window"],
                                      res["multitaper (current)"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
