"""Three cheap levers not yet pulled: feature selection, calibration, seeds.

**Feature selection.** Seven feature unions in this project have underperformed
their best member, which says dimensionality binds harder than information at
404 patients with 49 ET. Every one of those unions was chosen by hand and tested
for harm. Nobody has run a *search*. Greedy forward selection over the feature
pool, scored on the VALIDATION split inside each outer split, directly exploits
the finding instead of working around it.

**Calibration.** Validation-tuned class priors were the single largest gain in
the project (ET precision 0.475 -> 0.612), and they are fitted on *uncalibrated*
network probabilities. Temperature scaling first -- one scalar, fitted on the
same validation split -- should let the offsets act on a better-behaved
distribution.

**Seeds.** Everything runs 3 seeds. More seeds reduce the variance of the
averaged probability at no modelling risk.

Every arm is paired against the same baseline on the same 20 splits.

Run: ``python -m experiments.selection_and_calibration``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.trajectory_tuning import assemble
from models.architectures import ResidualTCN, Spectrum1DCNN, TRUNKS, TwoStreamNet

SPLITS = 20
TL = 64


def temperature(logp, y, grid=np.linspace(0.4, 3.0, 27)):
    """Scalar temperature minimising validation NLL. Fitted on val only."""
    best, bt = np.inf, 1.0
    for t in grid:
        z = logp / t
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best:
            best, bt = nll, float(t)
    return bt


def _fit_split(packed, spec, y, tr, va, te, nd, n_ch, seeds):
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL, n_traj_ch=n_ch)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s) for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def score(y_true, pred):
    P, _, F, _ = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def run(name, spec, desc, traj, n_ch, y, key, seeds=(0, 1, 2), calibrate=False,
        select=False, pool=None, verbose=True):
    """``pool``: list of (label, column-index-array) blocks selection may choose."""
    out, chosen = [], []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        if select:
            # greedy forward selection over feature blocks, scored on VALIDATION
            keep, best_s = [], -np.inf
            remaining = list(range(len(pool)))
            while remaining:
                cand_best, cand_i = best_s, None
                for i in remaining:
                    cols = np.concatenate([pool[j][1] for j in keep + [i]])
                    d = desc[:, cols]
                    pk = np.hstack([spec, d, traj])
                    pv, _ = _fit_split(pk, spec, y, tr, va, va, d.shape[1],
                                       n_ch, (0,))
                    s = score(y[va], pv.argmax(1))[3]
                    if s > cand_best:
                        cand_best, cand_i = s, i
                if cand_i is None:
                    break
                keep.append(cand_i); remaining.remove(cand_i); best_s = cand_best
            cols = np.concatenate([pool[j][1] for j in keep]) if keep else \
                np.array([], int)
            chosen.append(tuple(pool[j][0] for j in keep))
            d = desc[:, cols] if len(cols) else np.zeros((len(y), 1))
        else:
            d = desc
        packed = np.hstack([spec, d, traj])
        pv, pt = _fit_split(packed, spec, y, tr, va, te, d.shape[1], n_ch, seeds)

        lp_v, lp_t = np.log(pv + 1e-12), np.log(pt + 1e-12)
        if calibrate:
            T = temperature(lp_v, y[va])
            lp_v, lp_t = lp_v / T, lp_t / T
            pv = np.exp(lp_v - lp_v.max(1, keepdims=True))
            pv /= pv.sum(1, keepdims=True)
        pred = (lp_t + tune_offsets(pv, y[va])).argmax(1)
        out.append(score(y[te], pred))
    a = np.array(out)
    if verbose:
        m, s = a.mean(0), a.std(0)
        print(f"{name:>34}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
        if chosen:
            from collections import Counter
            c = Counter(chosen)
            print(f"{'':>34}  most-selected subsets: "
                  + "; ".join(f"{'+'.join(k) or 'none'} x{v}"
                              for k, v in c.most_common(3)), flush=True)
    return a


def paired(a, b, name, n=4000):
    d = a - b
    print(f"  {name}:")
    for i, nm in enumerate(("precN", "precPD", "precET", "macroP", "macroF1")):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def main():
    torch.set_num_threads(1)
    spec, desc, traj, n_ch, y, key = assemble(axis_mode="mean", n_out=TL)
    nd = desc.shape[1]
    # desc = [10 descriptors | 4 asymmetry | 1 availability]
    pool = [("desc10", np.arange(0, 10)),
            ("asym", np.arange(10, 14)),
            ("avail", np.arange(14, nd))]
    print(f"n={len(y)}  spectrum {spec.shape[1]}  desc pool {nd} "
          f"({', '.join(p[0] for p in pool)})  trajectory {n_ch}x{TL}\n")
    print(f"{'config':>34}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    res = {}
    res["base"] = run("FINAL (baseline, 3 seeds)", spec, desc, traj, n_ch, y, key)
    res["cal"] = run("+ temperature calibration", spec, desc, traj, n_ch, y, key,
                     calibrate=True)
    res["s7"] = run("+ 7 seeds", spec, desc, traj, n_ch, y, key,
                    seeds=(0, 1, 2, 3, 4, 5, 6))
    res["cal7"] = run("+ calibration + 7 seeds", spec, desc, traj, n_ch, y, key,
                      seeds=(0, 1, 2, 3, 4, 5, 6), calibrate=True)
    res["sel"] = run("+ greedy feature selection", spec, desc, traj, n_ch, y, key,
                     select=True, pool=pool)

    print(f"\npaired vs baseline, {SPLITS} splits:")
    for k, lbl in (("cal", "temperature calibration"), ("s7", "7 seeds"),
                   ("cal7", "calibration + 7 seeds"),
                   ("sel", "greedy feature selection")):
        paired(res[k], res["base"], lbl)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
