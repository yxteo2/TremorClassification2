"""How the ensemble's six members are pooled — a free lever nobody has pulled.

The reported model trains **two families x three seeds** and combines them with
`np.mean` of the softmax outputs, twice: mean over seeds inside a family, then
mean over families. That arithmetic average was never a decision. It is the
default, and it has never been compared against anything.

It is also the wrong default for this metric, on a specific argument.

**Precision at 12 % prevalence is read from the very top of the ranking.** A
patient is called ET only if the pooled ET probability wins after the priors.
Arithmetic pooling is *permissive*: one member that is confidently, wrongly sure
a patient is ET can carry the average past the threshold on its own, because
0.95 pulls a mean much harder than 0.05 pushes it back. Geometric pooling —
averaging in log space — is *vetoing*: any member that says 0.05 drags the
product down no matter what the others say. Vetoing costs recall and buys
precision, which is exactly the trade this project wants for ET.

The same argument applies to the median and to a trimmed mean, which discard the
outlying member rather than down-weighting it.

**Calibration is the other half.** `tune_offsets` fits per-class logit offsets on
validation and is the single largest measured gain in the project (ET precision
0.475 -> 0.612). It is fitted on **uncalibrated** network outputs, where the six
members disagree about scale as well as about the patient — a member trained on
a slightly different loss surface can be systematically over-confident, and the
arithmetic mean inherits that. A scalar temperature per member, fitted on the
same validation split, puts them on a common scale *before* pooling.

## What makes this cheap and clean

Every arm is computed from **the same six fitted models** in each split. There is
no retraining, no extra seed, no capacity change — the arms differ only in the
arithmetic applied to six probability matrices that are already in memory. This
is the tightest pairing available in this repo: any difference is the pooling
rule and nothing else, and the whole experiment costs exactly one baseline run.

Arms (each gets its own `tune_offsets` on its own pooled validation matrix):

  arithmetic (current)     mean over seeds, then over families
  geometric                mean of log-probabilities, renormalised
  median                   per-class median over the six members
  trimmed mean             drop the highest and lowest member per class
  temperature -> arith     scalar T per member on val, then arithmetic
  temperature -> geom      scalar T per member on val, then geometric
  family weight on val     geometric across families with the weight chosen on val

Validation is used for temperatures, family weight and priors, exactly as it is
already used for priors. **Test is never touched by any of it.**

Run: ``python -m experiments.pooling_rules``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL, build
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS, SEEDS = 20, (0, 1, 2)
WGRID = np.linspace(0.0, 1.0, 11)


def _norm(P):
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def _geo(stack):
    """Geometric mean over axis 0 of a (m, n, 3) probability stack."""
    return _norm(np.exp(np.log(np.clip(stack, 1e-12, None)).mean(0)))


def temperature(p, y, grid=np.linspace(0.4, 3.0, 27)):
    """Scalar T minimising validation NLL of ``p`` re-scaled in log space."""
    lp = np.log(np.clip(p, 1e-12, None))
    best, bt = np.inf, 1.0
    for t in grid:
        q = _norm(np.exp(lp / t))
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best:
            best, bt = nll, t
    return bt


def apply_T(p, t):
    return _norm(np.exp(np.log(np.clip(p, 1e-12, None)) / t))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def _val_macro_f1(pv, off, yva):
    _, _, F, _ = precision_recall_fscore_support(
        yva, (np.log(pv + 1e-12) + off).argmax(1), labels=[0, 1, 2],
        zero_division=0)
    return F.mean()


def fit_members(spec, desc, traj, y, tr, va, te):
    """The six fitted members. Returns (V, T) stacks of shape (6, n, 3)."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    V, T = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        for s in SEEDS:
            pv, pt = train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd,
                           y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd],
                           seed=s)
            V.append(pv)
            T.append(pt)
    return np.stack(V), np.stack(T)


def pooled(V, T, yva):
    """Every pooling rule, as {arm: (p_val, p_test)}. Only val sees labels."""
    out = {}
    out["arithmetic (current)"] = (_norm(V.mean(0)), _norm(T.mean(0)))
    out["geometric"] = (_geo(V), _geo(T))
    out["median"] = (_norm(np.median(V, 0)), _norm(np.median(T, 0)))

    def trim(S):
        S = np.sort(S, 0)
        return _norm(S[1:-1].mean(0))
    out["trimmed mean"] = (trim(V), trim(T))

    Ts = [temperature(V[i], yva) for i in range(len(V))]
    Vc = np.stack([apply_T(V[i], Ts[i]) for i in range(len(V))])
    Tc = np.stack([apply_T(T[i], Ts[i]) for i in range(len(T))])
    out["temperature -> arith"] = (_norm(Vc.mean(0)), _norm(Tc.mean(0)))
    out["temperature -> geom"] = (_geo(Vc), _geo(Tc))

    # family weight: geometric across the two family means, weight on val
    nf = len(V) // 2
    fv = [_geo(V[:nf]), _geo(V[nf:])]
    ft = [_geo(T[:nf]), _geo(T[nf:])]
    best = (-1.0, 0.5)
    for w in WGRID:
        bv = _norm(np.exp((1 - w) * np.log(fv[0] + 1e-12)
                          + w * np.log(fv[1] + 1e-12)))
        f = _val_macro_f1(bv, tune_offsets(bv, yva), yva)
        if f > best[0]:
            best = (f, w)
    w = best[1]
    mix = lambda a, b: _norm(np.exp((1 - w) * np.log(a + 1e-12)
                                   + w * np.log(b + 1e-12)))
    out["family weight on val"] = (mix(fv[0], fv[1]), mix(ft[0], ft[1]))
    return out, Ts, w


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print("all arms share the SAME six fitted members per split; "
          "only the pooling arithmetic differs\n", flush=True)

    res, temps, wts = {}, [], []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        V, T = fit_members(spec, D, traj, y, tr, va, te)
        arms, Ts, w = pooled(V, T, y[va])
        temps.append(Ts)
        wts.append(w)
        for lab, (pv, pt) in arms.items():
            res.setdefault(lab, []).append(score(pt, tune_offsets(pv, y[va]),
                                                 y[te]))
        print(f"  split {sp+1}/{SPLITS}  temps "
              f"{min(Ts):.2f}-{max(Ts):.2f}  family w={w:.1f}", flush=True)

    for a in res:
        res[a] = np.array(res[a])
    order = list(res)

    print(f"\n{'rule':>24}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in order:
        print(f"{a:>24}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["arithmetic (current)"]
    print("\npaired vs arithmetic pooling, same splits AND the same fitted "
          "models:")
    for a in order[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nsplit-level win rate vs arithmetic:")
    for a in order[1:]:
        print(f"  {a:>24}: " + "  ".join(
            f"{c} {float((res[a][:, i] > base[:, i]).mean()):.2f}"
            for i, c in enumerate(NM)))

    tt = np.array(temps)
    print(f"\nfitted temperatures: mean {tt.mean():.2f}, "
          f"range {tt.min():.2f}-{tt.max():.2f}  "
          f"(>1 means the members were over-confident)")
    print(f"family weight on val: mean {np.mean(wts):.2f} "
          f"(0 = TwoStream only, 1 = TCN only)")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
