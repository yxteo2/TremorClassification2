"""A fourth family for the adopted ensemble — does diversity keep paying?

`diverse_ensemble.md` produced the first gain in this project to survive 40
splits: adding three `SpectrumTransformer` members to the six incumbents, precET
+0.040 \\*, macroP +0.016 \\*, macroF1 +0.016 \\*. Against a matched control that
only added seeds, macroF1 +0.017 \\* survived — so part of the gain was genuinely
*diversity*, and part was ensemble size.

The mechanism it points to: a member that **reads the input differently** from
the incumbents decorrelates the ensemble. The bag-of-frames member failed that
test (significantly harmful), so "different" is not enough on its own; the member
also has to be individually decent.

## Two candidates, chosen on that criterion

    logistic regression   linear, on the 15 descriptor+asymmetry features only.
    on descriptors        The most different inductive bias available — no
                          spectrum, no trajectory, no nonlinearity — and
                          individually decent (macroP 0.619 in frozen_vit.md's
                          comparison, against 0.652 for the 6-member model).
    CrossStreamAttention  4.9 k params; spectrum tokens attend to the IF
                          trajectory. Individually macroP 0.633. Different from
                          the transformer member in *what* it attends between.

## Arms — the adopted model is now the baseline

    reported9          CNN, TCN, transformer                  (3 families)
    + logreg           + logistic regression                  (4)
    + crossstream      + CrossStreamAttention × 3 seeds       (4)
    CONTROL + family   + a second CNN family, 3 NEW seeds     (4)

Pooled as the **mean of family means**, so a deterministic logistic regression
carries the same weight as a three-seed family. For the three existing families
that is identical to the member mean already reported.

The control adds a fourth family that is *not* diverse — a second copy of the
strongest existing one — so family count is matched and only diversity differs.

## This is a screen, and says so

**20 splits.** A winner here is a *candidate*, not a result: this project has
seen five ~0.05 effects collapse on doubling, and `diverse_ensemble` itself went
+0.067 → +0.040. Per-arm checkpoints mean a winner can be extended to 40 without
re-running the rest.

## Prediction, recorded before the run

**Diversity saturates: neither candidate beats the control on precET or macroP.**
Logistic regression is the better of the two, possibly positive on macroF1,
because it is the most structurally different; CrossStreamAttention is null,
because it is attention again and the ensemble already has an attention member.
The prior is the `+ both` arm of the previous run, where adding a second new
family cancelled the first's gain.

Run: ``python -m experiments.diverse_ensemble_2``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import experiments.final_model as FM
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from models.architectures import (TRUNKS, CrossStreamAttention, ResidualTCN,
                                  Spectrum1DCNN, SpectrumTransformer,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
SEEDS, EXTRA = (0, 1, 2), (6, 7, 8)
TL = 64
ARMS = ("reported9", "+ logreg", "+ crossstream", "CONTROL + family")


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2],
            float((pred == 2).sum())]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]
    nd = D.shape[1]
    packed = np.hstack([SPEC, D, TR])
    print(f"n={len(y)}  baseline = adopted 9-member ensemble  splits={SPLITS}")
    print("prediction on record: diversity saturates; logreg the better of the"
          " two, crossstream null\n", flush=True)

    res, done = {}, {}
    for a in ARMS:
        r, dn = resume_load("divens2_" + a.replace(" ", "_"), (a,))
        res[a], done[a] = r[a], dn

    for sp in range(SPLITS):
        if all(sp in done[a] for a in ARMS):
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        def fam(X, mk, seeds):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            o = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in seeds]
            return (np.mean([a[0] for a in o], 0), np.mean([a[1] for a in o], 0))

        mk_cnn = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8),
                                      TRUNKS["cnn"], 8 * 2 * 4, NBIN, nd, TL)
        mk_tcn = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
        mk_tf = lambda: TwoStreamNet(SpectrumTransformer(NBIN, 3, d=32),
                                     TRUNKS["trunk"], 32, NBIN, nd, TL)
        mk_xs = lambda: CrossStreamAttention(NBIN, nd, TL)

        F = {"cnn": fam(packed, mk_cnn, SEEDS), "tcn": fam(SPEC, mk_tcn, SEEDS),
             "tf": fam(packed, mk_tf, SEEDS), "xs": fam(packed, mk_xs, SEEDS),
             "cnn2": fam(packed, mk_cnn, EXTRA)}
        sc = StandardScaler().fit(D[tr])
        lr = LogisticRegression(max_iter=5000, class_weight="balanced")
        lr.fit(sc.transform(D[tr]), y[tr])
        F["lr"] = (lr.predict_proba(sc.transform(D[va])),
                   lr.predict_proba(sc.transform(D[te])))

        pools = {"reported9": ["cnn", "tcn", "tf"],
                 "+ logreg": ["cnn", "tcn", "tf", "lr"],
                 "+ crossstream": ["cnn", "tcn", "tf", "xs"],
                 "CONTROL + family": ["cnn", "tcn", "tf", "cnn2"]}
        for arm, fams in pools.items():
            if sp in done[arm]:
                continue
            pv = np.mean([F[f][0] for f in fams], 0)
            pt = np.mean([F[f][1] for f in fams], 0)
            res[arm].append(score(pt, tune_offsets(pv, y[va]), y[te]))
            resume_save("divens2_" + arm.replace(" ", "_"), {arm: res[arm]}, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a]) for a in ARMS}
    print(f"\n{'arm':>18}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>18}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))
    for base, tag in (("reported9", "ADOPTION"),
                      ("CONTROL + family", "ATTRIBUTION — diversity or family "
                                           "count?")):
        print(f"\npaired vs {base!r} ({tag}):")
        for a in ARMS:
            if a == base:
                continue
            print(f"  {a}:")
            for i, ((dd, lo, hi), c) in enumerate(zip(paired(R[a], R[base]),
                                                      NM)):
                star = "*" if lo > 0 or hi < 0 else " "
                w = float((R[a][:, i] > R[base][:, i]).mean())
                print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                      f"   win {w:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
