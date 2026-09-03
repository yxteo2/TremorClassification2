"""Training-time logit adjustment instead of inverse-frequency class weighting.

The reported model corrects its 167 / 188 / 49 imbalance twice: **class weights**
inside the loss, and **validation-tuned logit offsets** applied afterwards. The
second is the single largest measured gain in the project (ET precision 0.475 →
0.612). The first has never been compared against its modern replacement.

Menon et al. (ICLR 2021) show that inverse-frequency *weighting* and additive
*logit adjustment* are different corrections for the same imbalance, and that
adjustment is the better-behaved one: the loss becomes ``CE(z + tau*log(prior), y)``
and inference uses ``z`` unadjusted. Weighting rescales each example's gradient,
which at 49 ET means a handful of patients dominate the update; adjustment
instead shifts the decision boundary *inside* the loss, is Fisher-consistent for
balanced error, and leaves gradient magnitudes alone. It is now the standard
long-tail baseline — logit adjustment, Balanced Softmax and LDAM are the
canonical family, and recent surveys treat class-balanced reweighting as the
weaker predecessor.

Both corrections are already implemented in the same place, so this is a
three-line change: `common.protocol.train(..., logit_adj=tau)` swaps the weighted
loss for the adjusted one. `logit_adj=None` keeps the existing path bit-exact.

## The prediction, recorded before the run — and it leans negative

This is deliberately not sold as promising. `prior_objective.md` measured what
happens when this project's imbalance correction is aimed at **balanced
accuracy**: precET −0.236 [−0.340, −0.142] *, the largest single loss in that
table. Logit adjustment is Fisher-consistent for exactly that objective. The
mechanism that makes it right for long-tail benchmarks — equalise per-class
error — is the mechanism this dataset has already punished.

So the prediction is: **small, and more likely negative than positive on precET**,
with tau = 0.5 less harmful than tau = 1.0. What would change my mind is the
representation argument — adjustment acts during training and could shape
features rather than only the threshold, which post-hoc offsets cannot do. That
is testable and is why this is worth one run rather than an assumption.

The arm that decides it is **not** LA-versus-baseline alone. `tune_offsets` runs
afterwards in every arm, so a post-hoc correction can absorb whatever LA changed
about the threshold. If LA helps, it must help *through the representation*, and
the signature of that is a gain that survives the offsets — which is what the
paired comparison here measures.

Arms: baseline (class weights), tau = 0.5, tau = 1.0, and tau = 1.0 with class
weights restored on top — the last one over-corrects on purpose, to show the two
mechanisms are not additive.

20 splits, paired. Run: ``python -m experiments.logit_adjustment``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS, SEEDS = 20, (0, 1, 2)


def fit_arm(spec, desc, traj, y, tr, va, te, logit_adj):
    """The reported 2-family x 3-seed ensemble under one imbalance correction."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s,
                   logit_adj=logit_adj)
             for s in SEEDS]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    spec, traj = d["SPEC"]["multitaper"], d["TRAJ"]
    c = np.bincount(y, minlength=3)
    print(f"class counts {c.tolist()}, prior {np.round(c / c.sum(), 3).tolist()}")
    print("logit adjustment adds tau*log(prior) to the logits inside the loss "
          "and predicts from the unadjusted logits\n", flush=True)

    ARMS = {"baseline (class weights)": None, "logit adj tau=0.5": 0.5,
            "logit adj tau=1.0": 1.0}
    res = {a: [] for a in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a, tau in ARMS.items():
            pv, pt = fit_arm(spec, D, traj, y, tr, va, te, tau)
            res[a].append(score(pt, tune_offsets(pv, y[va]), y[te]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    base = res["baseline (class weights)"]
    print("\npaired vs the class-weighted baseline (both then get tune_offsets, "
          "so a gain here is a REPRESENTATION gain):")
    for a in list(ARMS)[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), cc in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {cc:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nsplit-level win rate vs baseline:")
    for a in list(ARMS)[1:]:
        print(f"  {a:>22}: " + "  ".join(
            f"{cc} {float((res[a][:, i] > base[:, i]).mean()):.2f}"
            for i, cc in enumerate(NM)))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
