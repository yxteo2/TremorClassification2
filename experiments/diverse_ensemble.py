"""Different ARCHITECTURES as ensemble members — the one ensemble variation never tried.

Architecture as a *replacement* is the most thoroughly closed axis in this
project: 33 parameters (logreg) to 85.8 M (frozen ImageNet ViT), 15+ families,
and the measured ordering says **smaller is better**. Nothing there is left.

But every *ensemble* variation tested added *more of the same*:

    6 seeds instead of 3            null (macroP −0.006)
    balanced bagging, 6 bags        null (+0.001)
    7 pooling rules over 6 members  all within ±0.012

**Adding a structurally different family has never been tested**, and the six
current members correlate at **r = 0.859** on p(ET) — they are two families
(TwoStreamNet CNN, ResidualTCN) × three seeds. Ensemble gains come from
decorrelation, not from individual strength, so a member that is individually
*worse* but differently wrong is exactly what a correlated ensemble wants.

Two candidates, both individually null against the reported model, both
structurally unlike the incumbents:

    SpectrumTransformer    attends over frequency bins. Individually macroP
                           −0.011 n.s., but **halves sd(precET): 0.210 → 0.120**
                           (`attention_fair_test.md`). Variance is the binding
                           measurement constraint in this project, and that
                           observation was left explicitly open.
    ConvTimeTransformer    permutation-invariant bag-of-frames over the time axis
    (pos_enc=False)        that `P.mean(0)` deletes. Individually +0.013 n.s.
                           over a standalone CNN (`time_axis_transformer.md`),
                           and it reads a *different input* from every incumbent.

## Arms, and the control that decides it

    reported            the 6 members as published
    + transformer       6 + 3 SpectrumTransformer seeds       = 9
    + bag-of-frames     6 + 3 ConvTimeTransformer seeds       = 9
    + both              6 + 3 + 3                             = 12
    CONTROL: + seeds    6 + 3 more seeds of the EXISTING two families = 9

**The control is the whole experiment.** Arms 2–4 change two things at once —
they add members *and* they add architectural diversity. The extra-seeds arm adds
members only. Any gain that the seed control reproduces is ensemble size, not
diversity, and `balanced_bagging.md` already measured 6-vs-3 seeds as null, which
is what makes this control cheap to interpret.

Every member is fitted **once per split** and pooled in software, so the five
arms cost 15 fits per split rather than 45.

## Prediction, recorded before the run

**Null, and the seed control will match the diverse arms.** The prior is
unfavourable and specific: `pooling_rules.md` found that *how* the six members
are combined does not matter within ±0.012, `ensemble_diversity.md` found they
already disagree on 20.5 % of patients, and the contested 40.5 % score at chance
— so the binding constraint is not that the members are too alike, it is that
the patients they disagree about are genuinely unreadable.

**The one place a gain is plausible is variance, not mean.** If the transformer's
halved sd(precET) survives pooling, the ensemble's sd should fall without its
mean moving. That would be worth having — 20 splits resolves ~0.04 and every
candidate in this project has been smaller than that — but it is a *measurement*
improvement, not a performance one, and must be reported as such.

`nETpred` and `recET` are scored on every arm, because the variance claim cannot
be separated from a model that simply predicts ET less often and more uniformly —
the trap the epoch-selection run sprang (`newdata_epoch_choice.md`).

## A second pre-registered reading, added while this was running

Two unrelated interventions in this project land on the **identical** variance
figure:

    STAB replacing descriptors   precET sd 0.185 -> 0.120  (temporal_stability.md)
    SpectrumTransformer          precET sd 0.210 -> 0.120  (attention_fair_test.md)

That coincidence is explained by arithmetic rather than by either method working:

    ET predictions per test fold          8.95  (measured)
    precision granularity 1/nETpred       0.112
    binomial sd at p = 0.654, n = 8.95    0.159

**precET cannot have a standard deviation meaningfully below its own quantisation
step of 0.112.** So sd = 0.120 is a *floor set by the ET count*, not a reduction
earned by a method — and it sits *below* the binomial expectation of 0.159, which
is what a model that predicts ET more uniformly would produce.

**Prediction: the transformer arm's sd drop is conservatism.** If so, its
`nETpred` is lower and less variable than the reported model's, and the variance
"gain" should be reported as an artefact of the ET count. If instead `nETpred`
holds while sd falls, that is a genuine stabilisation and worth having. The arms
already score `nETpred`, so this run decides it without further work.

20 splits, per-arm checkpoints. Run: ``python -m experiments.diverse_ensemble``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import logbin
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pcen_hpss import HOP_DENSE, surface
from models.architectures import (ConvTimeTransformer, ResidualTCN,
                                  Spectrum1DCNN, SpectrumTransformer, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
SEEDS = (0, 1, 2)
EXTRA_SEEDS = (3, 4, 5)
TL, N_FRAMES = 64, 46
ARMS = ("reported", "+ transformer", "+ bag-of-frames", "+ both",
        "CONTROL + seeds")


def frame_stack(x):
    E, f = surface(x, HOP_DENSE)
    k = (f >= 3.0) & (f <= FM.GRID[-1])
    P = E.mean(axis=0)[k]
    rows = [np.clip(np.interp(FM.GRID, f[k], P[:, t], left=0.0, right=0.0),
                    0, None) for t in range(P.shape[1])]
    R = np.array([v / (v.sum() + 1e-20) for v in rows])
    if len(R) < N_FRAMES:
        R = np.vstack([R, np.repeat(R[-1:], N_FRAMES - len(R), axis=0)])
    return logbin(R[:N_FRAMES])


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

    # patient-level frame stacks for the bag-of-frames member. Averaging a
    # patient's recordings is safe here BECAUSE that member is
    # permutation-invariant, so the order-blurring objection from
    # time_axis_transformer.md does not apply; it does blur the distribution a
    # little, which is noted rather than hidden.
    (rA, rB, rC), keep = load_cohorts()
    stacks = defaultdict(list)
    for recs, ch, tag in ((rA, slice(3, 6), "2015"), (rB, slice(3, 6), "New"),
                          (rC, slice(0, 3), "PADS")):
        for r in recs:
            stacks[(tag, r.subject)].append(
                frame_stack(r.x[ch] if r.x.shape[0] > 3 else r.x))
    order = []
    for tag, recs in (("2015", rA), ("New", rB)):
        order += [(tag, s) for s in sorted({r.subject for r in recs})]
    order += [("PADS", s) for s in np.array(sorted({r.subject for r in rC}))[keep]]
    assert len(order) == len(y), f"{len(order)} vs {len(y)}"
    FS_ = np.array([np.stack(stacks[k]).mean(0) for k in order], np.float32)

    print(f"n={len(y)}  current members correlate r=0.859 on p(ET)")
    print("prediction on record: null, and the seed CONTROL will match the")
    print("diverse arms; the only plausible gain is in VARIANCE, not mean\n",
          flush=True)

    res, done = {}, {}
    for a in ARMS:
        r, dn = resume_load("divens_" + a.replace(" ", "_"), (a,))
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

        def fit(X, mk, seeds):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            out = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd,
                         y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                   for s in seeds]
            return ([a[0] for a in out], [a[1] for a in out])

        mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                   8 * 2 * 4, NBIN, nd, TL)
        mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
        mk3 = lambda: TwoStreamNet(SpectrumTransformer(NBIN, 3, d=32),
                                   TRUNKS["trunk"], 32, NBIN, nd, TL)
        mk4 = lambda: ConvTimeTransformer(NBIN, N_FRAMES, 3, d=32,
                                          pos_enc=False)

        V, T = defaultdict(list), defaultdict(list)
        for tag, X, mk, sds in (("base", packed, mk1, SEEDS),
                                ("base", SPEC, mk2, SEEDS),
                                ("xtra", packed, mk1, EXTRA_SEEDS),
                                ("xtra", SPEC, mk2, EXTRA_SEEDS),
                                ("tf", packed, mk3, SEEDS),
                                ("bof", FS_, mk4, SEEDS)):
            v, t = fit(X, mk, sds)
            V[tag] += v
            T[tag] += t

        pools = {
            "reported": ["base"],
            "+ transformer": ["base", "tf"],
            "+ bag-of-frames": ["base", "bof"],
            "+ both": ["base", "tf", "bof"],
            "CONTROL + seeds": ["base", "xtra"],
        }
        for arm, tags in pools.items():
            if sp in done[arm]:
                continue
            pv = np.mean([p for t_ in tags for p in V[t_]], 0)
            pt = np.mean([p for t_ in tags for p in T[t_]], 0)
            res[arm].append(score(pt, tune_offsets(pv, y[va]), y[te]))
            resume_save("divens_" + arm.replace(" ", "_"), {arm: res[arm]}, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a]) for a in ARMS}
    print(f"\n{'arm':>18}{'members':>9}" + "".join(f"{c:>9}" for c in NM)
          + f"{'sd(precET)':>12}")
    n_mem = {"reported": 6, "+ transformer": 9, "+ bag-of-frames": 9,
             "+ both": 12, "CONTROL + seeds": 9}
    for a in ARMS:
        print(f"{a:>18}{n_mem[a]:>9}"
              + "".join(f"{v:>9.3f}" for v in R[a].mean(0))
              + f"{R[a][:, 2].std():>12.3f}")

    for base, tag in (("reported", "ADOPTION"),
                      ("CONTROL + seeds", "ATTRIBUTION — is it diversity or "
                                          "just more members?")):
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
