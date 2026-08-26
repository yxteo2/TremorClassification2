"""Find the subjects whose PRESENCE IN TRAINING makes the model worse, and drop them.

`prune_training.md` asked a different question and got a clear answer: dropping
the *hardest* N and PD patients is worse than doing nothing **and worse than
dropping the same number at random** (precET −0.065 * hard-vs-random at k=5).
The hardest majority patients turned out to be boundary-defining, not
mislabelled — hard and *useful*.

Difficulty was the wrong criterion. The question is not "which subjects does the
model struggle to place" but **"which subjects, by being in the training set,
make the resulting model worse"**. Those are different sets: a boundary-defining
patient is hard and helpful; a harmful one may be easy to fit and still drag the
model somewhere unhelpful.

That quantity is **influence** — how validation performance changes with a
subject in the training set versus out of it.

## Estimating influence affordably

Exact leave-one-out means one retrain per subject per split, and a single removal
from ~190 subjects is dominated by noise. Instead this uses the standard cheap
Monte-Carlo estimate, the same idea as Data Shapley:

    draw M random subsets of the training patients
    train a surrogate on each, score it on a held-out slice
    influence(i) = mean(score | i present) − mean(score | i absent)

Averaging over many subsets makes the estimate far more stable than a single LOO
deletion, and every subset is a cheap logistic regression rather than the deep
model.

## Keeping it honest

* Influence is computed **inside the training fold only**. The fold is split into
  an inner-fit part and an inner-score part; nothing outside `tr` is read.
* **Validation is untouched** — it tunes the class priors, and scoring influence
  on it would let the selection chase the same set the priors are fitted to.
* **Test is never touched.**
* **ET is never dropped.** Only N (167) and PD (188).
* The surrogate is a logistic regression, not the deep model. That is the main
  approximation: subjects harmful to a linear model may not be harmful to the
  two-stream network. Stated up front rather than discovered later.

## Arms

  k=0                  baseline, the reported model
  influence-drop 5     the 5 most harmful N and 5 most harmful PD leave training
  influence-drop 15    dose-response
  random-drop 5        matched control
  random-drop 15       matched control

The random control is not optional. Removing *k* majority patients is itself a
class-balance change, and `prune_training.md` showed random removal of 10 costs
essentially nothing (macroP −0.002) — so any gain must be measured against random,
not against zero.

Run: ``python -m experiments.influence_prune``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, TL, build
from experiments.prune_training import fit_eval
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
DROP_CLASSES = (0, 1)          # N and PD only
M_SUBSETS, SUBSET_FRAC, SCORE_FRAC = 240, 0.5, 0.30


def influence(X, y, tr, seed=0, m=M_SUBSETS):
    """Monte-Carlo influence of each TRAINING subject on held-out macro precision.

    Positive means the model is BETTER when that subject is in the training set.
    Negative means its presence hurts — the subjects this experiment removes.

    Reads nothing outside ``tr``.
    """
    rng = np.random.default_rng(seed)
    # inner split of the training fold: fit on one part, score on the other
    isc, ifit = next(StratifiedShuffleSplit(
        1, test_size=1.0 - SCORE_FRAC, random_state=seed).split(X[tr], y[tr]))
    fit_idx, score_idx = tr[ifit], tr[isc]
    Xs, ys = X[score_idx], y[score_idx]

    present = np.zeros(len(fit_idx))
    s_in = np.zeros(len(fit_idx))
    s_out = np.zeros(len(fit_idx))
    n_out = np.zeros(len(fit_idx))

    for _ in range(m):
        mask = rng.random(len(fit_idx)) < SUBSET_FRAC
        if mask.sum() < 20 or len(np.unique(y[fit_idx[mask]])) < 3:
            continue
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))
        try:
            mdl.fit(X[fit_idx[mask]], y[fit_idx[mask]])
            P, _, _, _ = precision_recall_fscore_support(
                ys, mdl.predict(Xs), labels=[0, 1, 2], zero_division=0)
            sc = P.mean()
        except Exception:
            continue
        s_in[mask] += sc
        present[mask] += 1
        s_out[~mask] += sc
        n_out[~mask] += 1

    ok = (present > 0) & (n_out > 0)
    infl = np.full(len(fit_idx), np.nan)
    infl[ok] = s_in[ok] / present[ok] - s_out[ok] / n_out[ok]
    # subjects held out of the influence estimate are never candidates for removal
    out = {int(i): float(v) for i, v in zip(fit_idx, infl) if np.isfinite(v)}
    return out


def prune_by_influence(y, tr, infl, k):
    """Drop the k most NEGATIVE-influence subjects of each class in DROP_CLASSES."""
    if k <= 0:
        return tr, np.array([], int)
    drop = []
    for cl in DROP_CLASSES:
        cand = [(infl[i], i) for i in tr
                if i in infl and y[i] == cl]
        cand.sort()                       # most negative first
        drop.extend(i for _, i in cand[:k])
    drop = np.array(sorted(drop), int)
    return np.setdiff1d(tr, drop), drop


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]
    packed = np.hstack([spec, D, traj])
    coh = (np.array(["2015"] * 151 + ["NewData"] * 56
                    + ["PADS"] * (len(y) - 207)) if len(y) > 207 else None)

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"influence: {M_SUBSETS} random subsets per split, surrogate logreg, "
          f"scored on a held-out {SCORE_FRAC:.0%} of the training fold")
    print("ET is never dropped; validation and test are never read\n", flush=True)

    # Three arms, not five. This container reverts its working tree frequently
    # and has already killed two long runs; k=5 is the configuration the question
    # is about, and random-drop 5 is the control that decides it. Dose-response
    # at k=15 is worth adding back only once this completes.
    ARMS = (("k=0 (baseline)", 0, "infl"),
            ("influence-drop 5", 5, "infl"),
            ("random-drop 5", 5, "rand"))
    res = {a: [] for a, _, _ in ARMS}
    dropped_all, frac_neg = [], []

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        infl = influence(packed, y, tr, seed=sp)
        vals = np.array([infl[i] for i in tr if i in infl])
        neg = float((vals < 0).mean()) if len(vals) else float("nan")
        frac_neg.append(neg)

        rng = np.random.default_rng(5000 + sp)
        for lab, k, mode in ARMS:
            if mode == "infl":
                tr2, drop = prune_by_influence(y, tr, infl, k)
                if lab == "influence-drop 5":
                    dropped_all.append(drop)
            else:
                drop = []
                for cl in DROP_CLASSES:
                    pos = np.array([i for i in tr if y[i] == cl])
                    if len(pos) > k:
                        drop.extend(rng.choice(pos, k, replace=False))
                tr2 = np.setdiff1d(tr, np.array(sorted(drop), int))
            res[lab].append(fit_eval(spec, D, traj, y, tr2, va, te))
        print(f"  split {sp+1}/{SPLITS}  scored {len(vals)} subjects, "
              f"{neg:.0%} with negative influence", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\nnegative-influence subjects: mean {np.nanmean(frac_neg):.0%} "
          f"of scored training patients")
    print(f"\n{'arm':>22}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, _, _ in ARMS:
        print(f"{lab:>22}" + "".join(f"{v:>9.3f}" for v in res[lab].mean(0))
              + f"{res[lab][:, 3].std():>12.3f}")

    base = res["k=0 (baseline)"]
    print("\npaired vs k=0, same splits:")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\ninfluence vs random at k=5 — the comparison that decides it:")
    for (dd, lo, hi), c in zip(paired(res["influence-drop 5"],
                                      res["random-drop 5"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    from collections import Counter
    cnt = Counter(int(i) for g in dropped_all for i in g)
    print(f"\nmost frequently dropped at k=5 (of {SPLITS} splits):")
    for i, c in cnt.most_common(12):
        tag = f"  {coh[i]}" if coh is not None and i < len(coh) else ""
        print(f"    idx {i:>4}  class {int(y[i])}  {c:>2}/{SPLITS} splits{tag}")
    if coh is not None and cnt:
        top = [i for i, _ in cnt.most_common(30)]
        print(f"  cohort mix of the 30 most-dropped: "
              + "  ".join(f"{c} {sum(1 for i in top if coh[i] == c)}"
                          for c in ("2015", "NewData", "PADS")))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
