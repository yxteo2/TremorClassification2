"""Is "a harmful training subject" even an estimable quantity in this data?

`influence_prune.md` implemented the right idea — drop the N and PD subjects whose
presence in training makes the model worse — and it did not help: influence-drop 5
was no better than random-drop 5 and trended worse (precET −0.045 [−0.108,
+0.005]).

But that report identified a specific weakness rather than closing the question.
The influence ranking was **unstable**: across 20 outer splits the most-frequently
dropped subject was chosen in only 8, and most in 2–5. If the ranking is mostly
noise then the selection is random with extra steps, which is exactly what the
influence-vs-random comparison looked like. The obvious check — make the estimate
stronger and see whether it stabilises — was flagged and not run.

This runs it, and puts the diagnostic **before** the expensive part.

## A much stronger estimator

Instead of 240 random subsets scored by a Monte-Carlo contrast, this uses **exact
leave-one-out**, repeated over many inner splits:

    for each of R inner (fit / score) splits of the training fold:
        base = score(model fit on all of fit_idx)
        for each subject i in fit_idx:
            harm[i] += score(model fit on fit_idx minus i) - base
    harm /= R

`harm[i] > 0` means the model scores **better without** subject i — the definition
of a harmful subject. No Monte-Carlo sampling noise remains; the only noise is the
finite score set, which averaging over R inner splits reduces directly.

## The diagnostic that decides everything

Before any subject is dropped, the run reports **split-half reliability**: the
Spearman correlation between `harm` estimated from the first R/2 inner splits and
from the second R/2. Same subjects, same fold, independent halves of the evidence.

* **High reliability** → harmfulness is a real, measurable property of subjects,
  and it is worth acting on. Any failure after that is about the intervention.
* **Near-zero reliability** → the quantity is not estimable at this sample size,
  and *no* pruning rule built on it can work, however it is applied. That is a
  definitive answer to the question rather than another null result.

This is printed first so the answer survives even if the run is interrupted.

Constraints unchanged: influence is computed inside the training fold only,
validation is left alone for the priors, test is never touched, ET is never
dropped.

Run: ``python -m experiments.influence_stable``
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC
from experiments.alltasks_final import paired
from experiments.final_model import build
from experiments.prune_training import fit_eval

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
DROP_CLASSES = (0, 1)
R_INNER, SCORE_FRAC = 20, 0.35


def _fit_score(Xf, yf, Xs, ys):
    mdl = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000,
                                           class_weight="balanced"))
    try:
        mdl.fit(Xf, yf)
        P, _, _, _ = precision_recall_fscore_support(
            ys, mdl.predict(Xs), labels=[0, 1, 2], zero_division=0)
        return float(P.mean())
    except Exception:
        return np.nan


def loo_harm(X, y, tr, seed=0, r=R_INNER):
    """Exact leave-one-out harm per training subject, averaged over r inner splits.

    Returns (harm dict, split-half rho, top-k overlap, chance overlap).
    ``harm[i] > 0`` means the model scores BETTER without subject i.

    Calibrated on synthetic data: with 8 planted mislabels the top-k overlap is
    0.60 against a chance level of 0.12 (4.9x); with none planted it is 0.00. So
    the overlap statistic separates "there is a harmful set" from "there is not".
    """
    halves = [np.zeros(len(tr)), np.zeros(len(tr))]
    counts = [0, 0]
    for rep in range(r):
        isc, ifit = next(StratifiedShuffleSplit(
            1, test_size=1.0 - SCORE_FRAC,
            random_state=seed * 100 + rep).split(X[tr], y[tr]))
        fit_idx, sc_idx = ifit, isc
        Xf, yf = X[tr][fit_idx], y[tr][fit_idx]
        Xs, ys = X[tr][sc_idx], y[tr][sc_idx]
        base = _fit_score(Xf, yf, Xs, ys)
        if not np.isfinite(base):
            continue
        h = np.zeros(len(tr))
        for j, pos in enumerate(fit_idx):
            if y[tr][pos] not in DROP_CLASSES:
                continue
            keep = np.ones(len(fit_idx), bool)
            keep[j] = False
            if len(np.unique(yf[keep])) < 3:
                continue
            s = _fit_score(Xf[keep], yf[keep], Xs, ys)
            if np.isfinite(s):
                h[pos] = s - base            # >0 : better WITHOUT this subject
        halves[rep % 2] += h
        counts[rep % 2] += 1

    if min(counts) == 0:
        return {}, np.nan, np.nan, np.nan
    a, b = halves[0] / counts[0], halves[1] / counts[1]
    cand = np.flatnonzero(np.isin(y[tr], DROP_CLASSES)
                          & ((a != 0) | (b != 0)))
    rho = spearmanr(a[cand], b[cand]).statistic if len(cand) > 5 else np.nan

    # TOP-K OVERLAP is the diagnostic that matters, not the global rho.
    # Calibration on synthetic data with 8 deliberately mislabelled subjects gave
    # rho = 0.181 while still recovering 6 of them -- because most subjects have
    # true harm ~0, so the global correlation is dominated by ranking noise among
    # the irrelevant majority. What the intervention actually depends on is
    # whether the subjects it would DROP are the same in both halves.
    k = min(10, max(2, len(cand) // 8))
    ta = set(cand[np.argsort(-a[cand])[:k]])
    tb = set(cand[np.argsort(-b[cand])[:k]])
    overlap = len(ta & tb) / k if k else np.nan
    # chance overlap for two independent top-k picks out of len(cand)
    chance = k / len(cand) if len(cand) else np.nan

    mean_h = (halves[0] + halves[1]) / (counts[0] + counts[1])
    return ({int(tr[i]): float(mean_h[i]) for i in cand},
            float(rho), float(overlap), float(chance))


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
    print(f"exact leave-one-out harm, averaged over {R_INNER} inner splits\n",
          flush=True)

    # ---------------------------------------------------------------- #
    # Stage 1: is harmfulness estimable at all? Printed before anything
    # expensive, so the answer survives an interrupted run.
    # ---------------------------------------------------------------- #
    splits, harms, rhos, overlaps = [], [], [], []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        h, rho, ov, ch = loo_harm(packed, y, tr, seed=sp)
        splits.append((tr, va, te))
        harms.append(h)
        rhos.append(rho)
        overlaps.append((ov, ch))
        print(f"  split {sp+1}/{SPLITS}  scored {len(h)} subjects, "
              f"rho {rho:+.3f}, top-k overlap {ov:.2f} (chance {ch:.2f})",
              flush=True)

    rr = np.array([r for r in rhos if np.isfinite(r)])
    print(f"\n{'='*70}")
    print("SPLIT-HALF RELIABILITY OF THE HARM RANKING")
    print(f"{'='*70}")
    print(f"  mean rho {rr.mean():+.3f}   median {np.median(rr):+.3f}   "
          f"range [{rr.min():+.3f}, {rr.max():+.3f}]")
    ov = np.array([o for o, _ in overlaps if np.isfinite(o)])
    chn = np.array([c for _, c in overlaps if np.isfinite(c)])
    print(f"  TOP-K OVERLAP between halves: mean {ov.mean():.2f}  "
          f"chance {chn.mean():.2f}  ratio {ov.mean()/max(chn.mean(),1e-9):.1f}x")
    print("  (top-k overlap is the operative diagnostic; on synthetic data with")
    print("   8 planted mislabels it recovered 6 while global rho was only 0.181)")
    if ov.mean() < 2 * chn.mean():
        print("  => the subjects this would DROP are barely more consistent than")
        print("     chance. No pruning rule built on this ranking can work.")
    else:
        print("  => the dropped set is reproducible; the drop below is a fair test.")
    print(flush=True)

    # cross-split agreement: do the same subjects rank as harmful in other folds?
    allsub = sorted({i for h in harms for i in h})
    mat = np.full((len(harms), len(allsub)), np.nan)
    pos = {s: j for j, s in enumerate(allsub)}
    for r_, h in enumerate(harms):
        for i, v in h.items():
            mat[r_, pos[i]] = v
    cs = []
    for a in range(len(harms)):
        for b in range(a + 1, len(harms)):
            m = np.isfinite(mat[a]) & np.isfinite(mat[b])
            if m.sum() > 20:
                cs.append(spearmanr(mat[a][m], mat[b][m]).statistic)
    if cs:
        print(f"  cross-split agreement (harm ranking, different folds): "
              f"mean rho {np.nanmean(cs):+.3f}\n", flush=True)

    # ---------------------------------------------------------------- #
    # Stage 2: apply it anyway, with the matched random control
    # ---------------------------------------------------------------- #
    ARMS = ("k=0 (baseline)", "LOO-harm drop 5", "random-drop 5")
    res = {a: [] for a in ARMS}
    dropped = []
    for sp, (tr, va, te) in enumerate(splits):
        h = harms[sp]
        drop = []
        for cl in DROP_CLASSES:
            cand = sorted(((v, i) for i, v in h.items() if y[i] == cl),
                          reverse=True)          # most harmful first
            drop.extend(i for _, i in cand[:5])
        drop = np.array(sorted(drop), int)
        dropped.append(drop)

        rng = np.random.default_rng(7000 + sp)
        rdrop = []
        for cl in DROP_CLASSES:
            p = np.array([i for i in tr if y[i] == cl])
            if len(p) > 5:
                rdrop.extend(rng.choice(p, 5, replace=False))
        rdrop = np.array(sorted(rdrop), int)

        res["k=0 (baseline)"].append(fit_eval(spec, D, traj, y, tr, va, te))
        res["LOO-harm drop 5"].append(
            fit_eval(spec, D, traj, y, np.setdiff1d(tr, drop), va, te))
        res["random-drop 5"].append(
            fit_eval(spec, D, traj, y, np.setdiff1d(tr, rdrop), va, te))
        print(f"  eval split {sp+1}/{SPLITS}", flush=True)

    for a in res:
        res[a] = np.array(res[a])
    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>20}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    print("\npaired vs k=0:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], res["k=0 (baseline)"]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nLOO-harm vs random — the comparison that decides it:")
    for (dd, lo, hi), c in zip(paired(res["LOO-harm drop 5"],
                                      res["random-drop 5"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    from collections import Counter
    cnt = Counter(int(i) for g in dropped for i in g)
    print(f"\nmost frequently dropped (of {SPLITS} splits):")
    for i, c in cnt.most_common(12):
        tag = f"  {coh[i]}" if coh is not None and i < len(coh) else ""
        print(f"    idx {i:>4}  class {int(y[i])}  {c:>2}/{SPLITS}{tag}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
