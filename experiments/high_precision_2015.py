"""Can each class reach >= 80 % precision on 2015 OUT -- and at what recall?

At the macro-F1 operating point `ft` reads precN 0.723 / precPD 0.737 /
precET 0.350 (40 repeats); no model variant has moved ET near 0.80, which would
need a PD-vs-ET AUC around 0.9 (best measured 0.662). The honest route to a
precision target is a **high-confidence decision rule**: flag a patient as class
c only when P(c) clears a threshold, and leave the rest for clinical review.

## Protocol -- nested, so the target is tested, not fitted

Saved out-of-fold probabilities (40 repeats; `segments_2015_runs`,
`fusion_2015_runs` -- same partitions, `ft` identical in both). For repeat r the
5 folds are regenerated exactly (StratifiedKFold(5, shuffle, r)). For each held-
out fold, the threshold on P(c) is the **lowest** one at which the patients of
the **other four folds** reach the target precision with >= `MIN_FLAG` flags;
it is then applied unchanged to the held-out fold. If no threshold reaches the
target on the other folds, the held-out fold gets no flags. Pooled over folds:
precision, recall, flags per repeat.

An **oracle** line (threshold chosen on the scored patients themselves) is
printed beside it, labelled, to show how much a non-nested report would flatter.

## Scores (fixed before running this)

    ft         the adopted 2015 OUT model
    fuse       OUT + REST late fusion (better for N / PD, `fusion_2015.md`)
    ens        mean of ft_x2, ft_seg and fuse_w25 probabilities -- three
               differently built members on the same partitions

Plus the two binary screens a clinic would use: **tremor vs none** (P(PD) +
P(ET)) and **ET vs PD among patients the model calls tremor**.

## Prediction, recorded before the run

**N and PD reach >= 0.80 held-out precision**, at recall roughly 0.5-0.7 (N) and
0.3-0.6 (PD). **ET does not reach 0.80 reliably**: nested held-out precision
around 0.4-0.6 at recall <= 0.2, because a threshold chosen on ~12 training ET
does not carry to 3 held-out ET. The oracle line will claim more.

Run: ``python -m experiments.high_precision_2015``
"""

from __future__ import annotations

import glob

import numpy as np
from sklearn.model_selection import StratifiedKFold

TARGETS = (0.80, 0.90)
MIN_FLAG = 3
CLS = ("N", "PD", "ET")


def load():
    S = sorted(glob.glob("segments_2015_runs/rep*.npz"))
    F = sorted(glob.glob("fusion_2015_runs/rep*.npz"))
    assert len(S) == len(F) == 40, "need both 40-repeat runs"
    reps = []
    for s, f in zip(S, F):
        a, b = np.load(s), np.load(f)
        assert np.allclose(a["p_ft"], b["p_ft"]), f"{s}: partitions differ"
        y = a["y"]
        reps.append((y, {"ft": a["p_ft"], "fuse": b["p_fuse"],
                         "ens": (b["p_ft_x2"] + a["p_ft_seg"] + b["p_fuse_w25"]) / 3}))
    return reps


def threshold(s, pos, target):
    """Lowest threshold whose flagged set reaches `target` precision."""
    o = np.argsort(-s)
    tp = np.cumsum(pos[o])
    n = np.arange(1, len(s) + 1)
    ok = (tp / n >= target) & (n >= MIN_FLAG)
    if not ok.any():
        return np.inf
    return s[o][np.flatnonzero(ok).max()]


def evaluate(s, pos, rep, y, target, nested=True):
    flag = np.zeros(len(s), bool)
    if nested:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(s, y):
            t = threshold(s[tr], pos[tr], target)
            flag[te] = s[te] >= t
    else:
        flag = s >= threshold(s, pos, target)
    k = flag.sum()
    tp = (flag & pos).sum()
    return (tp / k if k else np.nan), tp / pos.sum(), k


def summary(rows):
    r = np.array(rows, float)
    p = r[:, 0]
    ok = ~np.isnan(p)
    bs = [np.nanmean(np.random.default_rng(s).choice(p[ok], ok.sum())) for s in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
    return (f"{np.nanmean(p):6.3f} [{lo:.3f}, {hi:.3f}]  recall {r[:, 1].mean():.3f}  "
            f"flags/rep {r[:, 2].mean():5.1f}  reps>=target {np.mean(p[ok] >= TGT):.2f}"
            f"  (no flags in {int((~ok).sum())})")


def main():
    global TGT
    reps = load()
    y0 = reps[0][0]
    print(f"2015 OUT, 151 patients (N/PD/ET = {np.bincount(y0).tolist()}), 40 repeats\n")
    for TGT in TARGETS:
        print(f"=== target precision {TGT:.2f} (one-vs-rest flag on P(class)) ===")
        for c in range(3):
            for arm in ("ft", "fuse", "ens"):
                for nested in (True, False):
                    rows = [evaluate(P[arm][:, c], y == c, r, y, TGT, nested)
                            for r, (y, P) in enumerate(reps)]
                    tag = "nested" if nested else "ORACLE"
                    print(f"  {CLS[c]:>2}  {arm:<5} {tag:<7} precision " + summary(rows))
        # binary screen: tremor vs none
        for arm in ("ft", "fuse", "ens"):
            rows = [evaluate(P[arm][:, 1] + P[arm][:, 2], y != 0, r, y, TGT)
                    for r, (y, P) in enumerate(reps)]
            print(f"  tremor-vs-none {arm:<5} nested precision " + summary(rows))
        print()
    print("MARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
