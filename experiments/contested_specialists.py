"""Is there any signal left in the 40 % of patients the ensemble argues about?

`ensemble_diversity.md` measured the ceiling in one number: the six ensemble
members agree on 59.5 % of patients and are 68.8 % correct there, and on the
contested 40.5 % they are **48.5 % correct against a 46.5 % constant-prediction
baseline**. The contested patients sit on the boundary (top-2 margin 0.094
against 0.369) and are collectively almost uninformative.

That result predicts that any method which merely reshuffles the contested set
cannot help — which is exactly what seven pooling rules, temperature scaling and
seven prior objectives all found. It leaves one question open, and it is the
question that decides where the project goes next:

**Is the contested set unclassifiable in principle, or only unclassifiable from
the multitaper spectrum?**

Those are very different situations. If some *other* representation of the same
recordings is above chance precisely where the spectral model is at chance, then
the ceiling is a representation problem and fusion targeted at that subset is the
way through. If every representation is at chance there, the ceiling is intrinsic
to the task at this sample size and no amount of modelling will move it.

## What is tested

The deep ensemble defines the contested set exactly as before — the patients its
six members do not label unanimously. Then each feature block that
`final_model.build()` already produces is fitted separately with a plain
logistic regression and scored **on the contested and unanimous subsets
separately**:

  multitaper      the reported model's spectrum
  welch           a different spectral estimator on the same recordings
  wavelet_packet  a time-frequency decomposition
  DESC            hand-built descriptors
  STAB            temporal stability
  TRAJ            the instantaneous-frequency trajectory
  ASYM+HAVE       limb asymmetry
  all blocks      the concatenation

A weak linear model is the right instrument here, not a weakness. The question is
whether *any* usable signal exists in the contested region, and a logistic
regression that beats the constant baseline there is stronger evidence of
complementary structure than a deep model would be, because it cannot be
memorising.

## The baseline that makes it honest

Two baselines are reported, because raw accuracy is the wrong headline for
class-balanced models.

**Raw accuracy** is scored against the **majority-class rate within the contested
subset on that split**, not against 1/3 — the contested set is enriched in one
class and a constant prediction can score high there. But every model here is
trained with balanced class weights and so deliberately does *not* predict the
majority class, which makes that comparison unfair to it.

**Balanced accuracy** — mean per-class recall on the contested subset — is
therefore the headline, and its chance level is 1/3 by construction regardless of
how the subset's classes are distributed. A block counts as finding signal if it
clears 1/3 on balanced accuracy with a paired interval clear of zero.

Nothing here changes the reported model. This is a diagnostic: it uses the
training fold to fit, the test fold to score, and never touches validation.

Run: ``python -m experiments.contested_specialists``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import TEST_FRAC, VAL_FRAC
from experiments.final_model import build
from experiments.pooling_rules import fit_members

SPLITS = 10


def lr_predict(X, y, tr, te):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000,
                                         class_weight="balanced"))
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    BLOCKS = {
        "multitaper": spec,
        "welch": d["SPEC"]["welch"],
        "wavelet_packet": d["SPEC"]["wavelet_packet"],
        "DESC": d["DESC"],
        "STAB": d["STAB"],
        "TRAJ": traj,
        "ASYM+HAVE": A,
    }
    BLOCKS["all blocks"] = np.hstack(list(BLOCKS.values()))

    print(f"n={len(y)}  {SPLITS} splits")
    print("contested = the 6 deep members do not agree on the argmax")
    print("baseline  = majority class WITHIN the subset, per split\n", flush=True)

    acc_c = {b: [] for b in BLOCKS}
    acc_u = {b: [] for b in BLOCKS}
    bal_c = {b: [] for b in BLOCKS}
    base_c, base_u, deep_c, deep_bc, frac = [], [], [], [], []

    def balacc(p, t):
        """Mean per-class recall; chance is 1/3 whatever the class mix."""
        r = [float((p[t == c] == c).mean()) for c in (0, 1, 2) if (t == c).any()]
        return float(np.mean(r)) if r else np.nan

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        _, T = fit_members(spec, np.hstack([d["DESC"], A]), traj, y, tr, va, te)
        arg = np.stack([T[i].argmax(1) for i in range(len(T))])
        unan = (arg == arg[0]).all(0)
        con = ~unan
        yte = y[te]
        frac.append(float(con.mean()))

        maj = lambda m: (float(np.bincount(yte[m], minlength=3).max() / m.sum())
                         if m.any() else np.nan)
        base_c.append(maj(con))
        base_u.append(maj(unan))
        dp = np.mean(T, 0).argmax(1)
        deep_c.append(float((dp[con] == yte[con]).mean()) if con.any()
                      else np.nan)
        deep_bc.append(balacc(dp[con], yte[con]) if con.any() else np.nan)

        for b, X in BLOCKS.items():
            p = lr_predict(X, y, tr, te)
            acc_c[b].append(float((p[con] == yte[con]).mean())
                            if con.any() else np.nan)
            acc_u[b].append(float((p[unan] == yte[unan]).mean())
                            if unan.any() else np.nan)
            bal_c[b].append(balacc(p[con], yte[con]) if con.any() else np.nan)

        print(f"  split {sp+1}/{SPLITS}  contested {con.mean():.3f}  "
              f"majority-in-contested {base_c[-1]:.3f}  "
              f"deep acc {deep_c[-1]:.3f} / bal {deep_bc[-1]:.3f}", flush=True)

    bc, bu = np.nanmean(base_c), np.nanmean(base_u)
    print(f"\ncontested fraction {np.mean(frac):.3f}   deep ensemble on "
          f"contested: acc {np.nanmean(deep_c):.3f}, "
          f"balanced {np.nanmean(deep_bc):.3f}")
    print(f"majority-class baseline: contested {bc:.3f}, unanimous {bu:.3f}")
    print("balanced-accuracy chance is 0.333 by construction\n")

    print(f"{'block':>16}{'bal|contested':>15}{'vs 1/3':>9}"
          f"{'acc|contested':>15}{'vs maj':>9}{'acc|unanim':>12}{'vs maj':>9}")
    rows = []
    for b in BLOCKS:
        c, u, g = np.nanmean(acc_c[b]), np.nanmean(acc_u[b]), np.nanmean(bal_c[b])
        rows.append((b, g, c))
        print(f"{b:>16}{g:>15.3f}{g-1/3:>+9.3f}{c:>15.3f}{c-bc:>+9.3f}"
              f"{u:>12.3f}{u-bu:>+9.3f}")

    # paired bootstrap over splits for EVERY block, balanced accuracy vs 1/3
    rng = np.random.default_rng(0)
    print("\nbalanced accuracy on the contested subset, against 1/3 chance:")
    for b, g, _ in sorted(rows, key=lambda r: -r[1]):
        diff = np.array(bal_c[b]) - 1 / 3
        bs = [rng.choice(diff, len(diff), replace=True).mean()
              for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {b:>16} {diff.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nif no block clears chance on the contested subset, the")
    print("ceiling is intrinsic at this sample size and not a representation")
    print("problem. If one does, targeted fusion on that subset is the lead.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
