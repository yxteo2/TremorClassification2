"""Auditing the project's headline claim at double the split count.

The reported result is:

    multitaper + IF trajectory vs the welch baseline
    macro precision +0.041 [+0.014, +0.067], 20 splits

and the ranked component contributions include

    instantaneous-frequency trajectory  +0.056 precET paired

Both were measured before invariant 6 existed. That invariant says:

    "Raise the split count before believing a difference under ~0.03. 20 splits
     resolves ~0.04, 40 resolves ~0.025. A paired bootstrap over 20 splits removes
     the fold-composition noise the two arms share but NOT the noise in how much
     that particular set of folds favours one arm: a paired macroP +0.021
     [-0.006, +0.048] became +0.005 [-0.020, +0.028] on doubling."

+0.041 sits right at the edge of what 20 splits resolves, and its lower bound is
+0.014 — inside the range where a difference has already evaporated once in this
project. **The headline has never been checked at 40 splits.** Everything in the
paper rests on it, and `kinetic_task_audit.md` has just shown what happens to a
load-bearing claim that predates the machinery needed to test it.

Three arms, 40 shared splits, otherwise the reported protocol exactly:

  base   welch + descriptors + asymmetry            the published baseline
  mt     multitaper + descriptors + asymmetry       transform change alone
  mt_t   multitaper + descriptors + trajectory      the reported model

  mt_t vs base  the headline +0.041 macroP
  mt_t vs mt    the trajectory contribution, reported as +0.056 precET
  mt vs base    the transform contribution on its own

Reported alongside: the split-level win rate, which says how often the claim is
true on an individual split rather than on average — a mean difference that comes
from a minority of splits is a different object from one that holds broadly.

The numbers above are historical motivation, predating the current fixes.
The current run also reports conditional patient-bootstrap intervals and exports
per-patient predictions; neither interval corrects historical model selection.

Run: ``python -m experiments.headline_audit``
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.final_model import SPLITS, build, evaluate
from experiments.patient_level_ci import patient_bootstrap, save_predictions

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
N_SPLITS = 40


def paired(a, b, n=8000):
    d = a - b
    out = []
    for i in range(len(NM)):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        out.append((d[:, i].mean(), *np.percentile(boot, [2.5, 97.5])))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="headline_current.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    d = build()
    y, key, SPEC = d["y"], d["key"], d["SPEC"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}")
    print(f"{N_SPLITS} splits (the reported protocol uses {SPLITS})\n")
    print(f"{'config':>40}{'precN':>9}{'precPD':>9}{'precET':>9}"
          f"{'macroP':>9}{'macroF1':>9}  |{'  sd':>7}")

    res = {}
    res["base"] = evaluate("welch + desc + asym (baseline)", SPEC["welch"],
                           D, None, y, key, splits=N_SPLITS, return_predictions=True)
    res["mt"] = evaluate("multitaper + desc + asym", SPEC["multitaper"],
                         D, None, y, key, splits=N_SPLITS, return_predictions=True)
    res["mt_t"] = evaluate("multitaper + trajectory (reported)",
                           SPEC["multitaper"], D, TR, y, key, splits=N_SPLITS, return_predictions=True)

    save_predictions(args.output, d, res)

    COMPARISONS = (("mt_t", "base", "THE HEADLINE: reported model vs welch baseline"),
                   ("mt_t", "mt", "trajectory contribution (current pipeline)"),
                   ("mt", "base", "transform contribution alone"))

    for a, b, label in COMPARISONS:
        print(f"\n{label}")
        a_rows, a_ps = res[a]
        b_rows, b_ps = res[b]
        pci = np.percentile(patient_bootstrap(y, b_ps, a_ps, len(y)),
                            [2.5, 97.5], axis=0).T
        for i, ((dd, lo, hi), c) in enumerate(zip(paired(a_rows, b_rows), NM)):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f} split [{lo:+.3f}, {hi:+.3f}] {star}"
                  f" patient [{pci[i, 0]:+.3f}, {pci[i, 1]:+.3f}]")
        wins = [(c, float((a_rows[:, i] > b_rows[:, i]).mean()))
                for i, c in enumerate(NM)]
        print("    split-level win rate: "
              + "  ".join(f"{c} {w:.2f}" for c, w in wins))

    print("\nIntervals condition on fitted models; split resampling describes")
    print("split sensitivity, not independent patient evidence. Neither interval")
    print("accounts for training-sample variation or historical model selection.")
    print(f"Predictions and provenance saved to {args.output}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()

