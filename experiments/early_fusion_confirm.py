"""Early fusion vs the reported model at 40 splits, to resolve a borderline gain.

`combined_best.py` measured, on 20 splits against the reported model:

    early channels, linear bins   precET +0.054 [-0.024, +0.131]
                                  macroP +0.021 [-0.006, +0.048]

The point estimates are the best in the project — precET **0.739** and macroP
**0.681** against the reported 0.685 and 0.660 — and neither clears zero. The
same change *is* significant in two matched comparisons: against late
concatenation inside a shared trunk (macroP +0.036 [+0.004, +0.071], `tcn_fusion.md`)
and against the log-binned reported model (macroP +0.047 [+0.013, +0.076]). So
the question is not whether early fusion beats late fusion — it does — but
whether it beats the *reported architecture*, whose `Spectrum1DCNN` trunk
evidently recovers part of what late concatenation costs.

The paired interval over splits narrows as 1/sqrt(splits), and the macroP
half-width at 20 splits is 0.027 against a point estimate of 0.021. Forty splits
should bring the half-width to roughly 0.019 — enough to resolve it either way.
This is estimating the same quantity more precisely, not testing a new one: the
question remains "is C better than A on these 404 patients", which is what the
split-level bootstrap answers (`patient_level_ci.md`).

Two arms only, so the extra splits cost about what four arms cost at 20.

Run: ``python -m experiments.early_fusion_confirm``
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.alltasks_final import evaluate as eval_reported
from experiments.final_model import SPLITS, build
from experiments.tcn_fusion import evaluate as eval_fusion

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
N_SPLITS = 40


def paired_ci(a, b, n=8000):
    d = a - b
    out = []
    for i in range(len(NM)):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        out.append((d[:, i].mean(), *np.percentile(boot, [2.5, 97.5])))
    return out


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {N_SPLITS} splits "
          f"(the reported protocol uses {SPLITS})\n", flush=True)

    print("A: reported model ...", flush=True)
    A = eval_reported(spec, D, traj, y, key, splits=N_SPLITS)
    print(f"{'A reported model':>28}" + "".join(f"{v:>9.3f}"
                                                for v in A.mean(0)), flush=True)
    print("C: early input channels ...", flush=True)
    C = eval_fusion("early", spec, D, traj, y, key, splits=N_SPLITS)
    print(f"{'C early input channels':>28}" + "".join(f"{v:>9.3f}"
                                                      for v in C.mean(0)),
          flush=True)

    print(f"\n{'':>28}" + "".join(f"{c:>9}" for c in NM))
    print(f"{'sd over splits, A':>28}" + "".join(f"{v:>9.3f}" for v in A.std(0)))
    print(f"{'sd over splits, C':>28}" + "".join(f"{v:>9.3f}" for v in C.std(0)))

    print(f"\npaired C - A over {N_SPLITS} shared splits:")
    for (dd, lo, hi), c in zip(paired_ci(C, A), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    # how often does C win outright on a given split?
    print("\nsplit-level win rate for C:")
    for i, c in enumerate(NM):
        w = float((C[:, i] > A[:, i]).mean())
        print(f"    {c:>8} {w:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
