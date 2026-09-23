"""What coverage buys 90 % precision per class — the only route to 0.90 that exists.

The request "reach 90 % precision for every category" cannot be met at full
coverage, and the reason is arithmetic rather than pessimism:

  **the model cannot be more accurate than its labels.** `self_consistency_gate.md`
  bounds the label-noise share of errors at ~54 %, implying at most ~19 %
  mislabelling — inside the clinical PD-vs-ET misdiagnosis band of 15-35 %, and
  ET has no gold standard even post-mortem. At 15-19 % wrong labels, 0.90
  per-class precision is unreachable by any architecture.

  **40.5 % of patients are contested and score at chance** (0.443 balanced
  accuracy against a 0.465 constant baseline), and ten methods that only
  reshuffle that set have tied.

  **the largest effect ever measured here is precET +0.104**, the entire
  two-stream headline contribution. Going 0.654 -> 0.90 needs 2.4x that, three
  times over, after 77 experiments found nothing.

What *is* reachable, and is the normal clinical framing for a triage tool, is
high precision with an **abstain** option: answer confidently for some fraction
of patients and refer the rest for specialist review. `metrics/selective.py` was
written for exactly this and has never been run against the reported model.

Two abstention rules, both measured:

    max_prob   abstain when the top class probability is low
    margin     abstain when the top two probabilities are close — better suited
               here, because the failure mode is PD-vs-ET confusion rather than
               uniform uncertainty

## What this reports

For each target coverage, per-class and macro precision over the patients the
model chose to answer for — plus, by inversion, **the coverage at which each
class first reaches 0.90**. That number is the honest answer to the request:
*"90 % precision on class X is available for Y % of patients."*

## Prediction, recorded before the run

**N reaches 0.90 at high coverage; PD and ET need to abstain on most patients,
and ET may not reach 0.90 at any coverage.** N-vs-Tremor already exceeds 0.90 at
full coverage in this project (0.910 / 0.924), so precN has the least distance to
travel. precET starts at 0.654 with ~10 ET patients per test fold, so its
precision moves in steps of ~0.14 — at low coverage it is computed over two or
three patients and becomes uninterpretable before it becomes high.

**The interesting risk is that ET's curve is non-monotone**, because abstention
ranked by confidence removes ET patients faster than it removes the PD patients
they are confused with. If so, say it plainly: the abstain route works for N and
PD and not for ET, which is a more useful finding than a single coverage number.

20 splits, checkpointed. Run: ``python -m experiments.selective_90``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.pooling_rules import fit_members
from metrics.selective import selective_scores

SPLITS = 20
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
RULES = ("margin", "max_prob")
COLS = ("precN", "precPD", "precET", "macroP")


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]
    print(f"n={len(y)}  ET={int((y == 2).sum())}  splits={SPLITS}")
    print("prediction on record: N reaches 0.90 at high coverage; PD and ET need")
    print("heavy abstention; ET may not reach it at all, and its curve may be")
    print("NON-MONOTONE because confidence ranking drops ET faster than the PD")
    print("patients it is confused with.\n", flush=True)

    ARMS = tuple(f"{r}@{c}" for r in RULES for c in COVERAGES)
    res, done = resume_load("selective_90", ARMS)
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        V, T = fit_members(SPEC, D, TR, y, tr, va, te)
        off = tune_offsets(V.mean(0), y[va])
        # apply the tuned priors, then renormalise to a probability vector so the
        # abstention rules see calibrated-ish scores rather than raw logits
        p = np.exp(np.log(T.mean(0) + 1e-12) + off)
        p = p / p.sum(1, keepdims=True)
        for rule in RULES:
            rows = selective_scores(p, y[te], coverages=COVERAGES, rule=rule)
            for cov, row in zip(COVERAGES, rows):
                res[f"{rule}@{cov}"].append(
                    [row["precN"], row["precPD"], row["precET"],
                     row["macroP"], row["n"]] + list(row["n_per_class"]))
        resume_save("selective_90", res, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    print()
    for rule in RULES:
        print(f"=== rule: {rule} ===")
        print(f"{'coverage':>9}{'n':>6}" + "".join(f"{c:>9}" for c in COLS)
              + f"{'ET answered':>13}")
        for cov in COVERAGES:
            a = np.array(res[f"{rule}@{cov}"])
            if not len(a):
                continue
            m = a.mean(0)
            mark = "  <- 0.90 reached" if m[0] >= 0.9 and m[1] >= 0.9 \
                and m[2] >= 0.9 else ""
            print(f"{cov:>9.1f}{m[4]:>6.1f}" + "".join(f"{m[i]:>9.3f}"
                                                       for i in range(4))
                  + f"{m[7]:>13.1f}{mark}")
        print()

    print("first coverage at which EACH class reaches 0.90 (margin rule):")
    for i, c in enumerate(COLS[:3]):
        hit = [cov for cov in COVERAGES
               if len(res[f"margin@{cov}"])
               and np.array(res[f"margin@{cov}"]).mean(0)[i] >= 0.90]
        print(f"  {c:>7}: " + (f"{max(hit):.0%} coverage" if hit
                               else "NOT REACHED at any coverage down to 20 %"))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
