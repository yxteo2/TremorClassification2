"""How redundant are the six ensemble members? The number that explains two results.

`pooling_rules.md` found that **no combination rule beats the arithmetic mean** —
geometric, median, trimmed and temperature-scaled pooling all land within
±0.012 macro precision of it, every interval spanning zero. That is a surprising
result on its face: geometric pooling is genuinely more conservative than
arithmetic, and conservatism is supposed to buy precision.

The obvious explanation is that **the members barely disagree**. The reported
ensemble is 2 families x 3 seeds, and the three seeds within a family differ only
in weight initialisation — same architecture, same features, same training data,
same early-stopping split. If their outputs are nearly identical, then every
pooling rule is averaging over almost nothing and they must all coincide,
whatever their theoretical differences.

That is an explanation, not a measurement, so this measures it.

Reported per split, on the test fold, over the six members:

  mean pairwise Pearson r of p(ET)     1.0 means the members are copies
  mean pairwise disagreement rate      fraction of patients where argmax differs
  within-family vs across-family       do the two architectures see the same thing?
  spread of p(ET) across members       the raw quantity every pooling rule acts on

## Why the answer matters beyond one report

If the members are near-copies, then **ensemble diversity is unexploited
headroom**, and the way to get at it is to make the members differ in their
*data*, not their initialisation — which is exactly what `balanced_bagging.py`
tests. A high correlation here predicts that bagging should help; a low one
predicts it should not, and would instead say the ensemble is already as diverse
as this data allows.

Either way the number is worth having, because "we averaged three seeds" is a
claim about variance reduction that nobody here has ever checked.

No labels are used for anything but scoring the diagnostic. This trains the same
six members as the reported model and measures them; it changes nothing.

Run: ``python -m experiments.ensemble_diversity``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC
from experiments.final_model import build
from experiments.pooling_rules import fit_members

SPLITS = 10


def pairwise(M, fn):
    """Mean of fn over all unordered member pairs."""
    v = [fn(M[i], M[j]) for i in range(len(M)) for j in range(i + 1, len(M))]
    return float(np.mean(v)) if v else np.nan


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]

    print(f"n={len(y)}  {SPLITS} splits, 6 members each "
          f"(2 families x 3 seeds)\n", flush=True)

    rows = []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(spec, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(spec[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        _, T = fit_members(spec, D, traj, y, tr, va, te)     # (6, n_te, 3)

        et = [T[i][:, 2] for i in range(6)]
        arg = [T[i].argmax(1) for i in range(6)]
        r = lambda a, b: np.corrcoef(a, b)[0, 1]
        dis = lambda a, b: float((a != b).mean())

        all_r = pairwise(et, r)
        all_d = pairwise(arg, dis)
        within = np.mean([pairwise(et[:3], r), pairwise(et[3:], r)])
        across = float(np.mean([r(et[i], et[j]) for i in range(3)
                                for j in range(3, 6)]))
        wd = np.mean([pairwise(arg[:3], dis), pairwise(arg[3:], dis)])
        ad = float(np.mean([dis(arg[i], arg[j]) for i in range(3)
                            for j in range(3, 6)]))
        sd_et = float(np.std(np.stack(et), 0).mean())

        rows.append([all_r, within, across, all_d, wd, ad, sd_et])
        print(f"  split {sp+1}/{SPLITS}  r(pET) all {all_r:.3f} "
              f"(within-family {within:.3f}, across {across:.3f})   "
              f"disagree {all_d:.3f}", flush=True)

    a = np.array(rows)
    lab = ("r(pET) all pairs", "r within family", "r across families",
           "disagreement all", "disagreement within", "disagreement across",
           "mean sd of p(ET)")
    print("\n" + "-" * 58)
    for i, l in enumerate(lab):
        print(f"{l:>24}  {a[:, i].mean():.3f}  (sd {a[:, i].std():.3f})")

    print("\nreading it:")
    print("  r near 1.00 and disagreement near 0 -> the members are copies and")
    print("  no pooling rule can distinguish itself, which is what")
    print("  pooling_rules.md measured. Diversity would then have to come from")
    print("  the DATA the members see, not their initialisation.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
