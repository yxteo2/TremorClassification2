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

    rows, by_class, by_coh = [], {}, {}
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

        # Where does the disagreement sit, and is it resolvable? Split the test
        # fold into patients the six members label unanimously and patients they
        # do not, and score the pooled prediction on each. If the contested
        # patients are near-chance, no pooling rule can win them and every rule
        # must tie -- which is the alternative explanation for the pooling null.
        A = np.stack(arg)
        unan = (A == A[0]).all(0)
        pooled = np.mean(T, 0)
        pred = pooled.argmax(1)
        acc_u = float((pred[unan] == y[te][unan]).mean()) if unan.any() else np.nan
        acc_c = float((pred[~unan] == y[te][~unan]).mean()) if (~unan).any() \
            else np.nan
        # margin between top-2 pooled probabilities: how close to the boundary?
        s = np.sort(pooled, 1)
        marg = s[:, -1] - s[:, -2]
        m_u = float(marg[unan].mean()) if unan.any() else np.nan
        m_c = float(marg[~unan].mean()) if (~unan).any() else np.nan

        rows.append([all_r, within, across, all_d, wd, ad, sd_et,
                     float(unan.mean()), acc_u, acc_c, m_u, m_c])

        # Are the contested patients concentrated anywhere? If they cluster in a
        # class or a cohort that is a handle; if they are spread evenly the
        # ceiling is intrinsic to the task at this sample size.
        for c in (0, 1, 2):
            m = y[te] == c
            by_class.setdefault(c, []).append(
                [float((~unan)[m].mean()),
                 float((pred[m & ~unan] == c).mean()) if (m & ~unan).any()
                 else np.nan])
        for cname in ("2015", "NewData", "PADS"):
            m = np.array([k.rsplit("_", 1)[0] == cname for k in key[te]])
            by_coh.setdefault(cname, []).append(
                float((~unan)[m].mean()) if m.any() else np.nan)
        print(f"  split {sp+1}/{SPLITS}  r(pET) all {all_r:.3f} "
              f"(within-family {within:.3f}, across {across:.3f})   "
              f"disagree {all_d:.3f}", flush=True)

    a = np.array(rows)
    lab = ("r(pET) all pairs", "r within family", "r across families",
           "disagreement all", "disagreement within", "disagreement across",
           "mean sd of p(ET)", "fraction unanimous", "accuracy | unanimous",
           "accuracy | contested", "top-2 margin | unanim", "top-2 margin | contest")
    print("\n" + "-" * 62)
    for i, l in enumerate(lab):
        print(f"{l:>26}  {a[:, i].mean():.3f}  (sd {a[:, i].std():.3f})")

    print("\ncontested rate and recall-on-contested, by class:")
    for c, nmc in ((0, "N"), (1, "PD"), (2, "ET")):
        v = np.array(by_class[c], float)
        print(f"  {nmc:>3}  contested {np.nanmean(v[:, 0]):.3f}   "
              f"correct when contested {np.nanmean(v[:, 1]):.3f}")
    print("\ncontested rate by cohort:")
    for cname in ("2015", "NewData", "PADS"):
        print(f"  {cname:>8}  {np.nanmean(by_coh[cname]):.3f}")

    print("\nreading it:")
    print("  High r with near-zero disagreement would mean the members are")
    print("  copies, and no pooling rule could distinguish itself. If instead")
    print("  they disagree substantially, the pooling null in pooling_rules.md")
    print("  needs the other explanation: the contested patients are near")
    print("  chance, so reordering them changes WHICH errors are made, not how")
    print("  many. The accuracy-on-contested row is what decides between the two.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
