"""Short-window spectrum inside the two-stage model — where it should finally pay.

Three results from this session point at one experiment.

1. The short-window spectrum (mean of 0.64 s STFT frames, 16 bins) is better than
   the current multitaper representation on **both binary axes**, measured
   separately with logistic regression:

        PD vs ET      PADS AUC +0.034 * , precET +0.033 * ; MERGED +0.030 * / +0.030 *
        N vs Tremor   AUC 0.774 -> 0.810

   and the binary CNN gains about half that (`tf_window_length.md`).

2. Yet it makes the reported **flat 3-class** model significantly worse
   (macroP −0.033 [−0.057, −0.007] *, losing on 77 % of splits). Four candidate
   mechanisms were tested and all four failed, so the loss is unexplained — but it
   is clearly about the 3-class *combination*, not about the representation, since
   both of the decisions that make up the 3-class problem improve.

3. A two-stage model that makes those decisions separately already exists
   (`axis_specific_inputs.md`). The hierarchy **alone** does nothing
   (macroP −0.008, reproducing the earlier two-stage negative); what helped there
   was giving each stage the input that suits it.

If a representation improves both sub-decisions but hurts their flat combination,
the model that makes the sub-decisions separately should be able to keep the gain.
That is the hypothesis, and it is the first one this session that is a *prediction
from measurements* rather than a story about a mechanism.

Four arms, merged 3-class protocol, 30 splits, composition by the chain rule
P(PD) = P(tremor)·P(PD | tremor) with validation-tuned priors applied to the
composed vector exactly as in the flat model:

  A  flat 3-class, multitaper        the reported model
  B  flat 3-class, short-window      the known loser, re-measured here
  C  two-stage, multitaper           hierarchy alone, no representation change
  D  two-stage, short-window         both changes together

Readings, written down first:

* **D > A** → the gain is recoverable and the flat combination was the obstacle.
* **D ≈ C** → the hierarchy is what matters and the representation is irrelevant
  inside it.
* **D < C** → the short-window loss is not about the 3-class combination at all,
  and hypothesis (2) above is wrong.

30 splits rather than 20, since `early_fusion_confirm.md` showed a paired macroP
near 0.02 evaporating on doubling.

Run: ``python -m experiments.shortwindow_twostage``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import evaluate as eval_flat
from experiments.alltasks_final import paired
from experiments.axis_specific_inputs import fit_stage
from experiments.final_model import build
from experiments.shortwindow_deep import short_window_spectrum
from frequency.tables import spectrum_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 30


def two_stage(spec, desc, traj, y, key, splits=SPLITS):
    """Stage A: N vs Tremor. Stage B: PD vs ET on tremor patients. Chain rule."""
    y_tre = (y != 0).astype(int)
    yb = (y == 2).astype(int)
    out = []
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        trT, vaT = tr[y[tr] != 0], va[y[va] != 0]

        pvA, ptA = fit_stage(spec, desc, traj, y_tre, tr, va, te, nc=2)
        pvB, ptB = fit_stage(spec, desc, traj, yb, trT, vaT, te, nc=2)

        pvB_full = np.full((len(va), 2), 0.5)
        idx = {p: i for i, p in enumerate(vaT)}
        for j, p in enumerate(va):
            if p in idx:
                pvB_full[j] = pvB[idx[p]]

        comp = lambda pA, pB: np.column_stack(
            [pA[:, 0], pA[:, 1] * pB[:, 0], pA[:, 1] * pB[:, 1]])
        pv, pt = comp(pvA, pvB_full), comp(ptA, ptB)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred,
                                                     labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
        print(f"    split {sp+1}/{splits}", flush=True)
    return np.array(out)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    S_mt = d["SPEC"]["multitaper"]

    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    A = spectrum_table(load_quaternion_recordings("Data", action="OUT",
                                                  mode="angular_velocity"),
                       ch=slice(3, 6))
    B_ = spectrum_table(load_2025_all(conditions=("OUT",)), ch=slice(3, 6))
    C = spectrum_table(load_pads_extracted("pads_stretchhold"), ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C[1][keep]]), y), \
        "patient order does not match build()"

    print("building the short-window spectrum ...", flush=True)
    S_sw = np.nan_to_num(short_window_spectrum(order))
    assert S_sw.shape == S_mt.shape
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits\n", flush=True)

    res = {}
    print("A flat 3-class, multitaper ...", flush=True)
    res["A flat, multitaper"] = eval_flat(S_mt, D, traj, y, key, splits=SPLITS)
    print("B flat 3-class, short-window ...", flush=True)
    res["B flat, short-window"] = eval_flat(S_sw, D, traj, y, key, splits=SPLITS)
    print("C two-stage, multitaper ...", flush=True)
    res["C two-stage, multitaper"] = two_stage(S_mt, D, traj, y, key)
    print("D two-stage, short-window ...", flush=True)
    res["D two-stage, short-window"] = two_stage(S_sw, D, traj, y, key)

    print(f"\n{'arm':>28}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for k in res:
        print(f"{k:>28}" + "".join(f"{v:>9.3f}" for v in res[k].mean(0))
              + f"{res[k][:, 3].std():>12.3f}")

    base = res["A flat, multitaper"]
    print(f"\npaired vs A (the reported model), {SPLITS} shared splits:")
    for k in list(res)[1:]:
        print(f"  {k}:")
        for (dd, lo, hi), c in zip(paired(res[k], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\npaired D vs C (representation, inside the hierarchy):")
    for (dd, lo, hi), c in zip(paired(res["D two-stage, short-window"],
                                      res["C two-stage, multitaper"]), NM):
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
