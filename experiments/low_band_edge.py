"""Move the low band edge. A representation change aimed at a measured mechanism.

`contested_profile.md` produced the one non-circular handle on the ceiling this
project has found. Contestedness — whether the six ensemble members agree — is
predictable from spectral descriptors at AUC 0.725, and while most of that signal
is circular (descriptors that flip sign between N and ET are just restating the
decision boundary), the **frequency-location descriptors do not flip**:

    mean_freq     rho | N -0.385   PD -0.241   ET -0.051   mean within -0.226
    median_freq              -0.315      -0.251      -0.050              -0.205
    max_freq                 -0.072      -0.174      -0.216              -0.154

**Lower tremor frequency means more contested, in every class.** That cannot be
class confusion, which produces opposing signs. It is not the cohort effect
either: mean frequency is 7.714 / 7.791 / 7.706 Hz across the three cohorts,
identical to within 1 %, while contested rate spans 0.356 to 0.498.

Two physical accounts were recorded there, and this tests the first:

* **Band-edge contamination.** The analysis grid starts at 3.0 Hz
  (`final_model.GRID = linspace(3, 15, 64)`), and everything below is discarded
  by interpolation with `left=0.0`. Voluntary movement and postural drift live
  there. A 4 Hz tremor sits one octave from that edge; a 9 Hz tremor sits far
  from it. If the model is losing slow patients because their signal is
  entangled with drift right at the boundary, then **giving it the sub-3 Hz
  region to see** should help slow patients specifically — the network can
  learn to discount drift only if drift is in its input.
* **Cycle count**, tested separately: a slow oscillation completes fewer periods
  in a fixed recording. Extending the band does nothing for that, so a null here
  is evidence for the cycle-count account by elimination.

## The prediction, recorded before the run

**Extending the low edge should cut the contested rate for low-frequency patients
more than for high-frequency ones.** The report stratifies by tercile of patient
mean frequency and reports the change in each. A uniform change across terciles —
or a gain with no movement in the contested rates — means the mechanism is wrong
even if precision improves, exactly as in `cohort_id_input.py`.

There is a real chance this makes things **worse**, and that outcome is
informative too: the sub-3 Hz region is mostly drift, the grid keeps 64 points
whatever the span, so widening it *coarsens* resolution inside the tremor band
proper. If precision drops, the band edge is where it is for a reason and that is
worth knowing explicitly rather than by assumption.

## What is held fixed

Only the **spectral input to the deep model** changes. The ten hand-built
descriptors keep their own 3–15 Hz band, the trajectory stream is untouched, and
the architecture, seeds, splits and priors are identical. The arms therefore
differ in one thing.

`final_model.GRID` is a module-level constant consumed by `method_table`, so each
arm is produced by setting it and calling `build()` — no edit to the reported
model's code, and the 3.0 Hz arm reproduces it exactly.

Arms: low edge at **3.0 (current), 2.0, 1.5 Hz**; high edge 15 Hz and 64 grid
points throughout. 20 splits, paired.

Run: ``python -m experiments.low_band_edge``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.pooling_rules import fit_members
from frequency.descriptors import DESCRIPTOR_NAMES

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
EDGES = (3.0, 2.0, 1.5)
N_GRID, F_HI = 64, 15.0


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)

    data = {}
    for e in EDGES:
        FM.GRID = np.linspace(e, F_HI, N_GRID)
        print(f"building spectra with band {e}-{F_HI} Hz ...", flush=True)
        data[e] = FM.build()

    ref = data[EDGES[0]]
    y, key = ref["y"], ref["key"]
    A = np.hstack([ref["ASYM"], ref["HAVE"]])
    desc, traj = ref["DESC"], ref["TRAJ"]          # held fixed across arms
    D = np.hstack([desc, A])

    # sanity: everything except the spectrum must be identical across arms
    for e in EDGES[1:]:
        assert np.array_equal(data[e]["y"], y), "label order changed"
        assert np.allclose(data[e]["DESC"], desc), "DESC changed with the grid"

    jf = DESCRIPTOR_NAMES.index("mean_freq")
    mf = desc[:, jf]
    q1, q2 = np.percentile(mf, [33.3, 66.7])
    terc = np.digitize(mf, [q1, q2])               # 0 slow, 1 mid, 2 fast
    print(f"\nmean_freq terciles: <{q1:.2f} / {q1:.2f}-{q2:.2f} / >{q2:.2f} Hz"
          f"   n = {[int((terc==t).sum()) for t in (0,1,2)]}")
    print("prediction on record: the low edge should cut the contested rate")
    print("for the SLOW tercile more than for the fast one\n", flush=True)

    res = {e: [] for e in EDGES}
    con = {e: {t: [] for t in (0, 1, 2)} for e in EDGES}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        line = []
        for e in EDGES:
            spec = data[e]["SPEC"]["multitaper"]
            V, T = fit_members(spec, D, traj, y, tr, va, te)
            pv, pt = V.mean(0), T.mean(0)
            res[e].append(score(pt, tune_offsets(pv, y[va]), y[te]))
            arg = np.stack([T[i].argmax(1) for i in range(len(T))])
            unan = (arg == arg[0]).all(0)
            for t in (0, 1, 2):
                m = terc[te] == t
                con[e][t].append(float((~unan)[m].mean()) if m.any() else np.nan)
            line.append(f"{e}Hz slow {con[e][0][-1]:.2f}")
        print(f"  split {sp+1}/{SPLITS}  " + " | ".join(line), flush=True)

    for e in res:
        res[e] = np.array(res[e])

    print(f"\n{'low edge':>10}" + "".join(f"{c:>9}" for c in NM)
          + "   sd(macroP)")
    for e in EDGES:
        print(f"{e:>8} Hz" + "".join(f"{v:>9.3f}" for v in res[e].mean(0))
              + f"{res[e][:, 3].std():>12.3f}")

    base = res[EDGES[0]]
    print(f"\npaired vs the current {EDGES[0]} Hz edge:")
    for e in EDGES[1:]:
        print(f"  {e} Hz:")
        for (dd, lo, hi), c in zip(paired(res[e], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- contested rate by mean-frequency tercile:")
    print(f"{'edge':>8}{'slow':>9}{'mid':>9}{'fast':>9}"
          f"{'slow-vs-base':>14}{'fast-vs-base':>14}")
    b = [np.nanmean(con[EDGES[0]][t]) for t in (0, 1, 2)]
    for e in EDGES:
        v = [np.nanmean(con[e][t]) for t in (0, 1, 2)]
        print(f"{e:>6} Hz" + "".join(f"{x:>9.3f}" for x in v)
              + f"{v[0]-b[0]:>+14.3f}{v[2]-b[2]:>+14.3f}")
    print("\nthe mechanism holds only if the slow column falls by MORE than")
    print("the fast column. Precision moving without that is the mechanism")
    print("being wrong even if the number goes up.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
