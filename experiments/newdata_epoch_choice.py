"""NewData's epoch selection is the one cohort-specific transform. Is it the right one?

`verify_data.py` established that the three cohorts are consistent on everything
it can check — no leakage, no duplicate recordings, no id collisions, amplitude
scales within 1.4x, in-band fraction 0.658 / 0.697 / 0.703. **One preprocessing
step is applied to a single cohort**, and this asks whether it is the right one.

NewData's exports are ~38 s `Free_Form` captures whose raw in-band fraction is
**9.9 %** — set-up and settling motion dominate. `load_2025` therefore selects a
10 s window, and offers two criteria:

    select_task_epoch    picks the window with the MOST 3-15 Hz power.
                         Correlated with the thing being classified.
    select_steady_epoch  picks the window where the body-frame GRAVITY direction
                         is steadiest -- how still the limb is held. Never looks
                         at the tremor band, so a healthy control and a tremor
                         patient are selected on the same criterion.

The current default is `tremor`. Its justification, in the loader's own
docstring, is *"better on both axes (N-vs-Tremor 0.787 vs 0.714 at OUT)"* — but
that was measured **on NewData alone, on the binary axis, before the three
preprocessing fixes**. It has never been checked in the merged 3-class model,
which is what the project reports.

## Why the concern is real but not obvious

Selecting the most tremor-dominated window is not label-reading: it uses no
labels. But it is selection on a criterion **correlated with the class**, and
this project has explicitly refused a competitor's number for that reason
(`ceiling_and_preprocessing.md`: *"a 2026 preprint claiming 87 % on PADS uses
class-dependent window overlap — preprocessing that reads the label; not a
comparator"*). Applying a weaker version of the same idea to one of our own
cohorts deserves a measurement rather than a docstring.

The nuance that cuts the other way, and is the reason this is not simply a
defect: selection moves NewData's in-band fraction from 9.9 % to 0.697, which
**matches** 2015 (0.658) and PADS (0.703). Without it NewData would be the
outlier. So the cohort-specific step produces cohort *consistency*.

## Arms

    reported      NewData tremor-selected (segment="tremor")   -- must reproduce build()
    steady        NewData steady-selected (segment="steady")   -- tremor-blind
    none          NewData whole recording (segment=False)      -- what selection buys at all

2015 and PADS are untouched in every arm, so any difference is NewData's epoch
rule and nothing else. NewData is 56 of 404 patients and 6 of 49 ET, so the
merged effect is bounded by how much 14 % of the training set can move.

## Prediction, recorded before the run

**Null on the merged model, with `none` the worst of the three.** NewData is a
seventh of the patients and an eighth of the ET, and the reported model's ET
precision is dominated by PADS. `none` should be worst because a 9.9 % in-band
fraction is mostly set-up motion, which is the one thing here that is clearly
not tremor.

**The comparison that would actually matter is `tremor` vs `steady`**, and the
honest expectation is that they are indistinguishable at this n — in which case
**prefer `steady` on principle**, because a tremor-blind criterion is defensible
in a paper and a tremor-correlated one invites exactly the objection this
project levelled at someone else.

20 splits, paired, checkpointed. Run: ``python -m experiments.newdata_epoch_choice``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import asym_for, desc_table, logbin
from common.protocol import N_ASYM, TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.final_model import TL, method_table
from experiments.pooling_rules import fit_members
from signal_processing.stability import trajectory_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
ARMS = ("reported (tremor)", "steady", "none")
SEG = {"reported (tremor)": "tremor", "steady": "steady", "none": False}


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2],
            float((pred == 2).sum())]


def build_for(segment, rA, rC, keep, nA):
    """The merged tables with NewData loaded under one epoch rule."""
    from common.load_2025 import SIDE, load_2025_all
    from frequency.tables import spectrum_table
    rB = load_2025_all(conditions=("OUT",), segment=segment)
    B0 = spectrum_table(rB, ch=slice(3, 6))
    side_new = lambda r: SIDE.get(__import__("os").path.basename(r.path)[:2])
    side_pads = lambda r: ("left" if "LeftWrist" in str(r.path)
                           else ("right" if "RightWrist" in str(r.path) else None))
    C0 = spectrum_table(rC, ch=slice(0, 3))

    SPEC = logbin(np.vstack([method_table(rA, "multitaper", slice(3, 6))[0],
                             method_table(rB, "multitaper", slice(3, 6))[0],
                             method_table(rC, "multitaper", slice(0, 3))[0][keep]]))
    DESC = np.vstack([desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6)),
                      desc_table(rC, slice(0, 3))[keep]])
    T = np.vstack([trajectory_table(rA, ch=slice(3, 6), n_out=TL)[0],
                   trajectory_table(rB, ch=slice(3, 6), n_out=TL)[0],
                   trajectory_table(rC, ch=slice(0, 3), n_out=TL)[0][keep]])
    aB, hB = asym_for(rB, side_new, slice(3, 6), B0[2])
    aC, hC = asym_for(rC, side_pads, slice(0, 3), C0[2])
    A = np.hstack([np.vstack([np.zeros((nA, N_ASYM)), aB, aC[keep]]),
                   np.concatenate([np.zeros(nA), hB, hC[keep]])[:, None]])
    return SPEC, np.hstack([DESC, A]), T.reshape(len(T), -1), B0[1]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]

    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table
    rA = load_quaternion_recordings("Data", action="OUT",
                                    mode="angular_velocity")
    rC = load_pads_extracted("pads_stretchhold")
    C0 = spectrum_table(rC, ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C0[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    nA = len(spectrum_table(rA, ch=slice(3, 6))[1])

    TAB = {}
    for a in ARMS:
        SPEC, D, TR, yB = build_for(SEG[a], rA, rC, keep, nA)
        assert len(SPEC) == len(y), f"{a}: {len(SPEC)} rows vs {len(y)}"
        TAB[a] = (SPEC, D, TR)
    dev = float(np.abs(TAB["reported (tremor)"][0]
                       - d["SPEC"]["multitaper"]).max())
    print(f"reconstruction check vs build(): max|diff| = {dev:.2e}  "
          f"{'OK' if dev < 1e-9 else 'MISMATCH -- comparisons are invalid'}")
    assert dev < 1e-9, "the reported arm does not reproduce build()"
    print(f"n={len(y)}  splits={SPLITS}")
    print("prediction on record: null on the merged model, 'none' worst;")
    print("if tremor and steady tie, prefer steady on principle\n", flush=True)

    res, done = resume_load("newdata_epoch_choice", ARMS)
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in ARMS:
            SPEC, D, TR = TAB[a]
            V, T = fit_members(SPEC, D, TR, y, tr, va, te)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                y[te]))
        resume_save("newdata_epoch_choice", res, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a]) for a in ARMS}
    print(f"\n{'arm':>20}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>20}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))

    base = R["reported (tremor)"]
    print("\npaired vs the reported (tremor) arm:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for i, ((dd, lo, hi), c) in enumerate(zip(paired(R[a], base), NM)):
            star = "*" if lo > 0 or hi < 0 else " "
            w = float((R[a][:, i] > base[:, i]).mean())
            t = float((R[a][:, i] == base[:, i]).mean())
            print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"   win {w:.2f}  tie {t:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
