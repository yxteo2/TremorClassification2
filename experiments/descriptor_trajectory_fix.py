"""Two preprocessing defects found by synthetic verification — fixed — paired A/B.

`experiments/verify_preprocessing.py` checks every stage against signals with a
known answer. Two stages that feed the reported model failed:

**1. `describe()`'s Q-factor was not the peak's half-power width.** It took the
span of *every* bin above half-maximum anywhere in the band. A 6 Hz tone with a
0.8-amplitude 12 Hz harmonic therefore read Q 0.94 instead of 15.0. On real
recordings the supra-half set is non-contiguous for **85 % of PADS N, 74 % of
PD and 30 % of ET** (stft512), so the old `q_factor` — one of the ten `DESC`
inputs to `TwoStreamNet` — was measuring "has secondary spectral content" as
much as peak sharpness, and doing so in a class-ordered way. Fixed to walk the
contiguous half-power region around the peak (`frequency/descriptors.py`,
switchable via `Q_CONTIGUOUS`).

**2. The IF trajectory's end points were filter transients.** The band-pass /
Hilbert chain gets the instantaneous frequency wrong for the first and last
10–16 samples, and resampling to 64 points maps the raw ends onto trajectory
points 0 and 63 exactly. A rock-steady 6 Hz tone read 0.36 Hz of wander at point
0; a 6 ± 0.5 Hz FM tone read 2.7 Hz at point 63, against an interior correct to
0.06 Hz. So 2 of 64 points per channel, in every patient's `TRAJ` input, were
noise of larger magnitude than the signal. Fixed with a 0.25 s guard band — the
settling scale of the 4 Hz-wide 4th-order filter (`signal_processing/stability.py`,
`guard_s`, 0 reproduces the old path).

Both fixes are correct on their own terms. This measures what they do to the
reported model, because the model was fitted and evaluated on the defective
inputs consistently and may simply have learned around them.

## Predictions, recorded before the run

Both are measurement-derived, not mechanism stories.

**TRAJ guard: null.** The corrupted points are the same two positions for every
patient, and the transient's magnitude depends on the recording's own edge, not
on class. A network sees two noisy features out of 128; removing them should
change little. Predicted |ΔmacroP| < 0.01.

**Q fix: small, sign uncertain, and possibly slightly negative.** The old
`q_factor` was class-ordered *because* secondary spectral content is itself
PD-like (Häring's "several discrete oscillator states"). The fix removes a
mislabelled feature that nevertheless carried real class information, and
replaces it with the quantity it claimed to be. That could cost a little. The
honest claim is only that the descriptor now means what its name says; whether
the model preferred the mislabelled version is what this measures.

## Arms

  old DESC, old TRAJ   the pre-fix reported model, reconstructed
  new DESC, old TRAJ   Q fix alone
  old DESC, new TRAJ   guard alone
  new DESC, new TRAJ   the current defaults -- asserted bit-exact against build()

Everything else is identical. The spectrum is the corrected-axis multitaper in
every arm. 20 splits, paired.

Run: ``python -m experiments.descriptor_trajectory_fix``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import frequency.descriptors as FD
import experiments.final_model as FM
from common.cohorts import desc_table
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pooling_rules import fit_members
from frequency.descriptors import DESCRIPTOR_NAMES
from signal_processing.stability import trajectory_table

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20


def build_desc(recs, keep, contiguous):
    old = FD.Q_CONTIGUOUS
    FD.Q_CONTIGUOUS = contiguous
    try:
        rA, rB, rC = recs
        return np.vstack([desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6)),
                          desc_table(rC, slice(0, 3))[keep]])
    finally:
        FD.Q_CONTIGUOUS = old


def build_traj(recs, keep, guard_s):
    rA, rB, rC = recs
    T = np.vstack([trajectory_table(rA, ch=slice(3, 6), n_out=FM.TL, guard_s=guard_s)[0],
                   trajectory_table(rB, ch=slice(3, 6), n_out=FM.TL, guard_s=guard_s)[0],
                   trajectory_table(rC, ch=slice(0, 3), n_out=FM.TL, guard_s=guard_s)[0][keep]])
    return T.reshape(len(T), -1)


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    A = np.hstack([d["ASYM"], d["HAVE"]])
    spec = d["SPEC"]["multitaper"]
    recs, keep = load_cohorts()

    print("building DESC old / new and TRAJ old / new ...", flush=True)
    DESC = {"old": build_desc(recs, keep, False), "new": build_desc(recs, keep, True)}
    TRAJ = {"old": build_traj(recs, keep, 0.0), "new": build_traj(recs, keep, 0.25)}
    for nm, X, ref in (("DESC", DESC["new"], d["DESC"]), ("TRAJ", TRAJ["new"], d["TRAJ"])):
        dev = float(np.abs(X - ref).max())
        print(f"  new {nm} vs build(): max|diff| = {dev:.2e} {'OK' if dev < 1e-5 else 'MISMATCH'}")
        assert dev < 1e-5, f"{nm} reconstruction does not match build()"
    jq = DESCRIPTOR_NAMES.index("q_factor")
    print("\nq_factor by class, old -> new (patient means):")
    for k, c in ((0, "N"), (1, "PD"), (2, "ET")):
        print(f"  {c:>3}  {DESC['old'][y == k, jq].mean():6.2f} -> {DESC['new'][y == k, jq].mean():6.2f}")
    changed = float(np.mean(np.abs(TRAJ["new"] - TRAJ["old"]).reshape(len(y), -1).max(1) > 1e-6))
    print(f"TRAJ rows changed by the guard: {changed:.2f} of patients; "
          f"max |diff| {np.abs(TRAJ['new'] - TRAJ['old']).max():.3f}\n", flush=True)

    ARMS = {"old DESC, old TRAJ (pre-fix)": ("old", "old"),
            "new DESC, old TRAJ (Q fix)": ("new", "old"),
            "old DESC, new TRAJ (guard)": ("old", "new"),
            "new DESC, new TRAJ (current)": ("new", "new")}
    res = {a: [] for a in ARMS}
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None], key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a, (dk, tk) in ARMS.items():
            D = np.hstack([DESC[dk], A])
            V, T = fit_members(spec, D, TRAJ[tk], y, tr, va, te)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]), y[te]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)
    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>30}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>30}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")
    base = res["old DESC, old TRAJ (pre-fix)"]
    print("\npaired vs the pre-fix model:")
    for a in list(ARMS)[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
