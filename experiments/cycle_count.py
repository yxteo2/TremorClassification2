"""Truncate the recordings. The last standing explanation for the slow-patient gap.

`contested_profile.md` found the ceiling's one non-circular handle — **lower
tremor frequency means more contested, in every class**, independent of cohort —
and `low_band_edge.md` confirmed it at 20 splits in a second form:

    contested rate by mean-frequency tercile:  slow 0.515   mid 0.416   fast 0.253

`low_band_edge.md` then refuted the first of the two explanations. Extending the
spectral input down to 2.0 or 1.5 Hz is null on every column and does not touch
the slow tercile, so **band-edge contamination is not the cause**, and the model
ignores the sub-3 Hz region entirely.

That leaves one account, and it is sharper than "fewer periods" once written
down properly.

## The mechanism, stated quantitatively

A recording of length *T* resolves frequency to about Δf = 1/T. What matters for
telling two tremor types apart is not absolute resolution but resolution
**relative to the frequency being measured**:

    Δf / f  =  1 / (f · T)  =  1 / cycles

So *cycles = f · T* is the governing quantity, and a slow tremor in a
fixed-length recording is measured at systematically worse relative precision
than a fast one. At the median 15.5 s recording, a 4 Hz tremor gives 62 cycles
and a 10 Hz tremor gives 155 — a 2.5x difference in relative resolution, matching
the direction and roughly the size of the observed 2.0x contested-rate gradient.

This makes a prediction that band-widening could not: **shortening the recording
should hurt slow patients more than fast ones**, because it moves them further
down the same 1/cycles curve.

## The prediction, recorded before the run

Truncation hurts everyone — less data is less data. The mechanism claim is
specifically **differential**: the slow tercile's contested rate must rise *more*
than the fast tercile's. A uniform rise across terciles refutes cycle count just
as cleanly as the band experiment refuted contamination, and would leave the
frequency gradient without any tested mechanism.

The stronger form, also reported: if cycles is genuinely the governing variable,
then contested rate plotted against *f · T · fraction* should **collapse onto one
curve across truncation arms**. Truncated fast patients should look like
full-length slow patients at matched cycle count. That is a much harder test to
pass by accident than a differential sign.

## Why this cohort can answer it

Recording length already varies about 3x within the 2015 cohort alone (10.5 to
30 s, median 15.5 s), so cycle count varies naturally as well as by
intervention. The truncation arms provide the causal test; the natural variation
provides the collapse test.

Arms: keep the first **100 % / 50 % / 33 %** of every recording, then recompute
spectra exactly as the reported model does. Terciles are fixed once from the
full-length descriptors so the grouping does not move between arms. Everything
else — architecture, seeds, splits, priors, descriptors, trajectory — is
identical.

Run: ``python -m experiments.cycle_count``
"""

from __future__ import annotations

import dataclasses

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
FRACS = (1.0, 0.5, 0.33)
FS = 100.0
MIN_SAMPLES = 256          # below this a 3-15 Hz spectrum is not worth computing


def truncate(recs, frac):
    """First `frac` of every recording, as new Recording objects."""
    if frac >= 1.0:
        return recs
    out = []
    for r in recs:
        n = r.x.shape[-1]
        k = max(int(round(n * frac)), min(MIN_SAMPLES, n))
        out.append(dataclasses.replace(r, x=r.x[..., :k]))
    return out


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def build_at(frac):
    """`FM.build()` with every recording truncated first."""
    real = FM.load_all if hasattr(FM, "load_all") else None
    import common.loaders as CL
    import common.load_2025 as C25
    import common.quaternion_data as CQ

    orig = (CQ.load_quaternion_recordings, C25.load_2025_all,
            CL.load_pads_extracted)

    def wrap(fn):
        def inner(*a, **kw):
            return truncate(fn(*a, **kw), frac)
        return inner

    CQ.load_quaternion_recordings = wrap(orig[0])
    C25.load_2025_all = wrap(orig[1])
    CL.load_pads_extracted = wrap(orig[2])
    try:
        return FM.build()
    finally:
        (CQ.load_quaternion_recordings, C25.load_2025_all,
         CL.load_pads_extracted) = orig


def main():
    torch.set_num_threads(1)

    data = {}
    for f in FRACS:
        print(f"building at {f:.0%} of each recording ...", flush=True)
        data[f] = build_at(f)

    ref = data[FRACS[0]]
    y, key = ref["y"], ref["key"]
    A = np.hstack([ref["ASYM"], ref["HAVE"]])
    desc, traj = ref["DESC"], ref["TRAJ"]
    D = np.hstack([desc, A])
    for f in FRACS[1:]:
        assert np.array_equal(data[f]["y"], y), "label order changed"

    jf = DESCRIPTOR_NAMES.index("mean_freq")
    mf = desc[:, jf]                      # full-length frequencies, fixed
    q1, q2 = np.percentile(mf, [33.3, 66.7])
    terc = np.digitize(mf, [q1, q2])
    print(f"\nmean_freq terciles <{q1:.2f} / {q1:.2f}-{q2:.2f} / >{q2:.2f} Hz"
          f"   n = {[int((terc==t).sum()) for t in (0,1,2)]}")
    print("prediction on record: truncation must raise the SLOW tercile's")
    print("contested rate MORE than the fast tercile's\n", flush=True)

    res = {f: [] for f in FRACS}
    con = {f: {t: [] for t in (0, 1, 2)} for f in FRACS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        line = []
        for f in FRACS:
            V, T = fit_members(data[f]["SPEC"]["multitaper"], D, traj, y,
                               tr, va, te)
            pv, pt = V.mean(0), T.mean(0)
            res[f].append(score(pt, tune_offsets(pv, y[va]), y[te]))
            arg = np.stack([T[i].argmax(1) for i in range(len(T))])
            unan = (arg == arg[0]).all(0)
            for t in (0, 1, 2):
                m = terc[te] == t
                con[f][t].append(float((~unan)[m].mean()) if m.any() else np.nan)
            line.append(f"{f:.0%} slow {con[f][0][-1]:.2f} "
                        f"fast {con[f][2][-1]:.2f}")
        print(f"  split {sp+1}/{SPLITS}  " + " | ".join(line), flush=True)

    for f in res:
        res[f] = np.array(res[f])

    print(f"\n{'kept':>8}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for f in FRACS:
        print(f"{f:>7.0%}" + "".join(f"{v:>9.3f}" for v in res[f].mean(0))
              + f"{res[f][:, 3].std():>12.3f}")

    base = res[FRACS[0]]
    print("\npaired vs full-length recordings:")
    for f in FRACS[1:]:
        print(f"  {f:.0%}:")
        for (dd, lo, hi), c in zip(paired(res[f], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- contested rate by tercile, and the differential:")
    print(f"{'kept':>7}{'slow':>9}{'mid':>9}{'fast':>9}"
          f"{'slow-base':>11}{'fast-base':>11}{'differential':>14}")
    b = [np.nanmean(con[FRACS[0]][t]) for t in (0, 1, 2)]
    for f in FRACS:
        v = [np.nanmean(con[f][t]) for t in (0, 1, 2)]
        ds, df = v[0] - b[0], v[2] - b[2]
        print(f"{f:>6.0%}" + "".join(f"{x:>9.3f}" for x in v)
              + f"{ds:>+11.3f}{df:>+11.3f}{ds-df:>+14.3f}")

    print("\n  differential > 0 supports cycle count; ~0 refutes it and leaves")
    print("  the frequency gradient with no tested mechanism.")

    print("\nTHE COLLAPSE TEST -- contested rate at matched cycle count.")
    print("  cycles ~ mean_freq x fraction (recording length is common to all")
    print("  arms). If cycles governs, rows at similar x should agree.")
    print(f"{'kept':>7}{'tercile':>9}{'f x frac':>11}{'contested':>11}")
    for f in FRACS:
        for t, nmt in ((0, "slow"), (1, "mid"), (2, "fast")):
            xf = float(np.mean(mf[terc == t]) * f)
            print(f"{f:>6.0%}{nmt:>9}{xf:>11.2f}"
                  f"{np.nanmean(con[f][t]):>11.3f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
