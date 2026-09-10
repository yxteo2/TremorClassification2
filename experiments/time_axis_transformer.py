"""Conv over frequency, transformer over TIME — does the deleted axis carry anything?

Every spectral model here consumes ``P.mean(0)``: the time-frequency surface
averaged over its 49-81 frames **before any model sees it**. So the two standard
objections to a CNN — that it cannot reach long-range structure, and that a
recurrent net forgets the middle — have never applied here, because the sequence
the model is given is **16 frequency bins**, which a 2-layer kernel-5 CNN covers
in one receptive field.

They apply to the *time* axis, and nothing in this project has ever modelled it.
The closest attempts were coarser or aimed elsewhere:

    mil_recordings.md    learned pooling over a patient's RECORDINGS, not over
                         frames within one. Significantly worse than the uniform
                         mean (-0.117, -0.147).
    pcen_hpss.md         HPSS splits the surface into sustained and transient —
                         a two-way split, and class information sits in the
                         sustained half. Adoption null.
    tf_window_length.md  shorter windows, i.e. a different time-frequency
                         trade-off, not a temporal model.

The mechanism worth testing: **amplitude modulation**. Tremor waxes and wanes,
measured in this project at 0.05-5 Hz (`ampmod_features`), and a time-average
destroys that by construction. Burst structure and intermittency are likewise
invisible to `P.mean(0)`.

The dense 0.16 s hop that supplies the frames is already validated as free
(`pcen_hpss.md`: macroP +0.004, null on every column).

## Arms — standalone, so the ensemble cannot hide the effect

    A  Spectrum1DCNN on P.mean(0)          the standard input, time collapsed
    B  ConvTimeTransformer on frames       the same data, time axis kept
    C  ConvTimeTransformer, frames SHUFFLED   the control that decides it

A-vs-B asks whether keeping the time axis helps at all. **B-vs-C is the one that
matters**: if permuting the frames within each recording costs nothing, the
transformer is not using temporal order, and any B-vs-A difference is extra
capacity or extra data rather than time. The permutation is redrawn per split,
per invariant 11 and the fixed-draw incident that once manufactured precET
+0.090 from meaningless columns.

**Every arm works at the RECORDING level and aggregates probabilities to the
patient**, exactly as `window_training.py` does. An earlier draft averaged each
patient's frame stacks together before training, which blurs the temporal
structure being tested across recordings that are not time-aligned — that would
have handicapped arm B and made its null uninformative. Splits stay
patient-disjoint: every recording of a patient sits in one fold.

## Arm D, added after the first run — and why

The first run inverted the control. **B tied A exactly** (macroP 0.608 both), and
**C, with temporal order destroyed, beat both**: macroP +0.027 [+0.005, +0.052] \*
and precET +0.052 [+0.013, +0.093] \* over A.

The reading: scrambling each recording's frames with its own permutation makes
the position encoding carry no consistent information, so the network is forced
to learn a **permutation-invariant, bag-of-frames** function. That beats both the
ordered model *and* the time-average — which says the frame **distribution**
carries something the mean does not, while the frame **order** carries nothing.

If that is right, a model that simply *cannot* see order should match C without
needing a shuffle. **Arm D is `ConvTimeTransformer(pos_enc=False)`**, verified
permutation-invariant to 1.2e-07. It is the principled version of C.

    D ~ C   -> mechanism confirmed; D is what to adopt, being deterministic
    D < C   -> the shuffle is doing something other than removing order
               (regularisation through per-split input noise), and C is not a
               model, it is an augmentation

## Prediction, recorded before the run

**(Run 1) B ties A, and C ties B.** Half right: B tied A exactly, and C did
not tie — it beat both. **(Run 2, for arm D) D matches C to within noise**,
because the mechanism above says the shuffle's only effect is to disable
position, which is exactly what `pos_enc=False` does deliberately.

The original prior is left below unedited. It was unfavourable and worth stating plainly:
four separate attempts to exploit the time axis have been null or worse, and
`window_vs_patient_level.md` measured that *aggregating over time helps* — the
average is a denoiser, and this replaces it with something that must learn the
denoising from 404 patients.

The informative outcome is **C**. If shuffled frames match ordered ones, that is
a direct measurement that **the time axis carries no usable class information at
this n**, which would close the family rather than leave it open — a stronger
result than another null against the baseline.

20 splits, checkpointed. Run: ``python -m experiments.time_axis_transformer``
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import logbin
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pcen_hpss import HOP_DENSE, surface
from models.architectures import ConvTimeTransformer, Spectrum1DCNN

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS, SEEDS = 20, (0, 1, 2)
N_FRAMES = 46           # min across cohorts at hop 0.16 s is 47 (NewData)
ARMS = ("A: CNN on P.mean(0)", "B: conv+time-transformer",
        "C: B, frames SHUFFLED", "D: B, NO position encoding")


def frame_stack(x):
    """(N_FRAMES, NBIN) log-binned spectra, one row per time frame."""
    E, f = surface(x, HOP_DENSE)
    k = (f >= 3.0) & (f <= FM.GRID[-1])
    P = E.mean(axis=0)[k]                      # (n_freq, n_time), axes averaged
    rows = []
    for t in range(P.shape[1]):
        v = np.clip(np.interp(FM.GRID, f[k], P[:, t], left=0.0, right=0.0),
                    0, None)
        rows.append(v / (v.sum() + 1e-20))
    R = np.array(rows)
    if len(R) < N_FRAMES:                       # pad by repeating the last frame
        R = np.vstack([R, np.repeat(R[-1:], N_FRAMES - len(R), axis=0)])
    return logbin(R[:N_FRAMES])


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2],
            float((pred == 2).sum())]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]

    (rA, rB, rC), keep = load_cohorts()
    # per-recording frame stacks, grouped by patient in build()'s row order
    stacks, order = defaultdict(list), []
    for recs, ch, tag in ((rA, slice(3, 6), "2015"), (rB, slice(3, 6), "New"),
                          (rC, slice(0, 3), "PADS")):
        for r in recs:
            stacks[(tag, r.subject)].append(
                frame_stack(r.x[ch] if r.x.shape[0] > 3 else r.x))
    for tag, recs in (("2015", rA), ("New", rB)):
        order += [(tag, s) for s in sorted({r.subject for r in recs})]
    pads_sorted = np.array(sorted({r.subject for r in rC}))
    order += [("PADS", s) for s in pads_sorted[keep]]
    assert len(order) == len(y), f"{len(order)} patients vs {len(y)}"

    pos = {k: i for i, k in enumerate(order)}
    Xrec, rec2pat = [], []
    for k, v in stacks.items():
        if k not in pos:                       # a PADS patient the cap dropped
            continue
        for st in v:
            Xrec.append(st)
            rec2pat.append(pos[k])
    X = np.asarray(Xrec, np.float32)           # (n_recordings, frames, bins)
    rec2pat = np.asarray(rec2pat)
    Aflat = X.mean(1)                          # time collapsed, same recordings
    print(f"n={len(y)} patients, {len(X)} recordings; stack {X.shape[1:]} "
          f"(frames x bins)   splits={SPLITS}")
    print("prediction on record: B ties A, and C ties B; C is the informative "
          "arm\n", flush=True)

    # Per-arm checkpoints. A single shared file is discarded whenever the arm
    # list changes, so adding arm D would have thrown away A/B/C -- and the
    # container reset that killed the 4-arm run would then have cost all four.
    # `attention_test.py` already checkpoints this way; this follows it.
    res, done = {}, {}
    for a in ARMS:
        r, dn = resume_load("time_axis_" + a.split(":")[0], (a,))
        res[a], done[a] = r[a], dn

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        if all(sp in done[a] for a in ARMS):
            continue
        rng = np.random.default_rng(7000 + sp)
        Xs = X.copy()
        for i in range(len(Xs)):                # permute frames, redrawn/split
            Xs[i] = Xs[i][rng.permutation(X.shape[1])]

        itr = np.isin(rec2pat, tr)
        iva = np.isin(rec2pat, va)
        ite = np.isin(rec2pat, te)
        agg = lambda pr, m, idx: np.array(
            [pr[rec2pat[m] == p].mean(0) for p in idx])

        for arm in ARMS:
            if sp in done[arm]:
                continue
            if arm.startswith("A"):
                Z, mk = Aflat, (lambda: Spectrum1DCNN(NBIN, 3, ch=8))
            elif arm.startswith("D"):
                Z = X
                mk = (lambda: ConvTimeTransformer(NBIN, N_FRAMES, 3, d=32,
                                                  pos_enc=False))
            else:
                Z = X if arm.startswith("B") else Xs
                mk = (lambda: ConvTimeTransformer(NBIN, N_FRAMES, 3, d=32))
            mu = Z[itr].mean(0, keepdims=True)
            sd = Z[itr].std(0, keepdims=True) + 1e-8
            r = [train(mk, (Z[itr] - mu) / sd, y[rec2pat[itr]],
                       (Z[iva] - mu) / sd, y[rec2pat[iva]],
                       [(Z[iva] - mu) / sd, (Z[ite] - mu) / sd], seed=s)
                 for s in SEEDS]
            pv = agg(np.mean([a[0] for a in r], 0), iva, va)
            pt = agg(np.mean([a[1] for a in r], 0), ite, te)
            res[arm].append(score(pt, tune_offsets(pv, y[va]), y[te]))
            resume_save("time_axis_" + arm.split(":")[0], {arm: res[arm]}, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a]) for a in ARMS}
    print(f"\n{'arm':>26}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))

    for base, tag in ((ARMS[0], "vs A (does keeping time help?)"),
                      (ARMS[1], "vs B (IS IT TEMPORAL ORDER? the deciding one)")):
        print(f"\npaired {tag}:")
        for a in ARMS:
            if a == base:
                continue
            print(f"  {a}:")
            for i, ((dd, lo, hi), c) in enumerate(zip(paired(R[a], R[base]), NM)):
                star = "*" if lo > 0 or hi < 0 else " "
                w = float((R[a][:, i] > R[base][:, i]).mean())
                print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                      f"   win {w:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
