"""Time-frequency parameters for the bag-of-frames model — a search, not a result.

`time_axis_transformer.md` found that frame **order** carries nothing while the
frame **distribution** carries something. Its parameters were inherited from the
time-averaged pipeline and never chosen for a temporal model:

    window (nperseg)  256 = 2.56 s   -> 0.39 Hz frequency resolution per frame
    hop               16  = 0.16 s   -> 46 usable frames
    bins              16             -> already swept elsewhere, held fixed

For a model that **collapses time anyway**, the time-frequency trade-off is a
different question than it was for the time-averaged model, and this is the one
parameter family that has not been swept.

## What is already swept, and is therefore held fixed here

    tf_window_length.md    window length, for the TIME-AVERAGED model
    band_truncation.md     bin count and band coverage
    low_band_edge.md       the 3 Hz low edge
    estimator_smoothing.md estimator sharpness -- a plateau at 40 splits

## The prediction, which makes this a test rather than tuning

The finding says **order is worthless**. If that is right, finer time resolution
buys nothing, so:

> **long windows should win or tie, and hop should be flat.** A 5.12 s window
> gives 0.20 Hz frequency resolution where 1.28 s gives 0.78 Hz — and the
> class contrast lives in peak *position* and *width*, which needs frequency
> resolution. Time detail has already been shown to be discardable.

**If short windows win instead, that contradicts the order-is-worthless
finding** and is far more interesting than a tuning gain — it would mean the
frame distribution is capturing something that needs time localisation after
all. Either outcome is informative, which is the standard `pads_onset_trim.md`
set for an experiment worth running.

## The discipline this sweep is run under

`estimator_smoothing.md`, re-measured today, is the cautionary template: at 20
splits it produced a **significant +0.069 precET** that fell to +0.031 n.s. at 40,
and its five arms turned out to span 0.007 against a 0.025 resolution — a
plateau whose argmax was noise.

So:

1. **One factor at a time**, not a grid. Five configurations, not nine, which
   keeps the multiple-comparison burden honest at precET sd ~0.18.
2. **Report the shape against the resolution**, not the argmax. If the spread
   is inside what 20 splits resolves (~0.04), it is a plateau and the parameter
   does not matter — say that, do not pick a winner.
3. **Any winner is a candidate, not a result**, and must be confirmed at 40
   splits before it is adopted.

Run: ``python -m experiments.time_axis_sweep``
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
from models.architectures import ConvTimeTransformer
from signal_processing.tfd import apply_multitaper
from signal_processing.transforms import F_MAX, _kept_rfftfreq

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS, SEEDS, FS = 20, (0, 1, 2), 100.0

#: (window samples, hop samples). One factor at a time about (256, 16).
CONFIGS = {
    "win 1.28s / hop 0.16s": (128, 16),
    "win 2.56s / hop 0.16s [current]": (256, 16),
    "win 5.12s / hop 0.16s": (512, 16),
    "win 2.56s / hop 0.08s": (256, 8),
    "win 2.56s / hop 0.32s": (256, 32),
}


def surface(x, nperseg, hop):
    """Multitaper POWER surface (n_ch, n_freq, n_time) on the corrected axis."""
    n = min(nperseg, x.shape[-1])
    S = apply_multitaper(x, fs=FS, nperseg=n, nfft=n, noverlap=max(n - hop, 0),
                         f_max=F_MAX)
    n_ch = np.atleast_2d(x).shape[0]
    n_freq = np.asarray(S).shape[0] // n_ch
    f = _kept_rfftfreq(n, FS)
    assert len(f) == n_freq, f"axis {len(f)} != spectrum {n_freq}"
    return np.asarray(S, float).reshape(n_ch, n_freq, -1) ** 2, f


def frame_stack(x, nperseg, hop, n_frames):
    E, f = surface(x, nperseg, hop)
    k = (f >= 3.0) & (f <= FM.GRID[-1])
    P = E.mean(axis=0)[k]
    rows = [np.clip(np.interp(FM.GRID, f[k], P[:, t], left=0.0, right=0.0),
                    0, None) for t in range(P.shape[1])]
    R = np.array([v / (v.sum() + 1e-20) for v in rows])
    if len(R) < n_frames:
        R = np.vstack([R, np.repeat(R[-1:], n_frames - len(R), axis=0)])
    return logbin(R[:n_frames])


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
    (rA, rB, rC), keep = load_cohorts()
    cohorts = ((rA, slice(3, 6), "2015"), (rB, slice(3, 6), "New"),
               (rC, slice(0, 3), "PADS"))
    order = []
    for tag, recs in (("2015", rA), ("New", rB)):
        order += [(tag, s) for s in sorted({r.subject for r in recs})]
    order += [("PADS", s) for s in np.array(sorted({r.subject for r in rC}))[keep]]
    pos = {k: i for i, k in enumerate(order)}
    assert len(order) == len(y)

    # frames available per configuration is set by the SHORTEST recording
    DATA = {}
    print(f"{'configuration':>32}{'frames':>8}{'Hz/bin':>9}")
    for name, (npg, hop) in CONFIGS.items():
        nf = min(1 + max(0, (min(len(r.x[0]) for rs, _, _ in cohorts
                                 for r in rs) - npg)) // hop, 10 ** 6)
        nf = max(int(nf), 4)
        X, r2p = [], []
        for recs, ch, tag in cohorts:
            for r in recs:
                if (tag, r.subject) not in pos:
                    continue
                X.append(frame_stack(r.x[ch] if r.x.shape[0] > 3 else r.x,
                                     npg, hop, nf))
                r2p.append(pos[(tag, r.subject)])
        DATA[name] = (np.asarray(X, np.float32), np.asarray(r2p), nf)
        print(f"{name:>32}{nf:>8}{FS / npg:>9.2f}")
    print(f"\n{SPLITS} splits. Prediction on record: LONG windows win or tie, "
          "hop is flat.\nRead the spread against the ~0.04 that 20 splits "
          "resolves before reading the argmax.\n", flush=True)

    res, done = resume_load("time_axis_sweep", list(CONFIGS))
    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        for name in CONFIGS:
            X, r2p, nf = DATA[name]
            itr, iva, ite = (np.isin(r2p, tr), np.isin(r2p, va),
                             np.isin(r2p, te))
            mu = X[itr].mean(0, keepdims=True)
            sd = X[itr].std(0, keepdims=True) + 1e-8
            mk = (lambda nf=nf: ConvTimeTransformer(NBIN, nf, 3, d=32,
                                                    pos_enc=False))
            r = [train(mk, (X[itr] - mu) / sd, y[r2p[itr]],
                       (X[iva] - mu) / sd, y[r2p[iva]],
                       [(X[iva] - mu) / sd, (X[ite] - mu) / sd], seed=s)
                 for s in SEEDS]
            agg = lambda pr, m, idx: np.array(
                [pr[r2p[m] == p].mean(0) for p in idx])
            pv = agg(np.mean([a[0] for a in r], 0), iva, va)
            pt = agg(np.mean([a[1] for a in r], 0), ite, te)
            res[name].append(score(pt, tune_offsets(pv, y[va]), y[te]))
        resume_save("time_axis_sweep", res, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a]) for a in CONFIGS}
    print(f"\n{'configuration':>32}" + "".join(f"{c:>9}" for c in NM))
    for a in CONFIGS:
        print(f"{a:>32}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))

    cur = "win 2.56s / hop 0.16s [current]"
    print(f"\npaired vs {cur!r}:")
    for a in CONFIGS:
        if a == cur:
            continue
        print(f"  {a}:")
        for i, ((dd, lo, hi), c) in enumerate(zip(paired(R[a], R[cur]), NM)):
            star = "*" if lo > 0 or hi < 0 else " "
            w = float((R[a][:, i] > R[cur][:, i]).mean())
            print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"   win {w:.2f}")

    span = float(max(R[a][:, 3].mean() for a in CONFIGS)
                 - min(R[a][:, 3].mean() for a in CONFIGS))
    print(f"\nmacroP spread across all five configurations: {span:.3f}")
    print(f"  {SPLITS} splits resolves ~0.040 -> "
          + ("INSIDE the resolution: a plateau, the argmax is noise"
             if span < 0.04 else
             "outside the resolution: the ordering may be real, confirm at 40"))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
