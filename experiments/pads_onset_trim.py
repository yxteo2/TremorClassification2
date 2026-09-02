"""PADS carries an untrimmed arm-raising onset, and it is class-ordered.

`extract_pads.py` has a ``--trim-start`` option for "the arm-raising onset" and
its default is **0.0 s**, on the strength of one early, unpaired, PADS-only
ET-F1 comparison (0.26 -> 0.20 on 28 ET patients). That was never re-tested
under the current protocol. This does, because the onset turns out to be real,
PADS-specific, and correlated with class.

## What is in the data, measured

Robust-z outliers (>10 MAD, any axis) in PADS StretchHold sit almost entirely at
the start of the recording, and they are multi-sample events, not sensor
glitches:

    where:        N  start 0.98   PD  start 0.89   ET  start 0.92
    run length:   median 3 samples, 24 % of runs >= 5 samples, max 221 (2.2 s)

In absolute terms — in-band (3-15 Hz) RMS of the first 1.5 s over the remainder:

    cohort   class   median ratio   fraction > 2x
    PADS     N        1.39            0.31
    PADS     PD       1.33            0.18
    PADS     ET       1.06            0.11
    2015     all     ~0.95            0.00-0.02
    NewData  all     ~1.00            0.05-0.25 (n small)

So: **only PADS has an onset excess, and within PADS it is ordered N > PD > ET.**
Healthy controls raise the arm against almost no tremor, so the transient
dominates their first second; ET patients tremble from the moment the posture is
adopted, so the transient barely registers against it. The onset therefore adds
in-band power *preferentially to N and PD*, giving them a broadband signature
that ET lacks.

## Why this is a PD-vs-ET problem and not just noise

That signature is **class-correlated in the direction that helps the classifier
on PADS** — which is exactly why removing it "did not help" PADS-only ET-F1. The
harm shows up where the artifact is absent: on in-house cohorts, whose
recordings have no onset. This project has already measured that pattern
without a mechanism for it — *"PADS does not transfer to in-house patients — it
adds nothing to ET (+0.003) and significantly hurts PD (−0.082)"*. A
PADS-specific, class-correlated artifact is a candidate explanation, and it is
the kind of preprocessing defect a mixed-cohort headline cannot see because
both arms of every paired comparison share it.

## Arms

  untrimmed          the current extraction (first 1.5 s kept)
  trim-start 1.5 s   drop the first 150 samples of every PADS recording
  trim-end 1.5 s     drop the LAST 150 samples instead — same length, same
                     number of multitaper frames, onset left in. This is the
                     control that separates "onset removed" from "shorter
                     recording".

Only PADS is modified. 2015 and NewData are untouched in every arm.

## Three measurements, in the order they decide things

**(A) Mechanism.** After trim-start, the N > PD > ET onset ordering in the RMS
ratio should collapse toward 1.0 for all classes, and per-class peak sharpness
should rise most for N and PD. If it does not, the trim missed the onset and (B)
and (C) are uninformative.

**(B) Mixed-cohort headline**, 20 splits, paired. Predicted **small and of
uncertain sign**: the artifact helps within PADS, PADS is about half the cohort,
and the effect is a fraction of a bin's worth of broadband power. Do not read
this arm as the verdict.

**(C) PADS -> in-house transfer**, PD vs ET: train a logistic regression on
PADS patients only, test on every in-house PD and ET patient, AUC on p(ET), paired
subject-level bootstrap across arms. **This is the arm the mechanism actually
predicts on**: trim-start should raise transfer AUC, trim-end should not.
Read it against the in-house detection floor of **AUC 0.655**
(`permutation_null.md`); anything below that is indistinguishable from chance.

## A limitation stated up front

The descriptor and trajectory streams (`DESC`, `TRAJ`) are taken from `build()`
and therefore from **untrimmed** recordings in every arm; only the spectral
stream is trimmed. For (B) that leaves part of the onset in the model in all
arms, which biases (B) toward null — any gain there is conservative. (C) uses
the spectrum alone and is fully trimmed.

Run: ``python -m experiments.pads_onset_trim``
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts, spec_for
from experiments.pooling_rules import fit_members
from frequency.descriptors import describe
from signal_processing.transforms import METHODS

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SPLITS = 20
FS, TRIM_S = 100.0, 1.5
FLOOR = 0.655                                   # in-house PD-vs-ET null, 21 ET
NBOOT = 2000


def trim(recs, where):
    k = int(TRIM_S * FS)
    if where == "none":
        return recs
    return [dataclasses.replace(r, x=(r.x[..., k:] if where == "start"
                                      else r.x[..., :-k])) for r in recs]


def onset_ratio(x):
    k = int(TRIM_S * FS)
    def rms(seg):
        f, P = welch(seg, fs=FS, nperseg=min(128, seg.shape[-1]), axis=-1)
        P = P.mean(0)
        return float(np.sqrt(P[(f >= 3) & (f <= 15)].sum()))
    return rms(x[:, :k]) / (rms(x[:, k:]) + 1e-12)


def mechanism(rC, label):
    """Per-class onset ratio and peak Q for one PADS arm."""
    rat, q = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
    for r in rC:
        rat[r.y].append(onset_ratio(r.x))
        f, P = METHODS["multitaper"](r.x)
        q[r.y].append(describe(f, P)["q_factor"])
    print(f"  {label:>16}  " + "   ".join(
        f"{c}: onset {np.median(rat[k]):.2f} Q {np.mean(q[k]):.2f}"
        for k, c in ((0, "N"), (1, "PD"), (2, "ET"))))


def score(pt, off, yte):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()]


def transfer_auc(S, y, idx_pads, idx_inh):
    """PD-vs-ET: fit on PADS, score every in-house PD/ET patient. Returns p."""
    tr = idx_pads[np.isin(y[idx_pads], [1, 2])]
    te = idx_inh[np.isin(y[idx_inh], [1, 2])]
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=5000, class_weight="balanced"))
    m.fit(S[tr], (y[tr] == 2).astype(int))
    return te, m.predict_proba(S[te])[:, 1]


def main():
    torch.set_num_threads(1)
    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]
    (rA, rB, rC), keep = load_cohorts()
    nA = len({r.subject for r in rA})
    nB = len({r.subject for r in rB})
    idx_inh = np.arange(nA + nB)
    idx_pads = np.arange(nA + nB, len(y))
    assert len(idx_pads) == len(keep)

    ARMS = {"untrimmed": "none", "trim-start 1.5 s": "start",
            "trim-end 1.5 s (control)": "end"}

    print("(A) MECHANISM -- PADS per-class onset ratio (first 1.5 s / rest) "
          "and peak Q:")
    RC = {a: trim(rC, w) for a, w in ARMS.items()}
    for a in ARMS:
        mechanism(RC[a], a)
    print("  prediction: trim-start collapses the N > PD > ET onset ordering "
          "toward 1.0; trim-end does not\n", flush=True)

    SPEC = {}
    for a in ARMS:
        print(f"building spectra: {a} ...", flush=True)
        S = spec_for(METHODS["multitaper"], (rA, rB, RC[a]), keep)
        assert len(S) == len(y)
        SPEC[a] = S
    dev = float(np.abs(SPEC["untrimmed"] - d["SPEC"]["multitaper"]).max())
    print(f"\nuntrimmed arm vs build(): max|diff| = {dev:.2e} "
          f"{'OK' if dev < 1e-6 else 'MISMATCH'}")
    assert dev < 1e-6

    # ---------------- (C) transfer, computed first: it is the verdict --------
    print("\n(C) TRANSFER -- PADS -> in-house PD-vs-ET AUC on the spectrum "
          f"(floor {FLOOR}):")
    P = {}
    for a in ARMS:
        te, p = transfer_auc(SPEC[a], y, idx_pads, idx_inh)
        P[a] = p
    yt = (y[te] == 2).astype(int)
    rng = np.random.default_rng(0)
    boots = {a: [] for a in ARMS}
    for _ in range(NBOOT):
        i = rng.integers(0, len(te), len(te))
        if yt[i].min() == yt[i].max():
            continue
        for a in ARMS:
            boots[a].append(roc_auc_score(yt[i], P[a][i]))
    for a in ARMS:
        b = np.array(boots[a])
        print(f"  {a:>24}  AUC {roc_auc_score(yt, P[a]):.3f}  "
              f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
    for a in list(ARMS)[1:]:
        dif = np.array(boots[a]) - np.array(boots["untrimmed"])
        lo, hi = np.percentile(dif, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {a:>24} - untrimmed: {dif.mean():+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}] {star}")
    print(f"  in-house PD/ET test patients: {len(te)} "
          f"({int(yt.sum())} ET)\n", flush=True)

    # ---------------- (B) mixed-cohort headline ------------------------------
    res = {a: [] for a in ARMS}
    for sp in range(SPLITS):
        tv, te2 = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                              random_state=sp).split(y[:, None],
                                                                     key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]
        for a in ARMS:
            V, T = fit_members(SPEC[a], D, traj, y, tr, va, te2)
            res[a].append(score(T.mean(0), tune_offsets(V.mean(0), y[va]),
                                y[te2]))
        print(f"  split {sp+1}/{SPLITS}", flush=True)
    for a in res:
        res[a] = np.array(res[a])

    print(f"\n(B) MIXED-COHORT HEADLINE -- predicted small, sign uncertain")
    print(f"{'arm':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")
    print("\npaired vs untrimmed:")
    for a in list(ARMS)[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], res["untrimmed"]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
