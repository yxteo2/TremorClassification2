"""MiniRocket on the waveform: the unlearned time-domain estimator this repo predicts should work.

## Why this specific method, from this repo's own standing conclusion

`time_domain_deep.md` closed learned time-domain models: a TCN on the band-passed
waveform is macroP **−0.034 [−0.066, −0.004]** * against the reported model, and
a TCN on the analytic channels is −0.076 *. But it did not close the *time
domain* — it closed **learning temporal filters from 404 patients**, and it said
so:

> Time-domain information is only reachable here through estimators that do not
> have to be learned from this cohort.

The evidence for that was catch22, whose 22 formulas were fixed offline on 93
unrelated datasets and which **ties** the spectral descriptors on PADS PD-vs-ET
(AUC 0.798 vs 0.794) at half the fold variance.

**MiniRocket is exactly the missing member of that family.** Its ~10 000
convolutional kernels are drawn from a fixed dictionary, not learned from the
data; only a linear head is fitted. The independent literature agrees on the
regime: Donié et al. (*Scientific Reports* 2025) benchmark ROCKET against
InceptionTime on wrist-accelerometer PD symptom estimation and conclude ROCKET
"is suited to small datasets" where high-capacity learners struggle.

So this is a **measurement-derived** prediction rather than a mechanism story,
and `failed_predictions.md` records that the former has by far the better record
here (eight held, fifteen mechanism stories failed).

## The prediction, recorded before the run

**MiniRocket on the waveform should beat the learned TCN on the same waveform**
(macroP 0.626, precET 0.579 in `time_domain_deep.md`), landing nearer catch22
than nearer the TCN. That is the claim this experiment is for.

It is **not** a prediction that it beats the reported model. The spectrum has
been a near-sufficient statistic at this n, and `score_vs_feature_fusion.md` is
explicit that combination pays only when no member dominates. If MiniRocket is
much weaker, the fusion arm should be read as expected-to-fail, not as a
disappointment.

## Design

Input is **the identical tensor the failed TCN received** — `patient_tensor`:
band-pass 3–15 Hz, principal-axis projection, decimate to 40 Hz, z-score,
centre-crop to 384 samples, two recordings per patient as channels (slot 1
zero-filled for the 10 % of patients with one recording). Same input, learned
versus unlearned: that is the controlled comparison.

**MiniRocket is fitted on the training fold only** and applied to validation and
test. Its bias terms are sampled from training-data quantiles, so fitting it on
all patients would leak; the transform is re-fitted inside every split.

Arms:

  reported model            the deep two-stream baseline
  MiniRocket + ridge        the canonical ROCKET pairing (RidgeClassifierCV)
  MiniRocket + logreg       same features, probabilistic head for fusion
  score fusion              geometric mix of reported and MiniRocket-logreg,
                            weight chosen on the untouched validation split

20 splits, paired. Run: ``python -m experiments.rocket_waveform``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import experiments.final_model as FM
from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from experiments.estimator_smoothing import load_cohorts
from experiments.pooling_rules import fit_members
from signal_processing.waveform import patient_tensor

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred")
SPLITS = 20
WGRID = np.linspace(0.0, 1.0, 11)
# time_domain_deep.md, the learned model on this same input
TCN_MACROP, TCN_PRECET = 0.626, 0.579


def _norm(P):
    return P / np.clip(P.sum(1, keepdims=True), 1e-12, None)


def score(pt, off, yte):
    """Per-class precision plus ET recall and the ET prediction count.

    Precision alone is degenerate at 12 % prevalence: a model that predicts ET
    once and happens to be right scores precET 1.000. The 1-split smoke test did
    exactly that (precET 1.000 with macroF1 0.440), so ET recall and the number
    of ET predictions are reported alongside and any precET figure must be read
    with them.
    """
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2], float((pred == 2).sum())]


def _val_f1(pv, off, yva):
    _, _, F, _ = precision_recall_fscore_support(
        yva, (np.log(pv + 1e-12) + off).argmax(1), labels=[0, 1, 2],
        zero_division=0)
    return F.mean()


def build_waveform(recs, keep, y):
    rA, rB, rC = recs
    XA, _, yA, _ = patient_tensor(rA, ch=slice(3, 6))
    XB, _, yB, _ = patient_tensor(rB, ch=slice(3, 6))
    XC, _, yC, _ = patient_tensor(rC, ch=slice(0, 3))
    X = np.concatenate([XA, XB, XC[keep]]).astype(np.float32)
    yy = np.concatenate([yA, yB, yC[keep]])
    assert np.array_equal(yy, y), "waveform tensor row order != build()"
    return X


def main():
    torch.set_num_threads(1)
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    d = FM.build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    spec, traj = d["SPEC"]["multitaper"], d["TRAJ"]
    recs, keep = load_cohorts()
    W = build_waveform(recs, keep, y)
    print(f"waveform tensor {W.shape} aligned with build(); "
          f"MiniRocket fitted per split on the training fold only\n", flush=True)

    ARMS = ("reported model", "MiniRocket + ridge", "MiniRocket + logreg",
            "score fusion (w on val)")
    res, done = resume_load("rocket_waveform", list(ARMS))
    ws = []

    for sp in range(SPLITS):
        if sp in done:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y[:, None],
                                                                    key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(
                                                y[tv][:, None], key[tv]))
        tr, va = tv[t0], tv[v0]

        V, T = fit_members(spec, D, traj, y, tr, va, te)
        pv_d, pt_d = _norm(V.mean(0)), _norm(T.mean(0))
        off_d = tune_offsets(pv_d, y[va])
        res["reported model"].append(score(pt_d, off_d, y[te]))

        mr = MiniRocketMultivariate(random_state=sp)
        Ftr = np.asarray(mr.fit_transform(W[tr]), dtype=np.float64)
        Fva = np.asarray(mr.transform(W[va]), dtype=np.float64)
        Fte = np.asarray(mr.transform(W[te]), dtype=np.float64)
        Ftr, Fva, Fte = (np.nan_to_num(z) for z in (Ftr, Fva, Fte))

        rc = make_pipeline(StandardScaler(),
                           RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                             class_weight="balanced"))
        rc.fit(Ftr, y[tr])
        dv = rc.decision_function(Fva)
        dt = rc.decision_function(Fte)
        sm = lambda z: _norm(np.exp(z - z.max(1, keepdims=True)))
        res["MiniRocket + ridge"].append(
            score(sm(dt), tune_offsets(sm(dv), y[va]), y[te]))

        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=5000,
                                              class_weight="balanced"))
        lr.fit(Ftr, y[tr])
        pv_r, pt_r = lr.predict_proba(Fva), lr.predict_proba(Fte)
        res["MiniRocket + logreg"].append(
            score(pt_r, tune_offsets(pv_r, y[va]), y[te]))

        best = (-1.0, 0.0)
        for w in WGRID:
            bv = _norm(np.exp((1 - w) * np.log(pv_d + 1e-12)
                              + w * np.log(pv_r + 1e-12)))
            f = _val_f1(bv, tune_offsets(bv, y[va]), y[va])
            if f > best[0]:
                best = (f, w)
        w = best[1]
        mix = lambda a, b: _norm(np.exp((1 - w) * np.log(a + 1e-12)
                                       + w * np.log(b + 1e-12)))
        res["score fusion (w on val)"].append(
            score(mix(pt_d, pt_r), tune_offsets(mix(pv_d, pv_r), y[va]), y[te]))
        ws.append(w)
        resume_save("rocket_waveform", res, sp, extra={"w": ws})
        print(f"  split {sp+1}/{SPLITS}  rocket features {Ftr.shape[1]}  "
              f"fusion w={w:.1f}", flush=True)

    for a in res:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>26}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>26}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")
    print(f"{'TCN on this waveform':>26}{'':>9}{'':>9}{TCN_PRECET:>9.3f}"
          f"{TCN_MACROP:>9.3f}   (time_domain_deep.md, for reference)")
    n_et = int(round(float((y == 2).mean()) * len(y) * TEST_FRAC))
    print(f"\n  ~{n_et} ET patients per test fold. Read precET beside recET and "
          f"nETpred: a high\n  precET with near-zero recall is the degenerate "
          f"corner, not a result.")

    base = res["reported model"]
    print("\npaired vs the reported model:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nTHE PREDICTION -- MiniRocket vs the LEARNED TCN on the same input:")
    for a in ("MiniRocket + ridge", "MiniRocket + logreg"):
        print(f"  {a:>22}  macroP {res[a][:,3].mean():.3f} vs TCN {TCN_MACROP:.3f}"
              f"   precET {res[a][:,2].mean():.3f} vs TCN {TCN_PRECET:.3f}")
    print("  (TCN numbers are from a different run, so this is a scale "
          "comparison, not a paired test)")
    print(f"\nfusion weight on val: mean {np.mean(ws):.2f}, "
          f"w=0 in {int(np.sum(np.array(ws)==0))}/{SPLITS} splits")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
