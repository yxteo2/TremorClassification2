"""Multi-segment tremor coupling: hand / lower arm / upper arm, in-house only.

## Why

IEEE TNSRE 2023 ("A Wearable Multi-Segment Upper Limb Tremor Assessment System
for Differential Diagnosis of PD Versus ET") separated PD from ET using **phase
relationships between limb segments**, and lost 9-10 % when sensors were removed.
Both in-house cohorts record three segments, but the pipeline reads only the
lower arm. An early, pre-fix 2015 study here found 3-sensor coherence/phase
features raised ET-F1 0.324 -> 0.421 (`signal_processing_summary.md`), and
cross-sensor phase at REST separated PD from ET (p = 0.004). Neither was tested
in the current pipeline or against a null.

## Features (8 per recording, fixed before looking)

Each sensor's 3-axis angular velocity is band-passed 3-15 Hz and projected on its
own principal axis. Sensors cannot be compared axis-by-axis: the canonical
(rotation-invariant) 3x3 coherence reads 0.98-1.00 for everyone (bias at ~7
Welch segments), and world-frame coherence is weakly related to it, so the
sensors' headings are not reliably shared. The principal axis has an arbitrary
sign, so phase is read only as **|cos(phase)|**: 1 = segments move in or against
phase (rigid), 0 = a quarter-cycle lag (propagating wave).

    coh_hl, coh_lu, coh_hu     coherence of the projections at the tremor peak
    cos_hl, cos_lu, cos_hu     |cos(phase)| at the tremor peak
    grad_hand, grad_upper      log band power, hand/lower and upper/lower

The tremor peak is the lower-arm peak (the pipeline's sensor), and coherence
and phase are averaged over bins within 0.5 Hz of it. Patient value = mean over
recordings.

## Tests

1. Feature level, PD vs ET and N vs Tremor, OUT (the model's task) and REST, per
   cohort and pooled (z-scored within cohort): model-free AUCs and logistic
   regression, 5-fold x 20 repeats, 200-permutation null.
2. Model level (``--model``): the block, with a have-flag and zeros for PADS, on
   the adopted 9-member ensemble, 20 splits, against a shuffled control re-drawn
   every split. Same harness and in-house metrics as `inhouse_rest.py`.

## Prediction, recorded before the run

**N vs Tremor clearly above null at OUT** (tremor propagates coherently through
the arm; noise does not). **PD vs ET at OUT inside its null**, pooled and per
cohort -- every OUT feature tried so far carries no in-house PD/ET difference,
and coupling is downstream of the same tremor. **REST lower-upper phase
replicates on 2015 at p < 0.05** (the earlier biomarker). Model level: **no
gain over the shuffled control** on in-house AUC or any merged metric.

Run: ``python -m experiments.multisegment`` (features, minutes) and
``python -m experiments.multisegment --model`` (model arm, ~1.5 h).
"""

from __future__ import annotations

import re
import sys

import numpy as np
from scipy.signal import butter, coherence, csd, sosfiltfilt, welch
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ("coh_hl", "coh_lu", "coh_hu", "cos_hl", "cos_lu", "cos_hu",
         "grad_hand", "grad_upper")
PAIRS = ((0, 1), (1, 2), (0, 2))           # hand-lower, lower-upper, hand-upper
FS, NPS = 100.0, 256
REPEATS, NPERM = 20, 200
_TASK = re.compile(r"_(OUT|REST)$")


def principal(x3):
    x3 = x3 - x3.mean(1, keepdims=True)
    u = np.linalg.svd(x3, full_matrices=False)[0][:, 0]
    return u @ x3


def seg_features(x):
    """8 coupling features of one 9-channel recording (hand, lower, upper)."""
    x = np.asarray(x, float)
    if x.shape[0] < 9 or x.shape[1] < NPS:
        return None
    sos = butter(4, [3.0, 15.0], btype="band", fs=FS, output="sos")
    s = [principal(sosfiltfilt(sos, x[3 * k:3 * k + 3], axis=-1)) for k in range(3)]
    f, Pl = welch(x[3:6], fs=FS, nperseg=NPS, axis=-1)
    b = (f >= 3) & (f <= 15)
    f0 = f[b][np.argmax(Pl.sum(0)[b])]
    near = np.abs(f - f0) <= 0.5
    out = {}
    for (i, j), tag in zip(PAIRS, ("hl", "lu", "hu")):
        _, C = coherence(s[i], s[j], fs=FS, nperseg=NPS)
        _, S = csd(s[i], s[j], fs=FS, nperseg=NPS)
        out[f"coh_{tag}"] = float(C[near].mean())
        out[f"cos_{tag}"] = float(abs(np.cos(np.angle(S[near].sum()))))
    bp = [welch(x[3 * k:3 * k + 3], fs=FS, nperseg=NPS, axis=-1)[1].sum(0)[b].sum()
          for k in range(3)]
    out["grad_hand"] = float(np.log((bp[0] + 1e-20) / (bp[1] + 1e-20)))
    out["grad_upper"] = float(np.log((bp[2] + 1e-20) / (bp[1] + 1e-20)))
    return out


def seg_table(recs):
    rows, lab = {}, {}
    for r in recs:
        d = seg_features(r.x)
        if d is None:
            continue
        rows.setdefault(r.subject, []).append([d[k] for k in FEATS])
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.array([np.mean(rows[p], 0) for p in pats]),
            np.array([lab[p] for p in pats]), np.array(pats))


def model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def cv_auc(X, y, seed0=0):
    return float(np.mean([roc_auc_score(y, cross_val_predict(
        model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed0 + r),
        method="predict_proba")[:, 1]) for r in range(REPEATS)]))


def with_null(tag, X, y):
    if min(y.sum(), len(y) - y.sum()) < 10:
        print(f"  {tag:<40} n={len(y):>3} pos={int(y.sum()):>2}  "
              f"fewer than 10 in a class -- univariate only")
        return
    real = cv_auc(X, y)
    rng = np.random.default_rng(0)
    null = np.array([cv_auc(X, rng.permutation(y), 1000 + i * REPEATS)
                     for i in range(NPERM)])
    lo, hi = np.quantile(null, [0.025, 0.975])
    p = (1 + np.sum(null >= real)) / (1 + NPERM)
    print(f"  {tag:<40} n={len(y):>3} pos={int(y.sum()):>2}  CV AUC {real:.3f}   "
          f"null 95% [{lo:.3f}, {hi:.3f}]   p={p:.3f}")


def univariate(tag, X, y):
    cells = []
    for j, k in enumerate(FEATS):
        auc = roc_auc_score(y, X[:, j])
        p = mannwhitneyu(X[y == 1, j], X[y == 0, j]).pvalue
        cells.append(f"{k}={auc:.2f}{'*' if p < 0.05 else ''}")
    print(f"  {tag:<28} " + "  ".join(cells))


def load(task):
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings
    return {"2015": load_quaternion_recordings("Data", action=task,
                                               mode="angular_velocity"),
            "NewData": load_2025_all(conditions=(task,))}


def features_main():
    print("Multi-segment coupling features, univariate AUC (* p < 0.05, "
          "> 0.5 = positive class higher)\n")
    import os
    for task in os.environ.get("MS_TASKS", "OUT,REST").split(","):
        tabs = {c: seg_table(r) for c, r in load(task).items()}
        for axis in ("PD_vs_ET", "N_vs_Tremor"):
            print(f"{task}  {axis}")
            pooled_X, pooled_y = [], []
            for c, (X, y, _) in tabs.items():
                if axis == "PD_vs_ET":
                    m = y != 0; Xa, ya = X[m], (y[m] == 2).astype(int)
                else:
                    Xa, ya = X, (y != 0).astype(int)
                univariate(f"{c} (pos={int(ya.sum())})", Xa, ya)
                pooled_X.append((Xa - Xa.mean(0)) / (Xa.std(0) + 1e-12))
                pooled_y.append(ya)
                with_null(f"{c} {task} {axis}", Xa, ya)
            with_null(f"pooled {task} {axis}", np.vstack(pooled_X),
                      np.concatenate(pooled_y))
            print()
    print("MARKER_DONE", flush=True)


def model_main():
    import torch
    from sklearn.metrics import precision_recall_fscore_support  # noqa: F401
    from sklearn.model_selection import StratifiedShuffleSplit

    import experiments.final_model as FM
    from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
    from experiments._resume import resume_load, resume_save
    from experiments.alltasks_final import paired
    from experiments.inhouse_rest import score
    from frequency.tables import spectrum_table
    from models.architectures import (TRUNKS, ResidualTCN, Spectrum1DCNN,
                                      SpectrumTransformer, TwoStreamNet)
    from common.loaders import load_pads_extracted

    torch.set_num_threads(1)
    NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred",
          "inAUC", "inPrecET", "inRecET")
    ARMS = ("reported9", "+ segments", "CONTROL shuffled")
    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]
    D0 = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]
    rec = load("OUT")
    A = spectrum_table(rec["2015"], ch=slice(3, 6))
    B = spectrum_table(rec["NewData"], ch=slice(3, 6))
    C = spectrum_table(load_pads_extracted("pads_stretchhold"), ch=slice(0, 3))
    nA, nB = len(A[1]), len(B[1])
    nC = len(y) - nA - nB
    assert np.array_equal(np.concatenate([A[1], B[1]]), y[:nA + nB]), \
        "in-house patient order does not match build()"
    coh = np.array(["2015"] * nA + ["NewData"] * nB + ["PADS"] * nC)
    inh = coh != "PADS"
    blk = np.zeros((len(y), len(FEATS) + 1))
    for c, order in (("2015", A[2]), ("NewData", B[2])):
        X, _, pats = seg_table(rec[c])
        idx = {p: i for i, p in enumerate(pats)}
        off = 0 if c == "2015" else nA
        for j, p in enumerate(order):
            if p in idx:
                blk[off + j, :-1], blk[off + j, -1] = X[idx[p]], 1.0
    print(f"segment block: have {int(blk[:, -1].sum())} of {int(inh.sum())} "
          f"in-house patients")

    res, done = {}, {}
    for a in ARMS:
        r, dn = resume_load("multiseg_" + a.replace(" ", "_"), (a,))
        res[a], done[a] = r[a], dn
    for sp in range(20):
        if all(sp in done[a] for a in ARMS):
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv], key[tv]))
        tr, va = tv[t0], tv[v0]
        sh = blk.copy()
        g = np.random.default_rng(2000 + sp)
        for c in ("2015", "NewData"):
            i = np.flatnonzero((coh == c) & (blk[:, -1] == 1))
            sh[i] = blk[g.permutation(i)]

        def fam(X, mk):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            o = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in (0, 1, 2)]
            return (np.mean([a[0] for a in o], 0), np.mean([a[1] for a in o], 0))

        tcn = fam(SPEC, lambda: ResidualTCN(NBIN, num_classes=3, ch=16))
        for arm, D in (("reported9", D0), ("+ segments", np.hstack([D0, blk])),
                       ("CONTROL shuffled", np.hstack([D0, sh]))):
            if sp in done[arm]:
                continue
            nd = D.shape[1]
            packed = np.hstack([SPEC, D, TR])
            cnn = fam(packed, lambda: TwoStreamNet(
                Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"], 8 * 2 * 4, NBIN, nd, 64))
            tf = fam(packed, lambda: TwoStreamNet(
                SpectrumTransformer(NBIN, 3, d=32), TRUNKS["trunk"], 32, NBIN, nd, 64))
            pv = np.mean([cnn[0], tcn[0], tf[0]], 0)
            pt = np.mean([cnn[1], tcn[1], tf[1]], 0)
            res[arm].append(score(pt, tune_offsets(pv, y[va]), y[te], inh[te]))
            resume_save("multiseg_" + arm.replace(" ", "_"), {arm: res[arm]}, sp)
        print(f"  split {sp + 1}/20", flush=True)

    R = {a: np.array(res[a], float) for a in ARMS}
    rec9 = np.array([0.656, 0.655, 0.715, 0.675, 0.611, 0.460])
    assert np.abs(R["reported9"][:, :6].mean(0).round(3) - rec9).max() < 1e-9, \
        "baseline does not reproduce diverse_ensemble_2.md"
    print("baseline 20-split means reproduce diverse_ensemble_2.md exactly")
    print(f"\n{'arm':>18}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>18}" + "".join(f"{v:>9.3f}" for v in np.nanmean(R[a], 0)))
    for base in ("reported9", "CONTROL shuffled"):
        print(f"\npaired '+ segments' vs {base!r}:")
        a, b = R["+ segments"], R[base]
        for i, c in enumerate(NM):
            ok = ~np.isnan(a[:, i]) & ~np.isnan(b[:, i])
            dd, lo, hi = paired(a[ok][:, [i]], b[ok][:, [i]])[0]
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"   win {float((a[ok, i] > b[ok, i]).mean()):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    model_main() if "--model" in sys.argv else features_main()
