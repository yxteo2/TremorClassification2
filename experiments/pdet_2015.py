"""PD vs ET on 2015 OUT: can the ranking of tremor patients be improved?

Current PD-vs-ET AUC among the 90 tremor patients (40 repeats): `ft` 0.625,
L2-SP 0.642 (`transfer_2015.md`), `ft_seg` 0.639 (`multisegment_2015.md`),
OUT + REST fusion 0.662 (`fusion_2015.md`).

`stack_2015.md` showed the multi-segment coupling features carry real
information (vs a shuffled copy: precET +0.049 \\*, macroF1 +0.033 \\*) but a
3-class second stage lost it to the cost of re-learning thresholds on 151
patients. A **ranking** needs no threshold, so this asks the binary question
directly, from saved out-of-fold probabilities (no deep refit).

## Arms (PD vs ET among tremor patients; score = P(ET) / (P(PD) + P(ET)))

    ft, fuse, ft_seg      single models, as saved
    avg                   geometric mean of fuse and ft_seg probabilities
    stack                 LR on fuse's PD-vs-ET logit, refit by CV -- a
                          monotone recalibration, so it should equal fuse
    + coupling8           LR on fuse's logit + the 8 coupling features
    + coupling8 shuf      the same with the coupling rows permuted, re-drawn
                          per repeat (attribution control)

Second-stage arms use an independent 5-fold CV over the 90 tremor patients per
repeat (seed 8000 + repeat); fuse's probabilities are already out-of-fold.
Reported: AUC, and ET precision among the k tremor patients ranked most ET-like.

## Prediction, recorded before the run

**+ coupling8 beats both fuse and the shuffled control on PD-vs-ET AUC by +0.02
to +0.05**, since hand-forearm coherence separates the ET->PD errors from PD
(0.650 vs 0.911). **avg gains less than +0.015** over fuse (the two models share
the OUT backbone).

Run: ``python -m experiments.pdet_2015`` (needs `segments_2015_runs/` and
`fusion_2015_runs/`).
"""

from __future__ import annotations

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARMS = ("ft", "fuse", "ft_seg", "avg", "stack", "+ coupling8", "+ coupling8 shuf",
        "+ coupling8 + rms", "+ coupling8 + rms shuf")
TOPK = (3, 5, 10)


def logit_pdet(p):
    return np.log(p[:, 2] + 1e-9) - np.log(p[:, 1] + 1e-9)


def main():
    from common.quaternion_data import load_quaternion_recordings
    from experiments.multisegment import seg_table
    from frequency.tables import spectrum_table
    import os
    F = sorted(glob.glob(os.path.join(os.environ.get("FUSION_DIR", "fusion_2015_runs"),
                                      "rep*.npz")))
    seg_dir = os.environ.get("SEG_DIR", "segments_2015_runs")
    S = sorted(glob.glob(os.path.join(seg_dir, "rep*.npz"))) if seg_dir != "none" else []
    assert F and (not S or len(S) == len(F))
    global ARMS
    if not S:      # confirmation run: no ft_seg / avg arms
        ARMS = tuple(a for a in ARMS if a not in ("ft_seg", "avg"))
        S = F
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = spectrum_table(recs, ch=slice(3, 6))[2]
    Sg, _, sp = seg_table(recs)
    i = {p: k for k, p in enumerate(sp)}
    seg = Sg[[i[p] for p in ids]]
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [3 / 50, 15 / 50], btype="band", output="sos")
    rr = {}
    for rc in recs:
        rr.setdefault(rc.subject, []).append(np.sqrt(np.mean(
            sosfiltfilt(sos, np.asarray(rc.x[3:6], float), axis=-1) ** 2)))
    rms = np.log10([np.mean(rr[p]) for p in ids])
    res = {a: [] for a in ARMS}
    top = {a: [] for a in ARMS}
    wf = {a: [] for a in ARMS}
    fold_seed = int(os.environ.get("STAGE2_SEED", "8000"))
    for r, (s, f) in enumerate(zip(S, F)):
        a, b = np.load(s), np.load(f)
        assert np.allclose(a["p_ft"], b["p_ft"])
        a = dict(a)
        a.setdefault("p_ft_seg", a["p_ft"])
        y = a["y"]
        m = y != 0
        yy = (y[m] == 2).astype(int)
        lf = logit_pdet(b["p_fuse"])[m]
        sc = {"ft": logit_pdet(a["p_ft"])[m], "fuse": lf,
              "ft_seg": logit_pdet(a["p_ft_seg"])[m],
              "avg": logit_pdet(np.sqrt(b["p_fuse"] * a["p_ft_seg"]))[m]}
        cv = StratifiedKFold(5, shuffle=True, random_state=fold_seed + r)
        sh = seg[m][np.random.default_rng(9500 + r).permutation(m.sum())]
        # post-hoc (added after the first run, see report): coherence means
        # nothing without tremor, so give the second stage tremor amplitude and
        # its products with the coupling features; control = coupling rows
        # permuted (amplitude kept), re-drawn per repeat
        rz = (rms[m] - rms[m].mean()) / rms[m].std()
        cz = (seg[m] - seg[m].mean(0)) / (seg[m].std(0) + 1e-12)
        shz = (sh - sh.mean(0)) / (sh.std(0) + 1e-12)
        for arm, X in (("stack", lf[:, None]),
                       ("+ coupling8", np.c_[lf, seg[m]]),
                       ("+ coupling8 shuf", np.c_[lf, sh]),
                       ("+ coupling8 + rms", np.c_[lf, cz, rz, cz * rz[:, None]]),
                       ("+ coupling8 + rms shuf", np.c_[lf, shz, rz, shz * rz[:, None]])):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=5000, class_weight="balanced"))
            sc[arm] = cross_val_predict(mdl, X, yy, cv=cv,
                                        method="predict_proba")[:, 1]
        for arm in ARMS:
            res[arm].append(roc_auc_score(yy, sc[arm]))
            o = np.argsort(-sc[arm])
            top[arm].append([yy[o[:k]].mean() for k in TOPK])
            # within-fold scoring: second-stage folds have different calibration,
            # so pooled scores mix scales. Per-fold AUC (averaged) and the most /
            # two most ET-like patients of each fold are comparable for all arms.
            fa, f1, f2 = [], [], []
            for _, te in cv.split(np.zeros(len(yy)), yy):
                st, yt = sc[arm][te], yy[te]
                fa.append(roc_auc_score(yt, st))
                ot = np.argsort(-st)
                f1.append(yt[ot[0]]); f2.extend(yt[ot[:2]])
            wf[arm].append([np.mean(fa), np.mean(f1), np.mean(f2)])
    R = {k: np.array(v) for k, v in res.items()}
    T = {k: np.array(v) for k, v in top.items()}
    print("2015 OUT, PD vs ET among 90 tremor patients (15 ET), 40 repeats\n")
    print(f"{'arm':>18}{'AUC':>8}" + "".join(f"{'top' + str(k):>8}" for k in TOPK))
    for k in ARMS:
        print(f"{k:>18}{R[k].mean():>8.3f}" + "".join(f"{v:>8.3f}" for v in T[k].mean(0)))
    print("\npaired AUC differences (bootstrap 95 % CI, win rate)")
    for x, z in (("+ coupling8", "fuse"), ("+ coupling8", "+ coupling8 shuf"),
                 ("+ coupling8 shuf", "fuse"), ("avg", "fuse"), ("stack", "fuse"),
                 ("fuse", "ft"), ("+ coupling8", "ft")):
        if x not in R or z not in R:
            continue
        d = R[x] - R[z]
        bs = [np.random.default_rng(q).choice(d, len(d)).mean() for q in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {x + ' - ' + z:<34} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
              f"  win {np.mean(d > 0):.2f}")
    W = {k: np.array(v) for k, v in wf.items()}
    print("\nwithin-fold scoring (second-stage folds; same folds for every arm)")
    print(f"{'arm':>18}{'fold AUC':>10}{'top1/fold':>11}{'top2/fold':>11}")
    for k in ARMS:
        print(f"{k:>18}" + "".join(f"{v:>10.3f} " for v in W[k].mean(0)))
    for x, z in (("+ coupling8", "fuse"), ("+ coupling8", "+ coupling8 shuf"),
                 ("+ coupling8", "ft"), ("+ coupling8", "ft_seg"), ("ft_seg", "ft"),
                 ("+ coupling8 + rms", "+ coupling8"),
                 ("+ coupling8 + rms", "+ coupling8 + rms shuf"),
                 ("+ coupling8 + rms", "ft"), ("+ coupling8 + rms", "ft_seg")):
        if x not in W or z not in W:
            continue
        for j, nm in enumerate(("fold AUC", "top1", "top2")):
            d = W[x][:, j] - W[z][:, j]
            bs = [np.random.default_rng(q).choice(d, len(d)).mean() for q in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {x + ' - ' + z:<30} {nm:<9} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
