"""Cheap pre-fit diagnostic: does REST carry in-house PD-vs-ET signal OUT lacks?

In-house (2015 + NewData) is the cohort where every model sits at chance on
PD vs ET (`inhouse_pd_vs_et.md`, `inhouse_shortwindow.md`), because at OUT
in-house ET match in-house PD on amplitude, sharpness and frequency. The same
report found 2015 REST separates them (AUC 0.65-0.70 on six characteristics,
all three sensors above their nulls), and flagged "in-house REST in the deep
model is untested". Before spending deep fits on that (invariant 10), this asks
the same question with a linear model on the tables the deep model would see:
16-bin multitaper log-spectrum + ten descriptors, per task, per sensor, and the
OUT (+) REST union -- side by side, never averaged (`task_averaging.md`).

Protocol: in-house pooled, features z-scored within cohort (label-free),
L2 logistic regression, stratified 5-fold CV averaged over 10 partitions
(`classify(n_repeats=10)` rule), null from 100 label permutations within cohort
with the same averaged statistic. The four `N 2` REST/WING accelerometer files
(verify_data check 13) are dropped.

PREDICTION (recorded before the run, measurement-derived):
  * PD vs ET, OUT, any sensor: inside its null (as measured before, ~0.3-0.55).
  * PD vs ET, REST: above the null on the lower arm, AUC 0.60-0.70 -- lower
    than 2015 alone because NewData REST (6 ET) read 0.31 in
    `kinetic_task_audit.md` and dilutes it.
  * OUT (+) REST union: no better than REST alone (OUT adds nothing on this
    axis; invariant 12 union-vs-best-member).
  * N vs tremor: OUT > REST (0.84 vs 0.65 on NewData).
If REST is inside its null on the pooled in-house cohort, the deep REST arm is
not worth running.

Run: ``python -m experiments._inhouse_rest_diagnostic``
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common.cohorts import desc_table, logbin
from experiments.final_model import method_table

SENSORS = {"hand": slice(0, 3), "lower": slice(3, 6), "upper": slice(6, 9)}
N_REP, N_PERM = 10, 100


def _bad_2015_files():
    bad = set()
    for p in sorted(glob.glob("Data/raw_quaternion/*/*/*.txt")):
        Q = pd.read_csv(p, sep=None, engine="python", header=None).to_numpy(float)
        nq = np.median(np.linalg.norm(Q.reshape(len(Q), -1, 4), axis=2))
        if not 0.95 < nq < 1.05:
            bad.add(p.replace("\\", "/").split("raw_quaternion/")[1])
    return bad


def load_task(task):
    from common.load_2025 import load_2025_all
    from common.quaternion_data import load_quaternion_recordings_multi
    bad = _bad_2015_files()
    rA = [r for r in load_quaternion_recordings_multi(
              "Data", [task], mode="angular_velocity")
          if str(r.path).replace("\\", "/").split("raw_quaternion/")[1] not in bad]
    rB = load_2025_all(conditions=(task,))
    return rA, rB


def feats(recs, ch):
    S, y, p = method_table(recs, "multitaper", ch)
    return np.hstack([logbin(S), desc_table(recs, ch)]), y, p


def table(task):
    """{sensor: (X, y, patients, cohort)} pooled in-house, z-scored per cohort."""
    rA, rB = load_task(task)
    out = {}
    for sn, ch in SENSORS.items():
        parts = []
        for coh, recs in ((0, rA), (1, rB)):
            X, y, p = feats(recs, ch)
            X = (X - X.mean(0)) / (X.std(0) + 1e-8)
            parts.append((X, y, p, np.full(len(y), coh)))
        out[sn] = tuple(np.concatenate([a[i] for a in parts]) for i in range(4))
    return out


def cv_auc(X, y, n_rep=N_REP):
    aucs = []
    for r in range(n_rep):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
            s[te] = m.fit(X[tr], y[tr]).decision_function(X[te])
        aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs))


def with_null(X, y, coh, seed=0):
    obs = cv_auc(X, y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(N_PERM):
        yp = y.copy()
        for c in np.unique(coh):
            i = np.flatnonzero(coh == c)
            yp[i] = rng.permutation(yp[i])
        null.append(cv_auc(X, yp, n_rep=3))
    null = np.array(null)
    lo, hi = np.percentile(null, [2.5, 97.5])
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    return obs, lo, hi, p


def join(a, b):
    """Patients present in both task tables, features side by side."""
    ia = {p: i for i, p in enumerate(a[2])}
    ib = {p: i for i, p in enumerate(b[2])}
    common = [p for p in a[2] if p in ib]
    i1 = np.array([ia[p] for p in common]); i2 = np.array([ib[p] for p in common])
    assert (a[1][i1] == b[1][i2]).all(), "label mismatch across tasks"
    return (np.hstack([a[0][i1], b[0][i2]]), a[1][i1], np.array(common), a[3][i1])


def main():
    T = {t: table(t) for t in ("OUT", "REST")}
    for t in T:
        _, y, _, c = T[t]["lower"]
        print(f"{t}: n={len(y)}  N/PD/ET = {[(y == k).sum() for k in (0, 1, 2)]}"
              f"  (2015 {int((c == 0).sum())}, NewData {int((c == 1).sum())})")
    arms = []
    for sn in SENSORS:
        arms += [(f"OUT  {sn}", T["OUT"][sn]), (f"REST {sn}", T["REST"][sn]),
                 (f"OUT+REST {sn}", join(T["OUT"][sn], T["REST"][sn]))]
    allr = T["REST"]["hand"]
    for sn in ("lower", "upper"):
        allr = join(allr, T["REST"][sn])
    arms.append(("REST all sensors", allr))

    for axis in ("PD vs ET", "N vs tremor"):
        print(f"\n{axis}   (AUC, 10-partition mean; null 95 %, {N_PERM} perms)")
        for name, (X, y, p, c) in arms:
            if axis == "PD vs ET":
                m = y != 0
                Xa, ya, ca = X[m], (y[m] == 2).astype(int), c[m]
            else:
                Xa, ya, ca = X, (y != 0).astype(int), c
            obs, lo, hi, pv = with_null(Xa, ya, ca)
            star = "*" if pv < 0.05 else " "
            print(f"  {name:>20}  n={len(ya):>3} pos={ya.sum():>3}  AUC {obs:.3f}"
                  f"  null [{lo:.3f}, {hi:.3f}]  p={pv:.3f} {star}", flush=True)
    print("MARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
