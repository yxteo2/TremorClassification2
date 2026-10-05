"""Patient-level uncertainty for this session's 2015 claims.

Every paired CI in the 2015 reports resamples the 40 per-repeat differences.
All repeats score the same 151 rows, so those CIs measure re-partitioning and
seed noise only -- not which patients happened to be sampled, which with 15 ET
is the dominant uncertainty. "Confirmed on fresh partitions" is the same
patients again. (Audit, 2026-10-05.)

Here the resampling unit is the **patient**: each bootstrap draws patients with
replacement, stratified by class; every repeat's out-of-fold predictions are
scored on that draw (patient multiplicity as a weight); the per-repeat metric
is averaged over repeats; the paired difference between arms is recorded.
2000 draws. A claim survives if the 95 % interval of the difference excludes 0.

Saved runs predate the subject-id fix in `common.quaternion_data` and carry
PD 12 twice ("PD 12_OUT" and "PD 12_OUT N LOAD", one person). The two rows are
resampled together as one patient. Their leakage across folds inside the saved
runs cannot be undone post hoc; it touches 1 of 151 rows.

Run: ``python -m experiments.patient_bootstrap_2015``
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

B = 2000


def old_ids():
    """Row order of the saved runs: subject ids as the loader minted them pre-fix."""
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob("Data/raw_quaternion/OUT/*/*.txt")]
    return sorted({re.sub(r"[_\s]+\d+$", "", s) for s in stems})


def clusters():
    from common.quaternion_data import canonical_subject
    ids = old_ids()
    canon = [canonical_subject(i) for i in ids]
    u = {c: k for k, c in enumerate(dict.fromkeys(canon))}
    return np.array([u[c] for c in canon]), ids


def metrics(y, pred, prob, w):
    """precN, precPD, precET, macroP, PD-vs-ET AUC with patient weights w."""
    out = []
    for c in range(3):
        k = pred == c
        out.append(np.sum(w[k] * (y[k] == c)) / max(np.sum(w[k]), 1e-12))
    out.append(np.mean(out[:3]))
    m = (y != 0) & (w > 0)
    s = prob[m, 2] / (prob[m, 1] + prob[m, 2] + 1e-12)
    out.append(roc_auc_score((y[m] == 2).astype(int), s, sample_weight=w[m]))
    return np.array(out)


NAMES = ("precN", "precPD", "precET", "macroP", "AUC")


def compare(tag, reps, a, b, cl, y):
    """reps: list of dicts {arm: (pred, prob)}; a, b: arm names."""
    rng = np.random.default_rng(0)
    groups = {c: np.unique(cl[y == c]) for c in range(3)}
    point = np.mean([metrics(y, *r[a], np.ones(len(y))) - metrics(y, *r[b], np.ones(len(y)))
                     for r in reps], 0)
    draws = []
    for _ in range(B):
        pick = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups.values()])
        cnt = np.bincount(pick, minlength=cl.max() + 1)
        w = cnt[cl].astype(float)
        # a cluster's rows share its multiplicity; a 2-row cluster (PD 12) is
        # down-weighted per row so the person counts once
        size = np.bincount(cl)[cl]
        w = w / size
        draws.append(np.mean([metrics(y, *r[a], w) - metrics(y, *r[b], w)
                              for r in reps], 0))
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5], 0)
    print(f"\n{tag}: {a} − {b}  ({len(reps)} repeats, {B} patient draws)")
    for i, nm in enumerate(NAMES):
        star = "*" if lo[i] > 0 or hi[i] < 0 else " "
        print(f"   {nm:>7} {point[i]:+.3f}  patient-level 95% [{lo[i]:+.3f}, {hi[i]:+.3f}] {star}")


def bootstrap_dir(directory, a, b, tag=None):
    """Patient-level paired comparison of two arms saved in `directory`.

    Works for runs made before the PD 12 id fix (151 rows, PD 12 resampled as one
    patient) and after it (150 rows, one row per patient). Use from new
    experiments as ``bootstrap_dir("my_runs", "new_arm", "ft")``.
    """
    reps, y = load(directory, (a, b))
    if len(y) == 151:
        cl, _ = clusters()
    else:
        cl = np.arange(len(y))
    compare(tag or directory, reps, a, b, cl, y)


def load(directory, arms):
    reps = []
    for f in sorted(glob.glob(f"{directory}/rep*.npz")):
        d = np.load(f)
        reps.append({a: (d[a], d[f"p_{a}"]) for a in arms})
    return reps, np.load(f)["y"]


def main():
    cl, ids = clusters()
    print(f"saved-run rows: {len(ids)}; patients: {cl.max() + 1}")
    jobs = [("transfer (segments_2015)", "segments_2015_runs", ("ft", "scratch")),
            ("coupling in the network", "segments_2015_runs", ("ft_seg", "ft")),
            ("REST fusion, w 0.25", "fusion_2015_runs", ("fuse_w25", "ft")),
            ("REST fusion, w 0.25, fresh", "fusion_2015_confirm", ("fuse_w25", "ft")),
            ("REST fusion, w 0.5", "fusion_2015_runs", ("fuse", "ft")),
            ("WING", "fusion3_2015_runs", ("+ WING", "base")),
            ("WING, fresh", "fusion3_2015_confirm", ("+ WING", "base")),
            ("NewData pooled", "newdata_2015_runs", ("nd_pool", "ft")),
            ("NewData pooled, fresh", "newdata_2015_confirm", ("nd_pool", "ft"))]
    for tag, d, (a, b) in jobs:
        if not glob.glob(f"{d}/rep*.npz"):
            print(f"\n{tag}: {d} missing, skipped")
            continue
        reps, y = load(d, (a, b))
        assert len(y) == len(cl), f"{d}: {len(y)} rows vs {len(cl)}"
        compare(tag, reps, a, b, cl, y)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
