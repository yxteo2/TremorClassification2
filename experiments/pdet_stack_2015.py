"""Do the PD-vs-ET ranking gains stack? NewData pooling + REST fusion + coupling.

Three confirmed or measured PD-vs-ET ranking gains on 2015 OUT, each over `ft`
(0.625 within-fold AUC on the selection partitions):

* REST late fusion (`fusion_2015.md`)             -> 0.660
* coupling second stage on the fusion (`pdet_2015.md`) -> 0.712 (0.710 fresh)
* NewData pooled into fine-tuning (`newdata_2015.md`)  -> 0.669

Here they are combined from saved out-of-fold test probabilities -- ranking needs
no thresholds, so no refit: `fuse_nd` = geometric mean (0.5 / 0.5) of the
NewData-pooled OUT model and the REST model; `+ coupling` = the `pdet_2015`
second stage (LR on the PD-vs-ET logit + 8 coupling features, independent 5-fold
CV per repeat, seed 8000 + repeat). Within-fold AUC, as in `pdet_2015`.

The combination was assembled after seeing its parts, so the selection-partition
number is a candidate; the fresh-partition run (seeds 100-139) is the test.

Run: ``python -m experiments.pdet_stack_2015`` (selection) and
``FUSION_DIR=fusion_2015_confirm ND_DIR=newdata_2015_confirm python -m
experiments.pdet_stack_2015`` (fresh).
"""

from __future__ import annotations

import glob
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.pdet_2015 import logit_pdet


def geo(a, b, w):
    z = (1 - w) * np.log(a + 1e-12) + w * np.log(b + 1e-12)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    from common.quaternion_data import load_quaternion_recordings
    from experiments.multisegment import seg_table
    from frequency.tables import spectrum_table
    fdir = os.environ.get("FUSION_DIR", "fusion_2015_runs")
    ndir = os.environ.get("ND_DIR", "newdata_2015_runs")
    recs = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    ids = spectrum_table(recs, ch=slice(3, 6))[2]
    S, _, sp = seg_table(recs)
    i = {p: k for k, p in enumerate(sp)}
    seg = S[[i[p] for p in ids]]
    files = sorted(glob.glob(f"{ndir}/rep*.npz"))
    res = {}
    for f in files:
        rep = int(f.split("rep")[-1][:-4])
        N = np.load(f)
        F = np.load(f"{fdir}/rep{rep:02d}.npz")
        assert np.allclose(F["p_ft"], N["p_ft"]), f"rep {rep}: partitions differ"
        y = F["y"]
        m = y != 0
        yy = (y[m] == 2).astype(int)
        base = {"ft": N["p_ft"], "nd_pool": N["p_nd_pool"], "fuse": F["p_fuse"],
                # patients without REST keep the OUT model alone (as fusion_2015
                # does); an unmasked mean mixed in a placeholder (audit item 5)
                "fuse_nd": np.where(F["has"][:, None],
                                    geo(N["p_nd_pool"], F["p_rest"], 0.5),
                                    N["p_nd_pool"])}
        cv = StratifiedKFold(5, shuffle=True, random_state=8000 + rep)
        sc = {k: logit_pdet(v)[m] for k, v in base.items()}
        for k in ("fuse", "fuse_nd"):
            mdl = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=5000, class_weight="balanced"))
            sc[k + " + coupling"] = cross_val_predict(
                mdl, np.c_[sc[k], seg[m]], yy, cv=cv, method="predict_proba")[:, 1]
        for k, s in sc.items():
            res.setdefault(k, []).append(np.mean(
                [roc_auc_score(yy[te], s[te]) for _, te in cv.split(np.zeros(len(yy)), yy)]))
    R = {k: np.array(v) for k, v in res.items()}
    print(f"PD vs ET, within-fold AUC, {len(files)} repeats ({fdir}, {ndir})")
    for k in R:
        print(f"  {k:<22} {R[k].mean():.3f}")
    for a, b in (("nd_pool", "ft"), ("fuse_nd", "fuse"), ("fuse_nd", "nd_pool"),
                 ("fuse_nd + coupling", "fuse + coupling"),
                 ("fuse_nd + coupling", "ft")):
        d = R[a] - R[b]
        bs = [np.random.default_rng(s).choice(d, len(d)).mean() for s in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {a + ' - ' + b:<42} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
              f"{' *' if lo > 0 or hi < 0 else ''}  win {np.mean(d > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
