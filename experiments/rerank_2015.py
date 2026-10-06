"""Turn the PD-vs-ET ranking gain into ET precision -- at a fixed operating point.

`pdet_2015.md`: a second stage adding the 8 multi-segment coupling features to
the fusion model's PD-vs-ET logit raises within-fold AUC 0.636 -> 0.710 on fresh
partitions. A 3-class second stage (`stack_2015.md`) lost precision by
re-learning thresholds on 151 patients. So here **nothing is re-thresholded**:

* **Stage A** -- the recommended model (`ft` + 0.25 x REST, `fuse_w25`) makes its
  own 3-class predictions. Who is N, who is tremor, and how many ET it predicts
  in each fold stay exactly as they are.
* **Stage B** -- among the patients Stage A calls tremor in a held-out fold, the
  same number of ET labels go to the patients a PD-vs-ET second stage ranks most
  ET-like. The second stage (LR, class-weighted) is fitted on the true PD and ET
  patients of the other four folds (the stage-1 partition, so no patient's own
  label reaches its score), on [fuse_w25 PD-vs-ET logit, coupling8].

Arms:

    base          fuse_w25 as saved
    rerank        Stage B with the 8 coupling features
    rerank shuf   the coupling rows permuted, re-drawn per repeat  (control)
    rerank self   Stage B on the fuse_w25 logit alone: a monotone refit, must
                  reproduce `base` exactly (sanity check)

Both partition sets: selection (seeds 0-39, `fusion_2015_runs`) and fresh
(100-139, `fusion_2015_confirm`).

## Prediction, recorded before the run

**rerank raises precET over base and over rerank shuf by +0.03 to +0.08 on both
partition sets**, with precN unchanged by construction and precPD up slightly
(an ET label moved off a PD patient fixes a PD error too). Modest, because the
second stage's weakness is its single top pick and Stage A predicts ~3 ET per
fold.

Run: ``python -m experiments.rerank_2015``
"""

from __future__ import annotations

import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "recET")
ARMS = ("base", "rerank", "rerank shuf", "rerank self")


def coupling():
    from common.quaternion_data import load_quaternion_recordings
    from experiments.multisegment import seg_table
    from frequency.tables import spectrum_table
    from experiments.legacy_ids import n_rows, recordings_for
    recs = recordings_for(n_rows("fusion_2015_runs"))
    ids = spectrum_table(recs, ch=slice(3, 6))[2]
    S, _, sp = seg_table(recs)
    i = {p: k for k, p in enumerate(sp)}
    return S[[i[p] for p in ids]]


def score(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2]]


def rerank(base, logit, X, y, rep):
    """Re-assign Stage A's ET labels within each held-out fold by a stage-2 score."""
    out = base.copy()
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
        trt = tr[y[tr] != 0]
        mdl = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=5000, class_weight="balanced"))
        mdl.fit(np.c_[logit[trt], X[trt]], (y[trt] == 2).astype(int))
        tre = te[base[te] != 0]                      # Stage A's tremor calls
        k = int((base[te] == 2).sum())               # Stage A's ET count, kept
        if len(tre) == 0:
            continue
        s = mdl.predict_proba(np.c_[logit[tre], X[tre]])[:, 1]
        o = tre[np.argsort(-s)]
        out[o] = 1
        out[o[:k]] = 2
    return out


def run(directory, seg, tag):
    files = sorted(glob.glob(f"{directory}/rep*.npz"))
    reps = [int(f.split("rep")[-1].split(".")[0]) for f in files]
    res = {a: [] for a in ARMS}
    for rep, f in zip(reps, files):
        d = np.load(f)
        y, base, p = d["y"], d["fuse_w25"], d["p_fuse_w25"]
        logit = np.log(p[:, 2] + 1e-9) - np.log(p[:, 1] + 1e-9)
        sh = seg[np.random.default_rng(12000 + rep).permutation(len(y))]
        preds = {"base": base, "rerank": rerank(base, logit, seg, y, rep),
                 "rerank shuf": rerank(base, logit, sh, y, rep),
                 "rerank self": rerank(base, logit, np.zeros((len(y), 0)), y, rep)}
        for a in ARMS:
            res[a].append(score(y, preds[a]))
    R = {a: np.array(v) for a, v in res.items()}
    n = len(files)
    print(f"\n=== {tag}: {directory}, {n} repeats ===")
    print(f"{'arm':>12}" + "".join(f"{c:>9}" for c in NAMES))
    for a in ARMS:
        print(f"{a:>12}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))
    for a, b in (("rerank", "base"), ("rerank", "rerank shuf"),
                 ("rerank self", "base")):
        dd = R[a] - R[b]
        cells = []
        for i in range(len(NAMES)):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{NAMES[i]} {dd[:, i].mean():+.3f}"
                         f"{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"  {a + ' - ' + b:<24} " + "  ".join(cells))
    return R


def main():
    seg = coupling()
    run("fusion_2015_runs", seg, "selection partitions")
    run("fusion_2015_confirm", seg, "fresh partitions")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
