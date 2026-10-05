"""Second-stage decision on top of the 2015 OUT transfer model (no retraining).

## Why -- from the error analysis of `ft` (40 repeats, `segments_2015_runs`)

27 of 151 patients are wrong in >= 75 % of repeats: PD->N 11, ET->PD 6,
N->PD 5, ET->N 3, PD->ET 2. Model-free:

* **PD->N and ET->N have no tremor in any 2015 task** (log tremor RMS at OUT
  -1.96 / -2.03 vs correct controls -2.04; also at REST and WING). Unreachable
  by tremor features: the ceiling.
* **ET->PD differ from PD on hand-forearm coherence** (coh_hl median 0.650 vs
  0.911 for correct PD; correct ET 0.524), the one coupling feature that carried
  the PD-vs-ET signal in `multisegment_2015.md`. As 8 extra deep-model inputs it
  did not help `ft` significantly; here it enters the final decision directly.
* **N->PD controls have tremor at OUT but not at REST** (-2.14 vs PD -1.85).

## Design

For each of the 40 saved repeats: features = `ft`'s out-of-fold
log-probabilities (3), plus an extra block. A multinomial logistic regression
(class-weight balanced) is fitted and evaluated by a **second, independent**
5-fold CV over the 151 patients (seed 7000 + repeat), every patient predicted
out of fold. The first-stage probabilities are already out-of-fold, so no
patient's own label reaches its second-stage feature through stage one.

    stack            ft log-probs only (recalibration -- the fair baseline)
    + coh_hl         + hand-forearm coherence                    (primary)
    + coh_hl shuf    + the same column permuted, re-drawn per repeat (control)
    + coupling8      + all 8 coupling features
    + REST rms       + log REST tremor RMS (+ have flag)  -- late fusion with a
                     second task: OUTSIDE the "one action per model" scope,
                     reported separately

`ft` itself (offsets tuned on validation) is printed for reference. The second
stage uses **the same decision rule** by default (``RULE=offsets``: logit offsets
tuned for macro-F1 on a 25 % inner validation split), so it can be compared with
`ft`. The first run used ``RULE=balanced`` (argmax of a class-balanced LR), which
moved every stacked arm toward ET recall and made the comparison with `ft`
uninformative; its within-stack contrasts are kept in the report.

## Caveat stated up front

coh_hl was chosen *because* it separated the error group in the diagnostic on
these same patients. The shuffled control and the out-of-fold second stage
guard the evaluation, but the choice of feature is not blind: a gain here is a
candidate for fresh data, not a result.

## Prediction, recorded before the run

**+ coh_hl beats both `stack` and `+ coh_hl shuf` on precET by +0.03 to +0.06**,
with PD precision not falling; **+ coupling8 does no better than + coh_hl**
(the other seven carried nothing in the diagnostic). **+ REST rms raises precPD
and precN** (fewer N->PD controls) and leaves precET flat.

Run: needs `segments_2015_runs/` (`experiments.segments_2015`), then
``python -m experiments.stack_2015``.
"""

from __future__ import annotations

import glob
import re

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.protocol import tune_offsets

#: "offsets" (default): the same decision rule as ft -- class-weighted model,
#: logit offsets tuned for macro-F1 on a 25 % inner validation split.
#: "balanced": argmax of a class-balanced LR (first run; shifts toward ET recall).
RULE = os.environ.get("RULE", "offsets")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "recET")
ARMS = ("ft (reference)", "stack", "+ coh_hl", "+ coh_hl shuf", "+ coupling8",
        "+ REST rms")


def score(y, p):
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2]]


def extras(ids):
    from scipy.signal import butter, sosfiltfilt
    from common.quaternion_data import load_quaternion_recordings
    from experiments.multisegment import seg_table
    from experiments.legacy_ids import n_rows, recordings_for
    recs = recordings_for(n_rows("segments_2015_runs"))
    S, _, sp = seg_table(recs)
    i = {p: k for k, p in enumerate(sp)}
    seg = S[[i[p] for p in ids]]
    sos = butter(4, [3 / 50, 15 / 50], btype="band", output="sos")
    strip = lambda s: re.sub(r"_(OUT|REST|WING)$", "", str(s))
    rest = {}
    for r in load_quaternion_recordings("Data", action="REST", mode="angular_velocity"):
        x = np.asarray(r.x[3:6], float)
        rest.setdefault(strip(r.subject), []).append(
            np.sqrt(np.mean(sosfiltfilt(sos, x, axis=-1) ** 2)))
    v = np.array([np.log10(np.mean(rest[strip(p)])) if strip(p) in rest else np.nan
                  for p in ids])
    have = np.isfinite(v).astype(float)
    v[~np.isfinite(v)] = np.nanmedian(v)
    return seg, np.c_[v, have]


def main():
    from common.quaternion_data import load_quaternion_recordings
    from frequency.tables import spectrum_table
    files = sorted(glob.glob("segments_2015_runs/rep*.npz"))
    assert files, "run experiments.segments_2015 first"
    from experiments.legacy_ids import n_rows, recordings_for
    recs = recordings_for(n_rows("segments_2015_runs"))
    ids = spectrum_table(recs, ch=slice(3, 6))[2]
    seg, rest = extras(ids)
    coh = seg[:, [0]]
    res = {a: [] for a in ARMS}
    for k, f in enumerate(files):
        d = np.load(f)
        y = d["y"]
        L = np.log(d["p_ft"] + 1e-6)
        shuf = coh[np.random.default_rng(9000 + k).permutation(len(y))]
        blocks = {"stack": L, "+ coh_hl": np.hstack([L, coh]),
                  "+ coh_hl shuf": np.hstack([L, shuf]),
                  "+ coupling8": np.hstack([L, seg]),
                  "+ REST rms": np.hstack([L, rest])}
        res["ft (reference)"].append(score(y, d["ft"]))
        cv = StratifiedKFold(5, shuffle=True, random_state=7000 + k)
        for a, X in blocks.items():
            if RULE == "balanced":
                m = make_pipeline(StandardScaler(), LogisticRegression(
                    max_iter=5000, class_weight="balanced"))
                p = cross_val_predict(m, X, y, cv=cv)
            else:            # ft's own rule: offsets tuned on an inner validation
                p = np.zeros(len(y), int)
                for f_, (trv, te) in enumerate(cv.split(X, y)):
                    tr, va = train_test_split(trv, test_size=0.25, stratify=y[trv],
                                              random_state=k * 10 + f_)
                    m = make_pipeline(StandardScaler(), LogisticRegression(
                        max_iter=5000, class_weight="balanced")).fit(X[tr], y[tr])
                    off = tune_offsets(m.predict_proba(X[va]), y[va])
                    p[te] = (np.log(m.predict_proba(X[te]) + 1e-12) + off).argmax(1)
            res[a].append(score(y, p))
    R = {a: np.array(v) for a, v in res.items()}
    n = len(files)
    print(f"2015 OUT, second stage on ft's out-of-fold probabilities, {n} repeats, "
          f"decision rule: {RULE}\n")
    print(f"{'arm':>16}" + "".join(f"{c:>9}" for c in NAMES))
    for a in ARMS:
        print(f"{a:>16}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0)))
    for a, b in (("+ coh_hl", "stack"), ("+ coh_hl", "+ coh_hl shuf"),
                 ("+ coupling8", "+ coh_hl"), ("+ REST rms", "stack"),
                 ("stack", "ft (reference)"), ("+ coh_hl", "ft (reference)"),
                 ("+ coupling8", "ft (reference)"), ("+ coupling8", "stack")):
        dd = R[a] - R[b]
        print(f"\n{a} - {b}  (paired, {n} repeats)")
        for i, c in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
