"""Give the reported final model every task instead of only the postural one.

`experiments/mil_recordings.py` found that the win is the **discarded data**, not
the architecture: on a spectrum-only model, averaging over all of a patient's
recordings instead of the postural task alone gave macroP +0.040 and
precPD +0.057 [+0.000, +0.106], while learned pooling (attention, max) was
significantly worse than the same uniform average.

That was measured on a deliberately stripped-down model — spectrum only, no
descriptors, no asymmetry, no trajectory — so the baseline sat at macroP 0.599
rather than the reported 0.660. A gain on a weak model does not automatically
survive on a strong one; the extra tasks may be supplying information the
descriptor and trajectory streams already carry.

This runs the actual reported model (`multitaper + trajectory`, soft-voted with
`ResidualTCN`, validation-tuned priors) and changes exactly one thing: which
recordings the per-patient tables are averaged over.

  postural only (reported)   OUT / OUT / StretchHold -- 768 recordings
  ALL tasks, spectrum        every task, spectrum table only
  ALL tasks, spec + desc     every task, spectrum and descriptors

The implementation deliberately reuses the repo's own `method_table` and
`desc_table` rather than reimplementing the averaging. The only change is that
each cohort's recording list is the union across tasks, with subject ids
normalised so one patient is one row -- 2015 encodes the action into the subject
id (``ET 10_OUT`` / ``_REST`` / ``_WING``), which would otherwise split each
patient into three and leak across the train/test boundary.

Trajectory and asymmetry stay on the postural task in every arm. Both are
defined relative to a posture (the IF trajectory over a sustained hold, the
left-right asymmetry of one task), so averaging them across kinetic tasks would
change their meaning rather than just their sample size.

Run: ``python -m experiments.alltasks_final``
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.cohorts import desc_table, logbin
from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.final_model import NBIN, SPLITS, TL, build, method_table
from frequency.tables import spectrum_table
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
_ACTION = re.compile(r"_(OUT|REST|WING)$")


def norm(recs):
    """Recordings with task-independent subject ids (see module docstring)."""
    return [replace(r, subject=_ACTION.sub("", str(r.subject))) for r in recs]


def all_task_recs():
    """Per cohort: (postural recordings, all recordings, channel slice)."""
    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    a_post = load_quaternion_recordings("Data", action="OUT",
                                        mode="angular_velocity")
    a_all = list(a_post)
    for act in ("REST", "WING"):
        try:
            a_all += load_quaternion_recordings("Data", action=act,
                                                mode="angular_velocity")
        except Exception:
            pass

    b_post = load_2025_all(conditions=("OUT",))
    b_all = list(b_post)
    for t in ALL_TASKS_2025:
        if t == "OUT":
            continue
        try:
            r = load_2025_all(conditions=(t,))
            if r:
                b_all += r
        except Exception:
            pass

    c_post = load_pads_extracted("pads_stretchhold")
    c_all = list(c_post)
    if os.path.isdir("pads_relaxed"):
        try:
            c_all += load_pads_extracted("pads_relaxed")
        except Exception:
            pass

    return ((norm(a_post), norm(a_all), slice(3, 6)),
            (norm(b_post), norm(b_all), slice(3, 6)),
            (norm(c_post), norm(c_all), slice(0, 3)))


def aligned(tables, order):
    """Reorder a (patients, F) table onto the reference patient order."""
    X, pats = tables
    idx = {p: i for i, p in enumerate(pats)}
    return np.array([X[idx[p]] for p in order])


def evaluate(spec, desc, traj, y, key, splits=SPLITS):
    """The reported two-stream model, unchanged."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    out = []
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    return np.array(out)


def paired(a, b, n=4000):
    d = a - b
    return [(d[:, i].mean(),
             *np.percentile([np.mean(np.random.default_rng(s).choice(
                 d[:, i], len(d), replace=True)) for s in range(n)],
                 [2.5, 97.5]))
            for i in range(len(NM))]


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D_post = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj = d["TRAJ"]

    # reference patient order, matching build(): 2015, NewData, PADS[capped]
    cohorts = all_task_recs()
    (a_post, a_all, ach), (b_post, b_all, bch), (c_post, c_all, cch) = cohorts
    A, B, C = (spectrum_table(a_post, ch=ach), spectrum_table(b_post, ch=bch),
               spectrum_table(c_post, ch=cch))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B[2], C[2][keep]])
    assert len(order) == len(y), f"{len(order)} vs {len(y)}"
    assert np.array_equal(np.concatenate([A[1], B[1], C[1][keep]]), y), \
        "patient order does not match build()"

    n_post = len(a_post) + len(b_post) + len(c_post)
    n_all = len(a_all) + len(b_all) + len(c_all)
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"recordings: postural {n_post}   all tasks {n_all} "
          f"({n_all/n_post:.1f}x)\n", flush=True)

    def spec_for(which):
        parts, pats = [], []
        for (post, alls, ch) in cohorts:
            recs = post if which == "post" else alls
            X, _, p = method_table(recs, "multitaper", ch)
            parts.append(X); pats.append(p)
        Xall = np.vstack(parts)
        pall = np.concatenate(pats)
        return logbin(aligned((Xall, pall), order))

    def desc_for(which):
        parts, pats = [], []
        for (post, alls, ch) in cohorts:
            recs = post if which == "post" else alls
            parts.append(desc_table(recs, ch))
            pats.append(spectrum_table(recs, ch=ch)[2])
        return aligned((np.vstack(parts), np.concatenate(pats)), order)

    print("building tables ...", flush=True)
    S_post, S_all = spec_for("post"), spec_for("all")
    D_all = np.hstack([desc_for("all"), d["ASYM"], d["HAVE"]])

    ARMS = (("postural only (reported)", S_post, D_post),
            ("ALL tasks, spectrum", S_all, D_post),
            ("ALL tasks, spec + desc", S_all, D_all))

    res = {}
    print(f"\n{'arm':>28}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for lab, S, D in ARMS:
        res[lab] = evaluate(S, D, traj, y, key)
        m = res[lab].mean(0)
        print(f"{lab:>28}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["postural only (reported)"]
    print("\npaired vs postural only, same splits:")
    for lab, _, _ in ARMS[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
