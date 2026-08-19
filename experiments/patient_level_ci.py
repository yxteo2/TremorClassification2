"""Does the headline merged result survive patient-level uncertainty?

Every paired interval in this project is bootstrapped over **splits** on a fixed
set of 404 patients:

    b = [np.mean(rng.choice(diff[:, i], len(diff), replace=True)) ...]

`diff` there has one row per split, so resampling it estimates how much the
difference moves when the *fold assignment* changes. That is the right instrument
for "is A better than B on these patients", and this repo has used it
consistently and correctly for that question.

It is not the instrument for the question a paper asks, which is whether A beats
B on **patients we have not seen**. For that the sampling unit is the patient,
and at 49 ET the patient-level term is the one that binds — `permutation_null.md`
showed the same distinction the hard way, where a patient bootstrap that held the
fitted model fixed produced three confident verdicts a permutation test dismissed.

This measures both terms for the headline comparison, **multitaper + IF
trajectory vs the welch baseline**, currently reported as macro precision
+0.041 [+0.014, +0.067].

Method: run both arms on the same splits, keeping each split's **per-patient test
predictions** rather than only the summary metrics. Then bootstrap over patients:
draw 404 patients with replacement, and for every split recompute both arms'
precision using only the drawn patients that fall in that split's test fold,
carrying multiplicity. Average over splits, difference the arms, and read the
interval. Both arms see the identical patient draw and the identical folds, so
the comparison stays paired and only the patient sample varies.

Reported side by side:

  split-level    the existing interval -- "better on these 404 patients"
  patient-level  the new one           -- "better on patients like these"

The patient-level interval must be the wider of the two. How much wider is the
result.

Run: ``python -m experiments.patient_level_ci``
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.final_model import NBIN, SPLITS, TL, build
from models.architectures import (DescriptorFusion, ResidualTCN, Spectrum1DCNN,
                                  TRUNKS, TwoStreamNet)

NBOOT = 4000
NM = ("precN", "precPD", "precET", "macroP")


def evaluate_keep(spec, desc, traj, y, key, splits=SPLITS):
    """As ``final_model.evaluate``, but also returns per-split (te, pred)."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc]) if traj is None else \
        np.hstack([spec, desc, traj])
    if traj is None:
        mk1 = lambda: DescriptorFusion(Spectrum1DCNN(NBIN, 3, ch=8),
                                       TRUNKS["cnn"], NBIN, nd, 8 * 2 * 4)
    else:
        mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                   8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)

    per_split, rows = [], []
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
        per_split.append((te, pred))
        P, _, _, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        rows.append([P[0], P[1], P[2], P.mean()])
    return np.array(rows), per_split


def metrics_on(y_true, y_pred):
    P, _, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return np.array([P[0], P[1], P[2], P.mean()])


def patient_bootstrap(y, ps_a, ps_b, n_patients, n=NBOOT, seed=0):
    """Paired difference (b - a) resampling PATIENTS, splits held fixed.

    For each replicate a single patient multiset is drawn and used for **both**
    arms and **every** split, so fold assignment and model fitting contribute
    nothing to the spread -- only which patients were sampled.
    """
    rng = np.random.default_rng(seed)
    # position lookup per split, so a drawn patient can be located in its fold
    idx_a = [{p: i for i, p in enumerate(te)} for te, _ in ps_a]
    out = []
    for _ in range(n):
        draw = rng.integers(0, n_patients, n_patients)
        cnt = np.bincount(draw, minlength=n_patients)
        da, db = [], []
        for (te, pa), (_, pb), look in zip(ps_a, ps_b, idx_a):
            c = cnt[te]
            if c.sum() == 0:
                continue
            rep = np.repeat(np.arange(len(te)), c)
            yt = y[te][rep]
            if len(np.unique(yt)) < 2:
                continue
            da.append(metrics_on(yt, pa[rep]))
            db.append(metrics_on(yt, pb[rep]))
        if da:
            out.append(np.mean(db, 0) - np.mean(da, 0))
    return np.array(out)


def split_bootstrap(diff, n=NBOOT):
    out = []
    for i in range(diff.shape[1]):
        b = [np.mean(np.random.default_rng(s).choice(diff[:, i], len(diff),
                                                     replace=True))
             for s in range(n)]
        out.append(np.percentile(b, [2.5, 97.5]))
    return np.array(out)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key, SPEC = d["y"], d["key"], d["SPEC"]
    D_desc = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits, {NBOOT} bootstrap draws\n")

    print("running welch baseline ...", flush=True)
    a_rows, a_ps = evaluate_keep(SPEC["welch"], D_desc, None, y, key)
    print("running multitaper + trajectory ...", flush=True)
    b_rows, b_ps = evaluate_keep(SPEC["multitaper"], D_desc, d["TRAJ"], y, key)

    print(f"\n{'arm':>34}" + "".join(f"{c:>9}" for c in NM))
    print(f"{'welch + desc + asym (baseline)':>34}" +
          "".join(f"{v:>9.3f}" for v in a_rows.mean(0)))
    print(f"{'multitaper + trajectory':>34}" +
          "".join(f"{v:>9.3f}" for v in b_rows.mean(0)))

    diff = b_rows - a_rows
    sci = split_bootstrap(diff)
    pb = patient_bootstrap(y, a_ps, b_ps, len(y))
    pci = np.percentile(pb, [2.5, 97.5], axis=0).T

    print(f"\n{'':>10}{'diff':>9}{'split-level 95 %':>22}"
          f"{'patient-level 95 %':>24}{'  width x'}")
    for i, nm in enumerate(NM):
        lo_s, hi_s = sci[i]
        lo_p, hi_p = pci[i]
        s_s = "*" if lo_s > 0 or hi_s < 0 else " "
        s_p = "*" if lo_p > 0 or hi_p < 0 else " "
        w = (hi_p - lo_p) / (hi_s - lo_s + 1e-12)
        print(f"{nm:>10}{diff[:, i].mean():>+9.3f}"
              f"{f'[{lo_s:+.3f}, {hi_s:+.3f}] {s_s}':>22}"
              f"{f'[{lo_p:+.3f}, {hi_p:+.3f}] {s_p}':>24}{w:>9.1f}")

    print("\n  * = interval excludes zero. The patient-level column is the one a")
    print("    generalisation claim needs; the split-level column describes these")
    print("    404 patients only.")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
