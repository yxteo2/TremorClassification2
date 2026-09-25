"""Give the reported model the in-house REST task — as its own block, not averaged.

`01_tremor_characteristics.ipynb` (follow-up in `inhouse_pd_vs_et.md`) found
that in-house PD vs ET carries **no signal at OUT** — the only task the in-house
cohorts feed the model — but **does at REST**: six frequency characteristics
read AUC ~0.65-0.70 on 2015 REST, above chance on all three sensors (16 ET), with
PD slower, the textbook direction. PADS runs the **opposite** way (ET slower
than PD at rest and in posture).

Earlier task studies (`task_averaging.md`, `rest_postural_contrast.md`) found
REST harmful, but they (a) *averaged* rest into the postural spectrum, which
dilutes a frequency shift into the task it is absent from, and (b) did it on the
merged cohort, where PADS -- with its reversed direction -- dominates. Neither
tested what the notebook points at.

## What changes

The REST task enters as a separate descriptor block, the way asymmetry already
does: the 10 repo descriptors (``desc_table``, same function and channels as the
postural ones) computed on each patient's REST recordings, plus a have-REST
flag. Patients without it get zeros and flag 0. Spectrum, trajectory and the TCN
member are untouched.

    reported9          adopted 9-member ensemble, postural only
    + REST in-house    REST block for 2015 + NewData; PADS flag 0
    + REST all         REST block for all three (PADS Relaxed filled in)
    CONTROL shuffled   in-house REST block with rows permuted among patients
                       of the same cohort, RE-DRAWN every split -- same values,
                       same dimensionality, no patient-label link

``+ REST in-house`` vs ``reported9`` decides adoption; vs ``CONTROL`` decides
whether the REST *content* or the extra columns do the work. ``+ REST all`` vs
``+ REST in-house`` tests the reversed-direction account: if PADS teaches the
opposite rule, filling it in should cost the in-house axis.

**Four `N 2` REST files hold accelerometer data** (|q| ~ 9.8, `verify_data`
check 13). They are not quaternions, so that patient's REST block is marked
missing (flag 0); the patient and their OUT data stay in.

## Metrics

The usual seven on the merged test fold, plus the in-house axis the notebook is
about, on in-house test patients only (about 4 ET per split):

    inAUC     PD-vs-ET AUC from P(ET) / (P(ET) + P(PD)), in-house PD+ET only
    inPrecET  in-house ET precision, inRecET in-house ET recall

## Prediction, recorded before the run

**+ REST in-house raises in-house PD-vs-ET AUC over both reported9 and the
shuffled control (by roughly +0.03 to +0.08), but no merged-fold metric moves
beyond its CI.** Reasons: the notebook signal is real but modest at 16 ET, and
in-house ET are ~40 % of the ~10 test ET per split, and a third of the ensemble
(TCN) never sees the block. **+ REST all is worse than + REST in-house on
inAUC**, because PADS teaches the reversed frequency rule. This is a 20-split
screen; a winner is extended to 40 before anything is claimed.

Run: ``python -m experiments.inhouse_rest``
"""

from __future__ import annotations

import re

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import experiments.final_model as FM
from common.cohorts import desc_table
from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import resume_load, resume_save
from experiments.alltasks_final import paired
from frequency.tables import spectrum_table
from models.architectures import (TRUNKS, ResidualTCN, Spectrum1DCNN,
                                  SpectrumTransformer, TwoStreamNet)

NM = ("precN", "precPD", "precET", "macroP", "macroF1", "recET", "nETpred",
      "inAUC", "inPrecET", "inRecET")
SPLITS = 20
SEEDS = (0, 1, 2)
TL = 64
ARMS = ("reported9", "+ REST in-house", "+ REST all", "CONTROL shuffled")
_TASK = re.compile(r"_(OUT|REST)$")


def pid(s):
    return _TASK.sub("", str(s))


def rest_block(recs, ch, order):
    """(len(order), 11): REST descriptors + have flag, zeros where absent.

    Files that are not unit quaternions (the N 2 accelerometer files) are
    skipped here, so their patient reads as missing rather than as garbage.
    """
    ok = []
    for r in recs:
        q = getattr(r, "path", None)
        if q is not None and "raw_quaternion" in str(q):
            import pandas as pd
            Q = pd.read_csv(q, sep=None, engine="python", header=None).to_numpy(float)
            nq = np.median(np.linalg.norm(Q.reshape(len(Q), -1, 4), axis=2))
            if not 0.95 < nq < 1.05:
                print(f"    REST file skipped, not quaternions: {q.name}")
                continue
        ok.append(r)
    pats = sorted({r.subject for r in ok})
    D = desc_table(ok, ch)                       # rows in sorted(subject) order
    idx = {pid(p): i for i, p in enumerate(pats)}
    out = np.zeros((len(order), D.shape[1] + 1))
    for j, p in enumerate(order):
        i = idx.get(pid(p))
        if i is not None:
            out[j, :-1], out[j, -1] = D[i], 1.0
    return out


def score(pt, off, yte, inh):
    pred = (np.log(pt + 1e-12) + off).argmax(1)
    P, R, F, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    m = inh & (yte != 0)
    ya = (yte[m] == 2).astype(int)
    s = pt[m, 2] / (pt[m, 1] + pt[m, 2] + 1e-12)
    auc = roc_auc_score(ya, s) if 0 < ya.sum() < len(ya) else np.nan
    pi = pred[inh]
    yi = yte[inh]
    ip = float(((pi == 2) & (yi == 2)).sum() / max((pi == 2).sum(), 1))
    ir = float(((pi == 2) & (yi == 2)).sum() / max((yi == 2).sum(), 1))
    return [P[0], P[1], P[2], P.mean(), F.mean(), R[2],
            float((pred == 2).sum()), auc, ip, ir]


def main():
    torch.set_num_threads(1)
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    d = FM.build()
    y, key = d["y"], d["key"]
    SPEC = d["SPEC"]["multitaper"]
    D0 = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    TR = d["TRAJ"]

    # patient order exactly as build() makes it: 2015, NewData, PADS[capped]
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    A, B, C = (spectrum_table(rA, ch=slice(3, 6)), spectrum_table(rB, ch=slice(3, 6)),
               spectrum_table(rC, ch=slice(0, 3)))
    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    assert np.array_equal(np.concatenate([A[1], B[1], C[1][keep]]), y), \
        "patient order does not match build()"
    assert np.allclose(desc_table(rA, slice(3, 6)), d["DESC"][:len(A[1])]), \
        "descriptor function does not reproduce build()"
    nA, nB = len(A[1]), len(B[1])
    coh = np.array(["2015"] * nA + ["NewData"] * nB + ["PADS"] * len(keep))
    inh = coh != "PADS"

    print("building REST blocks ...", flush=True)
    RA = rest_block(load_quaternion_recordings("Data", action="REST",
                                               mode="angular_velocity"),
                    slice(3, 6), A[2])
    RB = rest_block(load_2025_all(conditions=("REST",)), slice(3, 6), B[2])
    RC = rest_block(load_pads_extracted("pads_relaxed"), slice(0, 3), C[2][keep])
    R_in = np.vstack([RA, RB, np.zeros_like(RC)])
    R_all = np.vstack([RA, RB, RC])
    for c in ("2015", "NewData", "PADS"):
        m = coh == c
        print(f"  {c:<8} have REST {int(R_all[m, -1].sum())}/{int(m.sum())}   "
              f"ET with REST {int(R_all[m & (y == 2), -1].sum())}/"
              f"{int((m & (y == 2)).sum())}")
    print(f"n={len(y)}  splits={SPLITS}  in-house test patients ~"
          f"{int(inh.sum() * TEST_FRAC)}  (~{int((inh & (y == 2)).sum() * TEST_FRAC)} ET)")
    print("prediction on record: + REST in-house lifts inAUC +0.03..+0.08 over "
          "both reported9 and the control; merged metrics flat; + REST all < "
          "+ REST in-house on inAUC\n", flush=True)

    # assert first: the baseline arm must reproduce the adopted model's rows
    # from diverse_ensemble_2 on the same splits (first seven columns)
    ref, ref_done = resume_load("divens2_reported9", ("reported9",))
    ref = dict(zip(sorted(ref_done), ref["reported9"]))

    res, done = {}, {}
    for a in ARMS:
        r, dn = resume_load("inrest_" + a.replace(" ", "_"), (a,))
        res[a], done[a] = r[a], dn

    for sp in range(SPLITS):
        if all(sp in done[a] for a in ARMS):
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        # control: permute the in-house REST rows within cohort, new draw per split
        R_sh = R_in.copy()
        g = np.random.default_rng(1000 + sp)
        for c in ("2015", "NewData"):
            i = np.flatnonzero((coh == c) & (R_in[:, -1] == 1))
            R_sh[i] = R_in[g.permutation(i)]

        def fam(X, mk):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            o = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in SEEDS]
            return (np.mean([a[0] for a in o], 0), np.mean([a[1] for a in o], 0))

        tcn = fam(SPEC, lambda: ResidualTCN(NBIN, num_classes=3, ch=16))
        blocks = {"reported9": D0, "+ REST in-house": np.hstack([D0, R_in]),
                  "+ REST all": np.hstack([D0, R_all]),
                  "CONTROL shuffled": np.hstack([D0, R_sh])}
        for arm, D in blocks.items():
            if sp in done[arm]:
                continue
            nd = D.shape[1]
            packed = np.hstack([SPEC, D, TR])
            cnn = fam(packed, lambda: TwoStreamNet(
                Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"], 8 * 2 * 4, NBIN, nd, TL))
            tf = fam(packed, lambda: TwoStreamNet(
                SpectrumTransformer(NBIN, 3, d=32), TRUNKS["trunk"], 32, NBIN, nd, TL))
            pv = np.mean([cnn[0], tcn[0], tf[0]], 0)
            pt = np.mean([cnn[1], tcn[1], tf[1]], 0)
            res[arm].append(score(pt, tune_offsets(pv, y[va]), y[te], inh[te]))
            if arm == "reported9" and sp in ref:
                diff = np.abs(np.array(res[arm][-1][:7]) - np.array(ref[sp][:7]))
                assert diff.max() < 1e-9, \
                    f"split {sp}: baseline does not reproduce reported9 ({diff})"
                print(f"  split {sp}: baseline reproduces reported9 exactly",
                      flush=True)
            resume_save("inrest_" + arm.replace(" ", "_"), {arm: res[arm]}, sp)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    R = {a: np.array(res[a], float) for a in ARMS}
    print(f"\n{'arm':>18}" + "".join(f"{c:>9}" for c in NM))
    for a in ARMS:
        print(f"{a:>18}" + "".join(f"{v:>9.3f}" for v in np.nanmean(R[a], 0)))
    for base, tag in (("reported9", "ADOPTION"),
                      ("CONTROL shuffled", "ATTRIBUTION — REST content or "
                                           "extra columns?"),
                      ("+ REST in-house", "does PADS REST hurt the in-house "
                                          "axis?")):
        print(f"\npaired vs {base!r} ({tag}):")
        for a in ARMS:
            if a == base or (base == "+ REST in-house" and a != "+ REST all"):
                continue
            ok = ~np.isnan(R[a][:, 7]) & ~np.isnan(R[base][:, 7])
            print(f"  {a}:")
            for i, c in enumerate(NM):
                aa, bb = (R[a][ok], R[base][ok]) if i == 7 else (R[a], R[base])
                dd, lo, hi = paired(aa[:, [i]], bb[:, [i]])[0]
                star = "*" if lo > 0 or hi < 0 else " "
                w = float((aa[:, i] > bb[:, i]).mean())
                print(f"    {c:>9} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}"
                      f"   win {w:.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
