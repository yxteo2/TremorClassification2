"""In-house 3-class model with the REST task: the one untested in-house lever.

`inhouse_pd_vs_et.md` closed OUT for in-house PD vs ET (in-house ET match PD
there) but found 2015 REST separates them, textbook direction, and listed
"in-house REST in the deep model" as untested. `_inhouse_rest_diagnostic.py`
is the cheap pre-check. This is the deep run, on the `own_data_10et` protocol
so the baseline is the published in-house number (0.652 / 0.769 / 0.193):
2015 + NewData only, 10 ET in every test set at natural prevalence, 20
repeats, PADS nowhere.

Arms, all on the SAME test patients (patients with both OUT and REST):

``OUT``        the reported recipe on postural recordings (baseline)
``REST``       the same recipe on rest recordings
``OUT+REST``   soft vote of the two (score fusion; the tasks are never
               averaged -- `task_averaging.md`)
``OUT+shufREST`` control: REST tables permuted across patients within
               cohort, redrawn per repeat (invariant 11). Same member count
               and feature distribution, no patient correspondence. The
               control decides attribution; ``OUT`` decides adoption.

PREDICTION (before the run): REST precET > OUT precET, and OUT+REST >
OUT+shufREST on precET, both by > +0.05 in the mean; precN lower for REST than
OUT (REST is the weaker N-vs-tremor task, 0.65 vs 0.84 on NewData). With 10 ET
per test set and 11 in training, expect CIs of +-0.08 on precET, so the
precET gain may not reach significance even if real.

SENSOR (env var). The pre-check, run before this file's first full run, put
the in-house PD-vs-ET signal on the HAND sensor only (REST 0.661 p=0.03,
OUT+REST 0.712 p=0.01; lower arm REST 0.556, OUT+REST 0.587, both at chance).
That is the best of nine arms and does not survive Bonferroni, so hand is the
pre-registered primary here and lower arm (the pipeline's sensor) is run as
the secondary. Updated prediction: the REST / OUT+REST precET gains above
appear on ``SENSOR=hand`` and not on ``SENSOR=lower``.

Run: ``SENSOR=hand python -m experiments.inhouse_rest_deep``
"""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import N_ASYM, NBIN, train, tune_offsets
from experiments._inhouse_rest_diagnostic import load_task
from experiments.final_model import method_table
from frequency.tables import spectrum_table
from models.architectures import ResidualTCN, Spectrum1DCNN, TRUNKS, TwoStreamNet
from signal_processing.stability import trajectory_table

REPEATS = int(os.environ.get("REPEATS", 20))
SENSOR = os.environ.get("SENSOR", "lower")
TL, ET_TEST = 64, 10
CH = {"hand": slice(0, 3), "lower": slice(3, 6), "upper": slice(6, 9)}[SENSOR]
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1")


def block(recs, has_bilat):
    from common.load_2025 import SIDE
    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    sp = spectrum_table(recs, ch=CH)
    spec, y, pats = method_table(recs, "multitaper", CH)
    spec = logbin(spec)
    traj, _, tp = trajectory_table(recs, ch=CH, n_out=TL)
    assert (tp == pats).all() and (sp[2] == pats).all()
    traj = traj.reshape(len(traj), -1)
    d = desc_table(recs, CH)
    if has_bilat:
        a, h = asym_for(recs, side_new, CH, sp[2])
    else:
        a, h = np.zeros((len(pats), N_ASYM)), np.zeros(len(pats))
    return spec, np.hstack([d, a, h[:, None]]), traj, y, pats


def task_tables(task):
    rA, rB = load_task(task)
    A, B = block(rA, False), block(rB, True)
    out = [np.concatenate([A[i], B[i]]) for i in range(5)]
    out.append(np.r_[np.zeros(len(A[3]), int), np.ones(len(B[3]), int)])
    return out                                  # spec, desc, traj, y, pats, coh


def align(T, pats):
    idx = {p: i for i, p in enumerate(T[4])}
    i = np.array([idx[p] for p in pats])
    return [t[i] for t in T]


def fit_probs(spec, desc, traj, y, tr, va, te, seeds=(0, 1, 2)):
    """The reported recipe (two-stream + ResidualTCN, 3 seeds); val/test probs."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    pv_l, pt_l = [], []
    for X, mk in ((packed, mk1), (spec, mk2)):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s) for s in seeds]
        pv_l.append(np.mean([a[0] for a in r], 0))
        pt_l.append(np.mean([a[1] for a in r], 0))
    return np.mean(pv_l, 0), np.mean(pt_l, 0)


def score(pv, pt, y, va, te):
    pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
    P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return [P[0], P[1], P[2], P.mean(), F.mean()], np.bincount(pred, minlength=3)


def main():
    torch.set_num_threads(1)
    O, R = task_tables("OUT"), task_tables("REST")
    pats = np.array(sorted(set(O[4]) & set(R[4])))
    O, R = align(O, pats), align(R, pats)
    assert (O[3] == R[3]).all() and (O[5] == R[5]).all()
    y, coh = O[3], O[5]
    from common.protocol import DEVICE
    print(f"sensor={SENSOR}  device={DEVICE}")
    print(f"patients with OUT and REST: n={len(y)}  N/PD/ET = "
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}")

    frac = ET_TEST / int((y == 2).sum())
    n_te = {0: int(round(frac * (y == 0).sum())),
            1: int(round(frac * (y == 1).sum())), 2: ET_TEST}
    print(f"test per repeat: N={n_te[0]} PD={n_te[1]} ET={n_te[2]}  ET prevalence "
          f"{n_te[2] / sum(n_te.values()):.3f};  {REPEATS} repeats\n", flush=True)

    arms = ("OUT", "REST", "OUT+REST", "OUT+shufREST")
    res = {k: [] for k in arms}
    npred = {k: [] for k in arms}
    for rep in range(REPEATS):
        rng = np.random.default_rng(rep)
        te, rest = [], []
        for c in (0, 1, 2):
            idx = np.flatnonzero(y == c); rng.shuffle(idx)
            te.extend(idx[:n_te[c]]); rest.extend(idx[n_te[c]:])
        te = np.array(sorted(te)); rest = np.array(sorted(rest)); rng.shuffle(rest)
        n_va = max(int(0.25 * len(rest)), 12)
        va, tr = np.sort(rest[:n_va]), np.sort(rest[n_va:])

        perm = np.arange(len(y))
        for c in (0, 1):
            i = np.flatnonzero(coh == c); perm[i] = rng.permutation(i)

        po = fit_probs(O[0], O[1], O[2], y, tr, va, te)
        pr = fit_probs(R[0], R[1], R[2], y, tr, va, te)
        ps = fit_probs(R[0][perm], R[1][perm], R[2][perm], y, tr, va, te)
        for k, (pv, pt) in (("OUT", po), ("REST", pr),
                            ("OUT+REST", ((po[0] + pr[0]) / 2, (po[1] + pr[1]) / 2)),
                            ("OUT+shufREST", ((po[0] + ps[0]) / 2, (po[1] + ps[1]) / 2))):
            m, n = score(pv, pt, y, va, te)
            res[k].append(m); npred[k].append(n)
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {res[k][-1][2]:.2f}"
                                            for k in arms), flush=True)

    print(f"\n{'arm':>14}" + "".join(f"{n:>9}" for n in NAMES) + "   ET preds/split")
    out = {}
    for k in arms:
        a = np.array(res[k]); out[k] = a
        print(f"{k:>14}" + "".join(f"{v:>9.3f}" for v in a.mean(0))
              + f"   {np.array(npred[k])[:, 2].mean():.1f}")
    for base, arm in (("OUT", "REST"), ("OUT", "OUT+REST"),
                      ("OUT+shufREST", "OUT+REST"), ("OUT", "OUT+shufREST")):
        d = out[arm] - out[base]
        print(f"\n{arm} - {base}  (paired, {REPEATS} repeats)")
        for i, nm in enumerate(NAMES):
            b = [np.random.default_rng(s).choice(d[:, i], len(d)).mean()
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {nm:>8} {d[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(d[:, i] > 0):.2f}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
