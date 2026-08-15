"""Strategies for combining the 2015 / NewData / PADS cohorts.

Every result so far POOLED the three cohorts, with the only cohort-specific
choice being how hard to cap PADS. Established already (`merge_design.md`,
`deep_model_improvement.md`):

* capping PADS at 90/class beats 30, 60 and uncapped for a 2015 target;
* per-cohort distribution alignment (z-score, rank, CORAL) all HURT, and all
  make cohort identity *more* detectable rather than less;
* dropping PADS entirely collapses ET precision from 0.519 to 0.065.

Four standard strategies had never been tested, each targeting a different part
of the problem:

``weight``      per-sample weights of 1/cohort-size instead of capping. Capping
                discards 113 of 383 PADS patients purely to stop PADS
                dominating; weighting achieves the same balance keeping all of
                them.
``cohort_id``   cohort identity as an explicit one-hot input. The probe shows
                the sites are nearly indistinguishable from features, yet they
                differ sharply in class balance (PADS 72 % PD against a balanced
                2015), so the model cannot calibrate per site unless told.
                **Legitimate only under the mixed protocol** -- under
                leave-one-cohort-out the test site is unseen and this leaks.
``per_cohort_priors``
                validation-tuned logit offsets fitted per cohort rather than
                one global set. Prior tuning was the largest single gain in the
                session; one offset across three differently-balanced cohorts is
                the obvious thing to refine.
``finetune``    pretrain on the PADS patients of the training split, then
                fine-tune at a lower learning rate on 2015 + NewData. The
                standard way to use a large auxiliary cohort; PADS has only ever
                been poured into the same pool here.

Run: ``python -m tfbench.cohort_strategies``
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from tfbench.descriptors import DESCRIPTOR_NAMES, describe
from tfbench.small_nets import (DescriptorFusion, ResidualTCN, Spectrum1DCNN,
                                TRUNKS, asym_feats, bilateral_table,
                                spectrum_table)
from tfbench.transforms import METHODS

NAMES = ("2015", "NewData", "PADS")
NBIN, N_ASYM = 16, 4
ASYM_KEEP = [0, 1, 2, 5]        # unsigned only: corr, cos, peak_df, l1
SPLITS, TEST_FRAC, VAL_FRAC = 10, 0.20, 0.20


def desc_table(recs, ch):
    fn = METHODS["stft512"]
    rows = defaultdict(list)
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        rows[r.subject].append([describe(*fn(x))[c] for c in DESCRIPTOR_NAMES])
    return np.nan_to_num(np.array([np.mean(rows[p], 0) for p in sorted(rows)]))


def logbin(X, nb=NBIN):
    L = np.log(X + 1e-8)
    n = X.shape[1] // nb * nb
    return L[:, :n].reshape(len(L), nb, -1).mean(2)


def asym_for(recs, side_fn, ch, patients):
    """Unsigned asymmetry per patient, zero where the second limb is missing.

    Returned with an availability vector so a zero is never read as "perfectly
    symmetric" -- 2015 is single-limb and contributes none.
    """
    Xb, _, pats = bilateral_table(recs, side_fn, ch=ch)
    A = asym_feats(Xb)[:, ASYM_KEEP]
    idx = {p: i for i, p in enumerate(pats)}
    out = np.zeros((len(patients), N_ASYM))
    have = np.zeros(len(patients))
    for j, p in enumerate(patients):
        if p in idx:
            out[j], have[j] = A[idx[p]], 1.0
    return np.nan_to_num(out), have


def train(model_fn, Xtr, ytr, Xva, yva, Xout, seed=0, epochs=200, lr=3e-3,
          wd=1e-3, nc=3, sw=None, pre=None, ft_lr=1e-3, ft_epochs=80):
    """Full-batch trainer. ``sw`` weights samples; ``pre`` pretrains first."""
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.long)
    xv, yv = T(Xva), torch.tensor(yva, dtype=torch.long)
    m = model_fn()

    def wt(yy):
        c = np.bincount(yy, minlength=nc).astype(float)
        return torch.tensor(c.sum() / (nc * np.maximum(c, 1)), dtype=torch.float32)

    if pre is not None:
        Xp, yp = pre
        xp, ypt = T(Xp), torch.tensor(yp, dtype=torch.long)
        lf = nn.CrossEntropyLoss(weight=wt(yp))
        op = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, epochs)
        m.train()
        for _ in range(epochs):
            op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
        lr, epochs = ft_lr, ft_epochs

    w = wt(ytr)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    swt = T(sw) if sw is not None else None
    best, state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        if swt is None:
            loss = nn.CrossEntropyLoss(weight=w)(m(xt), yt)
        else:
            per = nn.CrossEntropyLoss(weight=w, reduction="none")(m(xt), yt)
            loss = (per * swt).sum() / swt.sum()
        loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(nn.CrossEntropyLoss(weight=w)(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return [torch.softmax(m(T(z)), 1).numpy() for z in Xout]


def tune_offsets(pv, yv):
    """Per-class logit offsets maximising VALIDATION macro F1."""
    best, bo = -1.0, np.zeros(3)
    lp = np.log(pv + 1e-12)
    for b1 in np.linspace(-1, 1, 9):
        for b2 in np.linspace(-1, 1, 9):
            o = np.array([0.0, b1, b2])
            _, _, F, _ = precision_recall_fscore_support(
                yv, (lp + o).argmax(1), labels=[0, 1, 2], zero_division=0)
            if F.mean() > best:
                best, bo = F.mean(), o
    return bo


def run(name, sb, dc, y, key, coh, mode="pool", per_cohort_priors=False,
        splits=SPLITS, verbose=True):
    out = []
    Xf, Xs = np.hstack([sb, dc]), sb
    n_desc = Xf.shape[1] - NBIN
    builders = [
        (Xf, lambda: DescriptorFusion(Spectrum1DCNN(NBIN, 3, ch=8),
                                      TRUNKS["cnn"], NBIN, n_desc, 8 * 2 * 4)),
        (Xs, lambda: ResidualTCN(NBIN, num_classes=3, ch=16)),
    ]
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(sb, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(sb[tv], key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, mk in builders:
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            kw, tr_use = {}, tr
            if mode == "weight":
                cnt = {c: max((coh[tr] == c).sum(), 1) for c in NAMES}
                kw["sw"] = np.array([1.0 / cnt[c] for c in coh[tr]])
            elif mode == "finetune":
                p, q = tr[coh[tr] == "PADS"], tr[coh[tr] != "PADS"]
                if len(p) > 20 and len(q) > 20:
                    kw["pre"] = ((X[p] - mu) / sd, y[p])
                    tr_use = q
            r = [train(mk, (X[tr_use] - mu) / sd, y[tr_use], (X[va] - mu) / sd,
                       y[va], [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s, **kw)
                 for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        if per_cohort_priors:
            pred = np.zeros(len(te), int)
            for c in NAMES:
                mv, mt = coh[va] == c, coh[te] == c
                if not mt.any():
                    continue
                off = tune_offsets(pv[mv], y[va][mv]) if mv.sum() >= 8 \
                    else tune_offsets(pv, y[va])
                pred[mt] = (np.log(pt[mt] + 1e-12) + off).argmax(1)
        else:
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    a = np.array(out)
    if verbose:
        m, s = a.mean(0), a.std(0)
        print(f"{name:>36}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a


def load_all(cap=90):
    """Assemble the merged table with asymmetry as a missing modality."""
    from pdetn.crossdataset import load_pads_extracted
    from pdetn.load_2025 import SIDE, load_2025_all
    from tremor.quaternion_data import load_quaternion_recordings

    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    side_pads = lambda r: ("left" if "LeftWrist" in str(r.path)
                           else ("right" if "RightWrist" in str(r.path) else None))
    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    A, B, C = (spectrum_table(rA, ch=slice(3, 6)), spectrum_table(rB, ch=slice(3, 6)),
               spectrum_table(rC, ch=slice(0, 3)))
    DA, DB, DC = (desc_table(rA, slice(3, 6)), desc_table(rB, slice(3, 6)),
                  desc_table(rC, slice(0, 3)))
    aB, hB = asym_for(rB, side_new, slice(3, 6), B[2])
    aC, hC = asym_for(rC, side_pads, slice(0, 3), C[2])
    if cap is None:
        keep = np.arange(len(C[1]))
    else:
        rng = np.random.default_rng(0)
        keep = []
        for c in (0, 1, 2):
            i = np.flatnonzero(C[1] == c)
            keep.extend(rng.choice(i, min(cap, len(i)), replace=False))
        keep = np.array(sorted(keep))
    sb = logbin(np.vstack([A[0], B[0], C[0][keep]]))
    dc = np.hstack([np.vstack([DA, DB, DC[keep]]),
                    np.vstack([np.zeros((len(A[1]), N_ASYM)), aB, aC[keep]]),
                    np.concatenate([np.zeros(len(A[1])), hB, hC[keep]])[:, None]])
    y = np.concatenate([A[1], B[1], C[1][keep]])
    coh = np.concatenate([np.full(len(A[1]), "2015"),
                          np.full(len(B[1]), "NewData"),
                          np.full(len(keep), "PADS")])
    key = np.array([f"{c}_{l}" for c, l in zip(coh, y)])
    return sb, dc, y, key, coh


def main():
    torch.set_num_threads(1)
    H = (f"{'strategy':>36}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
         f"{'macroF1':>9}  |{'  sd':>7}")
    print(f"welch, asymmetry included, {SPLITS} splits, per-class precision\n")
    print(H)
    sb, dc, y, key, coh = load_all(cap=90)
    print(f"  [cap 90] n={len(y)}  "
          f"ET={int((y == 2).sum())}  bilateral={int(dc[:, -1].sum())}")
    res = {}
    res["baseline"] = run("cap 90 + global priors (baseline)", sb, dc, y, key, coh)
    res["percoh"] = run("cap 90 + PER-COHORT priors", sb, dc, y, key, coh,
                        per_cohort_priors=True)
    res["ft"] = run("cap 90 + PADS-pretrain/finetune", sb, dc, y, key, coh,
                    mode="finetune")
    oh = np.stack([(coh == c).astype(float) for c in NAMES], 1)
    res["cid"] = run("cap 90 + cohort-ID input", sb, np.hstack([dc, oh]), y, key, coh)

    sbA, dcA, yA, keyA, cohA = load_all(cap=None)
    print(f"  [uncapped] n={len(yA)}  ET={int((yA == 2).sum())}")
    res["unc"] = run("uncapped, unweighted", sbA, dcA, yA, keyA, cohA)
    res["w"] = run("uncapped + SAMPLE WEIGHTS", sbA, dcA, yA, keyA, cohA,
                   mode="weight")
    res["wp"] = run("uncapped + weights + per-coh priors", sbA, dcA, yA, keyA,
                    cohA, mode="weight", per_cohort_priors=True)

    NM = ("precN", "precPD", "precET", "macroP", "macroF1")
    print("\npaired vs cap-90 baseline, same splits (bootstrap 95 % CI):")
    for k, lbl in (("percoh", "per-cohort priors"), ("ft", "PADS pretrain/finetune"),
                   ("cid", "cohort-ID input")):
        d = res[k] - res["baseline"]
        print(f"  {lbl}:")
        for i, nm in enumerate(NM):
            b = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                         replace=True))
                 for s in range(4000)]
            lo, hi = np.percentile(b, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
