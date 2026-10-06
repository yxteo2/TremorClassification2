"""Ranking-aware fine-tuning losses for ET on the 2015 OUT transfer model (`ft`).

ET precision (~0.32 at prevalence 0.10, `verify_transfer_150.md`) is decided by
the TOP of the ET ranking: `tune_offsets` shifts logits by class and cannot
reorder patients. Class-weighted cross-entropy (the `ft` recipe) scores each
patient against its own label and never compares an ET patient with a non-ET
one. Closed already: logit adjustment, label smoothing, one-vs-rest ET detector
(harmful), WiSE-FT / SWA / SAM / LP-FT, checkpoint ensembling.

Every arm branches from the SAME PADS-pretrained weights per seed (pretraining
200 epochs, unchanged, identical to `common.protocol.train(pre=...)`) and the
torch RNG is reset to the post-pretraining state before each arm's fine-tune.
Arms differ ONLY in the fine-tuning training loss. The auxiliary terms act on
the ET margin

    m = z_ET - logsumexp(z_N, z_PD)          (monotone in p_ET)

computed full-batch over the training fold, on the same forward pass as the CE:

    ft    class-weighted CE only (the current recipe; bit-identical to
          common.protocol.train(pre=...), asserted by ``check``)
    auc   CE + 0.5 x pairwise squared-hinge AUC surrogate: mean over all
          (ET i, non-ET j) training pairs of max(0, 1 - (m_i - m_j))^2
          (Yuan et al., ICLR 2022 AUC-M style; their CE-then-AUC schedule is
          approximated by keeping CE throughout)
    pauc  CE + 0.5 x one-way partial-AUC surrogate: the same pairwise loss with
          the negatives restricted to the hardest 20 % of non-ET training
          patients by current margin (ceil(0.2 n_neg), top-k selected on the
          detached margin every step -- full batch, so every epoch; Zhu et al.,
          ICML 2022)
    ap    CE + 0.5 x (1 - smoothed AP) on the ET margin (Smooth-AP / the
          sigmoid relaxation SOAP builds on, Qi et al., NeurIPS 2021; full batch,
          so no moving-average estimators): for each ET i,
          rank_all(i) = 1 + sum_{j != i} sigmoid((m_j - m_i)/0.1),
          rank_pos(i) = 1 + sum_{j in ET, j != i} sigmoid((m_j - m_i)/0.1),
          AP = mean_i rank_pos / rank_all

Checkpoint rule identical across arms: the epoch with the lowest class-weighted
validation CE (first minimum), as in `train`. Weight 0.5, margin 1, k = 20 %,
temperature 0.1 are fixed here in advance; nothing is tuned on these
partitions. Members (two-stream + ResidualTCN) x 3 seeds, offsets tuned on
each arm's own validation probabilities.

Protocol = `transfer_2015` on the corrected 150-patient table (61 N / 74 PD /
15 ET): StratifiedKFold(5, shuffle, random_state=rep), 25 % stratified inner
validation (random_state rep*10+fold), PADS StretchHold capped 90/class redrawn
per repeat, `zfit` per domain. Repeats 0-19.

## Prediction, recorded before the run

The training fold has ~9 ET and ~81 non-ET patients; by the best-validation
epoch the network already ranks most training ET above most training non-ET,
so the pairwise terms act mainly as extra gradient on the few training ET that
sit inside the non-ET cloud -- i.e. an ET-focused reweighting of the hardest
ET, which fits those patients rather than finding new structure. On held-out
patients:

* No arm raises precET significantly at the PATIENT level (resolution ~+-0.03
  overall, wider for ET). Repeat level: precET |delta| <= 0.03 for every arm,
  macroF1 within +-0.015.
* auc and pauc: PD-vs-ET AUC within +-0.015 of ft; top-5 ET within +-0.05
  (n.s.). pauc is the noisier of the two (17 negatives per step) and, if
  anything, lower on precN / precPD (harder negatives are the N/PD patients
  with tremor-like spectra).
* ap: closest to ft (|delta| <= 0.01 on AUC, <= 0.02 on precET), because the
  sigmoid at temperature 0.1 saturates once training ET are separated and its
  gradient vanishes.
* The chosen epoch (best validation CE) moves earlier for auc/pauc (the aux
  term accelerates fitting of the training ET), median shift <= 10 epochs.
* Verdict expected: none adopted; plain `ft` stays.

Run: ``python -m experiments.rank_loss_150 check`` (bit-exactness + loss unit
tests), then ``REPS=0-20 python -m experiments.rank_loss_150`` (resumable, one
process, ``torch.set_num_threads(1)``), then
``python -m experiments.rank_loss_150 report``.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, members, zfit

ARMS = ("ft", "auc", "pauc", "ap")
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET",
         "aucETrest")
OUT_DIR = os.environ.get("OUT_DIR", "rank_loss_150_runs")
EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80
LAM, MARGIN, PAUC_FRAC, AP_TAU, ET = 0.5, 1.0, 0.20, 0.1, 2


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


# ---------------------------------------------------------------- aux losses
def et_margin(z):
    """m = z_ET - logsumexp(z_N, z_PD)."""
    return z[:, ET] - torch.logsumexp(z[:, :ET], 1)


def loss_auc(m, pos, neg):
    """Mean over (ET, non-ET) pairs of max(0, 1 - (m_i - m_j))^2."""
    d = m[pos][:, None] - m[neg][None, :]
    return torch.clamp(MARGIN - d, min=0).pow(2).mean()


def loss_pauc(m, pos, neg):
    """Pairwise loss against the hardest ceil(0.2 n_neg) negatives (by margin)."""
    k = max(1, math.ceil(PAUC_FRAC * len(neg)))
    hard = neg[torch.topk(m[neg].detach(), k).indices]
    return loss_auc(m, pos, hard)


def loss_ap(m, pos):
    """1 - smoothed AP of the ET margin, sigmoid temperature AP_TAU."""
    mp = m[pos]
    s_all = torch.sigmoid((m[None, :] - mp[:, None]) / AP_TAU)      # (P, n)
    s_pos = torch.sigmoid((mp[None, :] - mp[:, None]) / AP_TAU)     # (P, P)
    # self terms are sigmoid(0) = 0.5: 1 + sum_{j != i} = 0.5 + sum_j
    r_all = 0.5 + s_all.sum(1)
    r_pos = 0.5 + s_pos.sum(1)
    return 1.0 - (r_pos / r_all).mean()


def aux_fn(arm, yt):
    pos = torch.nonzero(yt == ET).flatten()
    neg = torch.nonzero(yt != ET).flatten()
    if arm == "auc":
        return lambda z: loss_auc(et_margin(z), pos, neg)
    if arm == "pauc":
        return lambda z: loss_pauc(et_margin(z), pos, neg)
    if arm == "ap":
        return lambda z: loss_ap(et_margin(z), pos)
    return None


# ---------------------------------------------------------------- training
def _finetune(m, xt, yt, xv, yv, w, arm):
    """common.protocol.train's fine-tuning loop; only the training loss differs."""
    lf = nn.CrossEntropyLoss(weight=w)
    aux = aux_fn(arm, yt)
    opt = torch.optim.AdamW(list(m.parameters()), lr=FT_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
    best, state, best_ep = np.inf, None, -1
    for ep in range(FT_EPOCHS):
        m.train(); opt.zero_grad()
        z = m(xt)
        loss = lf(z, yt)
        if aux is not None:
            loss = loss + LAM * aux(z)
        loss.backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lf(m(xv), yv))       # checkpoint rule: validation CE only
        if v < best:
            best, best_ep = v, ep
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    return m, best_ep


def train_arms(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed, arms=ARMS):
    """Pretrain once exactly as common.protocol.train, then fine-tune each arm."""
    torch.manual_seed(seed)
    xt, yt, xv, yv = _T(Xtr), _L(ytr), _T(Xva), _L(yva)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    xp, ypt = _T(Xp), _L(yp)
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    m.train()
    for _ in range(EPOCHS):
        op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
    rng = torch.get_rng_state()
    w = _wt(ytr)
    out, eps = {}, {}
    for a in arms:
        torch.set_rng_state(rng)
        ma, eps[a] = _finetune(copy.deepcopy(m), xt, yt, xv, yv, w, a)
        with torch.no_grad():
            out[a] = [torch.softmax(ma(_T(z)), 1).cpu().numpy() for z in Xout]
    return out, eps


def _pads(C, rep):
    rng = np.random.default_rng(rep)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                              min(CAP, int((C[3] == c).sum())),
                                              replace=False) for c in (0, 1, 2)]))


def _folds(spec, y, rep):
    for fold, (rest_i, te) in enumerate(
            StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
        tr, va = train_test_split(rest_i, test_size=0.25, stratify=y[rest_i],
                                  random_state=rep * 10 + fold)
        yield fold, np.sort(tr), np.sort(va), te


def _unit_tests():
    m = torch.tensor([3.0, 2.5, 0.0, -1.0, 1.4, -2.0])
    pos, neg = torch.tensor([0, 1]), torch.tensor([2, 3, 4, 5])
    # pairs violating margin 1: (1, 4): 1 - 1.1 < 0 -> 0; all others separated
    assert float(loss_auc(m, pos, neg)) == 0.0
    m2 = torch.tensor([0.0, 2.0, 0.5, -1.0])
    p2, n2 = torch.tensor([0, 1]), torch.tensor([2, 3])
    ref = np.mean([max(0, 1 - (a - b)) ** 2 for a in (0.0, 2.0) for b in (0.5, -1.0)])
    assert abs(float(loss_auc(m2, p2, n2)) - ref) < 1e-6
    # pauc: hardest 20 % of 4 negatives -> ceil(0.8) = 1 negative, the 1.4 one
    refp = np.mean([max(0, 1 - (a - 1.4)) ** 2 for a in (3.0, 2.5)])
    assert abs(float(loss_pauc(m, pos, neg)) - refp) < 1e-6
    # ap: well separated -> ~0; reversed -> large
    big = torch.tensor([50.0, 40.0, 0.0, -10.0])
    assert float(loss_ap(big, torch.tensor([0, 1]))) < 1e-6
    rev = torch.tensor([-50.0, -40.0, 0.0, 10.0])
    # exact AP of a ranking with both positives last among 4: (1/3 + 2/4) / 2
    assert abs(float(loss_ap(rev, torch.tensor([0, 1]))) - (1 - (1 / 3 + 2 / 4) / 2)) < 1e-4
    print("loss unit tests OK", flush=True)


def check():
    """Assert the `ft` arm equals common.protocol.train(pre=...) bit for bit."""
    from common.protocol import train
    torch.set_num_threads(1)
    _unit_tests()
    A, _, C = build()
    spec, desc, traj, y = A
    assert len(y) == 150, len(y)
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    kp = _pads(C, 0)
    pads = {"packed": Cp[kp], "spec": C[0][kp]}
    _, tr, va, te = next(_folds(spec, y, 0))
    for key, mk in (("packed", mk1), ("spec", mk2)):
        X, P = own[key], pads[key]
        f = zfit(X[tr])
        pre = (zfit(P)(P), C[3][kp])
        ref = train(mk, f(X[tr]), y[tr], f(X[va]), y[va], [f(X[va]), f(X[te])],
                    seed=0, pre=pre)
        out, eps = train_arms(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                              [f(X[va]), f(X[te])], pre, 0)
        assert np.array_equal(ref[0], out["ft"][0]) and \
            np.array_equal(ref[1], out["ft"][1]), key
        diff = {a: float(np.abs(out[a][1] - out["ft"][1]).max()) for a in ARMS[1:]}
        print(f"{key}: ft arm bit-identical to common.protocol.train; chosen epochs "
              f"{eps}; max |p_arm - p_ft| on test {diff}", flush=True)
    print("CHECK_OK", flush=True)


def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk1, mk2 = members(desc.shape[1])
    print(f"device={DEVICE}  2015 OUT n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}  arms={ARMS}", flush=True)
    assert len(y) == 150, len(y)
    os.makedirs(OUT_DIR, exist_ok=True)
    for rep in reps:
        if os.path.exists(f"{OUT_DIR}/rep{rep:02d}.npz"):
            continue
        kp = _pads(C, rep)
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {k: np.zeros((len(y), 3)) for k in ARMS}
        pred = {k: np.full(len(y), -1) for k in ARMS}
        ep = {k: np.zeros((5, 2, 3), int) for k in ARMS}
        for fold, tr, va, te in _folds(spec, y, rep):
            pv = {k: [] for k in ARMS}
            pt = {k: [] for k in ARMS}
            for j, (key, mk) in enumerate((("packed", mk1), ("spec", mk2))):
                X, P = own[key], pads[key]
                f = zfit(X[tr])
                sv = {k: [] for k in ARMS}
                st = {k: [] for k in ARMS}
                for s in (0, 1, 2):
                    out, eps = train_arms(mk, f(X[tr]), y[tr], f(X[va]), y[va],
                                          [f(X[va]), f(X[te])], (zfit(P)(P), C[3][kp]), s)
                    for k in ARMS:
                        sv[k].append(out[k][0]); st[k].append(out[k][1])
                        ep[k][fold, j, s] = eps[k]
                for k in ARMS:
                    pv[k].append(np.mean(sv[k], 0)); pt[k].append(np.mean(st[k], 0))
            for k in ARMS:
                v, t = np.mean(pv[k], 0), np.mean(pt[k], 0)
                prob[k][te] = t
                pred[k][te] = (np.log(t + 1e-12) + tune_offsets(v, y[va])).argmax(1)
        tmp = f"{OUT_DIR}/partial_{rep:02d}.npz"
        np.savez(tmp, y=y, **pred, **{f"p_{k}": v for k, v in prob.items()},
                 **{f"ep_{k}": v for k, v in ep.items()})
        os.replace(tmp, f"{OUT_DIR}/rep{rep:02d}.npz")
        Pr = {k: precision_recall_fscore_support(y, pred[k], labels=[0, 1, 2],
                                                 zero_division=0)[0] for k in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{k} ET {Pr[k][2]:.2f}" for k in ARMS)
              + "  | ep med " + " ".join(f"{k} {np.median(ep[k]):.0f}" for k in ARMS),
              flush=True)


def _metrics(y, pred, pr):
    m = y != 0
    P, _, F, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
    o = np.argsort(-pr[:, 2], kind="stable")
    return [P[0], P[1], P[2], P.mean(), F.mean(),
            roc_auc_score((y[m] == 2).astype(int), s),
            (y[o[:5]] == 2).mean(),
            roc_auc_score((y == 2).astype(int), pr[:, 2])]


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep[0-9][0-9].npz"))
    res = {k: [] for k in ARMS}
    E = {k: [] for k in ARMS}
    npred = {k: [] for k in ARMS}
    for f in files:
        d = np.load(f)
        y = d["y"]
        for k in ARMS:
            res[k].append(_metrics(y, d[k], d[f"p_{k}"]))
            E[k].append(d[f"ep_{k}"]); npred[k].append(int((d[k] == 2).sum()))
    n = len(files)
    R = {k: np.array(v) for k, v in res.items()}
    print(f"2015 OUT (n={len(y)}), ranking-aware fine-tuning losses, {n} repeats\n")
    print(f"{'arm':>6}" + "".join(f"{c:>10}" for c in NAMES) + "  sd(precET)  ET preds/rep")
    for k in ARMS:
        print(f"{k:>6}" + "".join(f"{v:>10.3f}" for v in R[k].mean(0))
              + f"  {R[k][:, 2].std():.3f}       {np.mean(npred[k]):.1f}")
    print("\nchosen fine-tuning epoch (0-based, of 80) per member [two-stream, TCN]:")
    for k in ARMS:
        a = np.stack(E[k])                                  # (n, 5, 2, 3)
        print(f"  {k:>6}: " + "   ".join(
            f"{nm} median {np.median(a[:, :, j]):.0f} IQR "
            f"{np.percentile(a[:, :, j], 25):.0f}-{np.percentile(a[:, :, j], 75):.0f}"
            for j, nm in enumerate(("two-stream", "TCN"))))
    for a in ARMS[1:]:
        dd = R[a] - R["ft"]
        print(f"\n{a} - ft  (paired over repeats, {n}; win = share of repeats with "
              f"delta > 0, ties = share equal)")
        for i, c in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean()
                  for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"  {c:>9} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}] {star}"
                  f"  win {np.mean(dd[:, i] > 0):.2f}  ties {np.mean(dd[:, i] == 0):.2f}")
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    for a in ARMS[1:]:
        bootstrap_dir(OUT_DIR, a, "ft")
    _patient_top5_f1(files)
    print("\nMARKER_DONE", flush=True)


def _patient_top5_f1(files, B=2000):
    """Patient-level bootstrap for the metrics bootstrap_dir does not cover.

    Same scheme (class-stratified patient draws with replacement, seed 0, every
    repeat scored on the draw, per-repeat metric averaged, paired difference):
    macroF1 with multiplicity weights, top-5 ET precision on the resampled
    cohort (duplicated patients are separate entries; ranking by p_ET, stable).
    """
    D = [np.load(f) for f in files]
    y = D[0]["y"]
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(y == c) for c in range(3)]

    def score(idx, w):
        out = {}
        for k in ARMS:
            f1, t5 = [], []
            for d in D:
                _, _, F, _ = precision_recall_fscore_support(
                    y, d[k], labels=[0, 1, 2], sample_weight=w, zero_division=0)
                s = d[f"p_{k}"][idx, 2]
                o = np.argsort(-s, kind="stable")[:5]
                f1.append(F.mean()); t5.append((y[idx][o] == 2).mean())
            out[k] = np.array([np.mean(f1), np.mean(t5)])
        return out

    draws = {k: [] for k in ARMS[1:]}
    for _ in range(B):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        w = np.bincount(idx, minlength=len(y)).astype(float)
        sc = score(idx, w)
        for k in ARMS[1:]:
            draws[k].append(sc[k] - sc["ft"])
    pt = score(np.arange(len(y)), np.ones(len(y)))
    for k in ARMS[1:]:
        a = np.array(draws[k])
        lo, hi = np.percentile(a, [2.5, 97.5], 0)
        print(f"\npatient-level (macroF1, top-5 ET): {k} − ft  ({len(D)} repeats, {B} draws)")
        for i, nm in enumerate(("macroF1", "top5ET")):
            star = "*" if lo[i] > 0 or hi[i] < 0 else " "
            print(f"   {nm:>7} {pt[k][i] - pt['ft'][i]:+.3f}  patient-level 95% "
                  f"[{lo[i]:+.3f}, {hi[i]:+.3f}] {star}")


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    elif sys.argv[1:] == ["check"]:
        check()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
