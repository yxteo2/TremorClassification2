"""Stop PADS pretraining where it transfers best -- chosen inside the training fold.

Background (`loss_curves_2015.md`): during the 200-epoch cosine PADS
pretraining of the `ft` recipe, the two-stream member's weighted CE on 2015
patients (never trained on) is lowest at epoch ~18 and rises afterwards -- it
overfits PADS; the TCN's bottoms near epoch 76 and stays flat. Fixed shorter
pretraining (100 epochs, which also compresses the cosine schedule) was mixed
(`transfer_2015.md`: precPD +0.021 \\*, AUC -0.020 \\*); 400 epochs was worse.

Here the 200-epoch cosine schedule is kept and the weights are snapshotted every
5 epochs. Every arm comes from the SAME pretraining trajectory per (fold,
member, seed) and is then fine-tuned with the standard recipe (AdamW lr 1e-3,
cosine 80 epochs, best-validation-loss checkpoint). The torch RNG state at the
end of pretraining is restored before each fine-tune, so the `ft` arm is
bit-identical to `common.protocol.train(..., pre=...)` (asserted on the first
model) and all arms share the same dropout stream.

Arms (the snapshot epoch is chosen per member, per seed, per fold):
    ft          epoch 200 (= the current recipe)
    stop_val    lowest weighted CE (class weights of the training fold) of the
                pretrained 3-class PADS head on the fold's 2015 VALIDATION
                patients -- no 2015 training
    stop_logme  highest LogME (You et al., ICML 2021; numpy port of
                thuml/LogME `_fit_icml`) of the penultimate-layer features
                (input of the last nn.Linear, eval mode) of the fold's 2015
                TRAINING patients with their labels
    stop_fixed  20 (two-stream) / 75 (TCN), read off the loss curves. BIASED
                reference only: those curves included test data of repeats 0-4
                and, through CV, every patient.

Protocol: `transfer_2015` -- corrected 150-patient 2015 OUT table (61 N / 74 PD /
15 ET), StratifiedKFold(5, shuffle, random_state=rep), stratified 25 % inner
validation (random_state=rep*10+fold), PADS capped 90/class redrawn per repeat
(rng(rep)), per-domain `zfit`, members = two-stream + ResidualTCN x seeds 0-2,
soft vote, `tune_offsets` on validation. 20 repeats (seeds 0-19), CPU, one
thread. Nothing below is chosen from test-fold data except the documented
stop_fixed reference.

## Prediction, recorded before the run

* Chosen epochs: stop_val two-stream median 15-40 with a wide spread (val CE on
  ~30 patients is noisy); TCN later and broader (median 50-150). stop_logme
  picks late epochs (median >= 100 for both members): PADS training keeps
  making features more class-separable, and LogME has no calibration term to
  penalise the over-confident head that drives the 2015 CE up.
* stop_val vs ft (like the 100-epoch pretrain, more so for the two-stream):
  precPD +0.01 to +0.03, PD-vs-ET AUC -0.01 to -0.03, precET within +-0.03,
  macroP within +-0.015. stop_fixed similar in direction and slightly larger.
* stop_logme vs ft: null on every metric (|diff| < 0.015).
* Patient level: no arm's interval excludes 0 on precET or macroP. Verdict
  expected: nothing adopted; `ft` (epoch 200) stays.

Run (resumable, one repeat per file):
    nohup python -m experiments.pretrain_stop_150 > pretrain_stop_150.log 2>&1 &
    python -m experiments.pretrain_stop_150 report
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common.protocol import DEVICE, train, tune_offsets
from experiments.own_data_10et import build
from experiments.transfer_2015 import CAP, members, zfit

EPOCHS, LR, WD, FT_LR, FT_EPOCHS = 200, 3e-3, 1e-3, 1e-3, 80   # = protocol.train
EVERY = 5
SNAPS = np.arange(EVERY, EPOCHS + 1, EVERY)                     # 5, 10, ..., 200
FIXED = {"two": 20, "tcn": 75}
ARMS = ("ft", "stop_val", "stop_logme", "stop_fixed")
MEMBERS = ("two", "tcn")
SEEDS = (0, 1, 2)
NAMES = ("precN", "precPD", "precET", "macroP", "macroF1", "aucPDET", "top5ET")
OUT_DIR = os.environ.get("OUT_DIR", "pretrain_stop_150_runs")


def _T(z):
    return torch.tensor(z, dtype=torch.float32, device=DEVICE)


def _L(z):
    return torch.tensor(z, dtype=torch.long, device=DEVICE)


def _wt(yy, nc=3):
    c = np.bincount(yy, minlength=nc).astype(float)
    return _T(c.sum() / (nc * np.maximum(c, 1)))


# ---------------------------------------------------------------- LogME
def _each_evidence(y_, f, fh, v, s, vh, N, D):
    """thuml/LogME `each_evidence` (ICML 2021 version), numpy."""
    eps = 1e-5
    alpha, beta = 1.0, 1.0
    lam = alpha / beta
    tmp = vh @ (f @ y_)
    for _ in range(11):
        gamma = (beta * s / (alpha + beta * s)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + eps)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + eps)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    ev = (D / 2.0 * np.log(alpha) + N / 2.0 * np.log(beta)
          - 0.5 * np.sum(np.log(alpha + beta * s))
          - beta / 2.0 * (beta_de + eps) - alpha / 2.0 * (alpha_de + eps)
          - N / 2.0 * np.log(2 * np.pi))
    return ev / N


def logme(features, y, nc=3):
    """Mean per-class (one-hot) log marginal evidence of a Bayesian linear model."""
    fh = features.astype(np.float64)
    f = fh.T
    D, N = f.shape
    v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)
    return float(np.mean([_each_evidence((y == k).astype(np.float64), f, fh, v, s, vh,
                                         N, D) for k in range(nc)]))


def _head(m):
    return [x for x in m.modules() if isinstance(x, nn.Linear)][-1]


# ---------------------------------------------------------------- one trajectory
def trajectory(model_fn, Xtr, ytr, Xva, yva, Xout, pre, seed, fixed_ep):
    """protocol.train with pretraining, snapshotting every EVERY epochs.

    Returns {arm: [probs for each of Xout]}, chosen epochs, val-CE and LogME curves.
    """
    torch.manual_seed(seed)
    xt, yt, xv, yv = _T(Xtr), _L(ytr), _T(Xva), _L(yva)
    m = model_fn().to(DEVICE)
    Xp, yp = pre
    xp, ypt = _T(Xp), _L(yp)
    lf = nn.CrossEntropyLoss(weight=_wt(yp))
    lw = nn.CrossEntropyLoss(weight=_wt(ytr))
    op = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sp = torch.optim.lr_scheduler.CosineAnnealingLR(op, EPOCHS)
    feats = {}
    hook = _head(m).register_forward_hook(lambda mod, inp, out: feats.__setitem__("f", inp[0]))
    snaps, vce, lme = {}, [], []
    m.train()
    for e in range(EPOCHS):
        op.zero_grad(); lf(m(xp), ypt).backward(); op.step(); sp.step()
        if (e + 1) % EVERY == 0:
            # eval-mode forwards change neither BN running stats nor the RNG
            m.eval()
            with torch.no_grad():
                vce.append(float(lw(m(xv), yv)))
                m(xt)
                lme.append(logme(feats["f"].cpu().numpy(), ytr))
            snaps[e + 1] = {k: t.detach().clone() for k, t in m.state_dict().items()}
            m.train()
    hook.remove()
    rng_after = torch.get_rng_state()
    vce, lme = np.array(vce), np.array(lme)
    chosen = {"ft": EPOCHS, "stop_val": int(SNAPS[np.argmin(vce)]),
              "stop_logme": int(SNAPS[np.argmax(lme)]), "stop_fixed": fixed_ep}

    def finetune(ep):
        m.load_state_dict(snaps[ep])
        torch.set_rng_state(rng_after)
        opt = torch.optim.AdamW(m.parameters(), lr=FT_LR, weight_decay=WD)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, FT_EPOCHS)
        best, state = np.inf, None
        for _ in range(FT_EPOCHS):
            m.train(); opt.zero_grad()
            lw(m(xt), yt).backward(); opt.step(); sch.step()
            m.eval()
            with torch.no_grad():
                v = float(lw(m(xv), yv))
            if v < best:
                best = v
                state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            return [torch.softmax(m(_T(z)), 1).cpu().numpy() for z in Xout]

    cache = {}
    for ep in sorted(set(chosen.values()), reverse=True):   # 200 first
        cache[ep] = finetune(ep)
    return {a: cache[chosen[a]] for a in ARMS}, chosen, vce, lme


# ---------------------------------------------------------------- run
def run(reps):
    torch.set_num_threads(1)
    A, _, C = build()
    spec, desc, traj, y = A
    assert len(y) == 150, f"expected the corrected 150-patient table, got {len(y)}"
    own = {"packed": np.hstack([spec, desc, traj]), "spec": spec}
    Cp = np.hstack([C[0], C[1], C[2]])
    mk = dict(zip(MEMBERS, members(desc.shape[1])))
    key = {"two": "packed", "tcn": "spec"}
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device={DEVICE}  2015 n={len(y)} N/PD/ET="
          f"{[int((y == k).sum()) for k in (0, 1, 2)]}", flush=True)
    checked = False
    for rep in reps:
        out = f"{OUT_DIR}/rep{rep:02d}.npz"
        if os.path.exists(out):
            continue
        rng = np.random.default_rng(rep)
        kp = np.sort(np.concatenate([rng.choice(np.flatnonzero(C[3] == c),
                                                min(CAP, int((C[3] == c).sum())),
                                                replace=False) for c in (0, 1, 2)]))
        pads = {"packed": Cp[kp], "spec": C[0][kp]}
        prob = {a: np.zeros((len(y), 3)) for a in ARMS}
        pred = {a: np.full(len(y), -1) for a in ARMS}
        log = {f"{w}_{n}": [] for n in MEMBERS
               for w in ("vce", "logme", *[f"ep_{a}" for a in ARMS[1:3]])}
        for fold, (rest, te) in enumerate(
                StratifiedKFold(5, shuffle=True, random_state=rep).split(spec, y)):
            tr, va = train_test_split(rest, test_size=0.25, stratify=y[rest],
                                      random_state=rep * 10 + fold)
            tr, va = np.sort(tr), np.sort(va)
            pv_m, pt_m = {a: [] for a in ARMS}, {a: [] for a in ARMS}
            for name in MEMBERS:
                X, P = own[key[name]], pads[key[name]]
                f = zfit(X[tr])
                pre = (zfit(P)(P), C[3][kp])
                res = []
                for s in SEEDS:
                    r, ch, vce, lme = trajectory(mk[name], f(X[tr]), y[tr], f(X[va]),
                                                 y[va], [f(X[va]), f(X[te])], pre, s,
                                                 FIXED[name])
                    if not checked:
                        ref = train(mk[name], f(X[tr]), y[tr], f(X[va]), y[va],
                                    [f(X[va]), f(X[te])], seed=s, pre=pre)
                        assert all(np.array_equal(a, b) for a, b in zip(ref, r["ft"])), \
                            "ft arm does not reproduce common.protocol.train"
                        print("ft arm reproduces common.protocol.train bit-exactly",
                              flush=True)
                        checked = True
                    res.append(r)
                    log[f"vce_{name}"].append(vce)
                    log[f"logme_{name}"].append(lme)
                    log[f"ep_stop_val_{name}"].append(ch["stop_val"])
                    log[f"ep_stop_logme_{name}"].append(ch["stop_logme"])
                for a in ARMS:
                    pv_m[a].append(np.mean([r[a][0] for r in res], 0))
                    pt_m[a].append(np.mean([r[a][1] for r in res], 0))
            for a in ARMS:
                pv, pt = np.mean(pv_m[a], 0), np.mean(pt_m[a], 0)
                prob[a][te] = pt
                pred[a][te] = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            print(f"  rep {rep} fold {fold} done", flush=True)
        np.savez(out, y=y, **pred, **{f"p_{a}": v for a, v in prob.items()},
                 **{k: np.array(v) for k, v in log.items()})
        P = {a: precision_recall_fscore_support(y, pred[a], labels=[0, 1, 2],
                                                zero_division=0)[0] for a in ARMS}
        print(f"rep {rep:>2}  " + "  ".join(f"{a} ET {P[a][2]:.2f} mP {P[a].mean():.3f}"
                                           for a in ARMS), flush=True)


# ---------------------------------------------------------------- report
def _metrics(y, pred, pr):
    P, _, F, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2],
                                                 zero_division=0)
    m = y != 0
    s = pr[m, 2] / (pr[m, 1] + pr[m, 2] + 1e-12)
    o = np.argsort(-pr[:, 2])
    return [P[0], P[1], P[2], P.mean(), F.mean(),
            roc_auc_score((y[m] == 2).astype(int), s), (y[o[:5]] == 2).mean()]


def report():
    files = sorted(glob.glob(f"{OUT_DIR}/rep*.npz"))
    n = len(files)
    R = {a: [] for a in ARMS}
    etp = {a: 0 for a in ARMS}
    D = {}
    for f in files:
        d = np.load(f)
        y = d["y"]
        for a in ARMS:
            R[a].append(_metrics(y, d[a], d[f"p_{a}"]))
            etp[a] += (d[a] == 2).sum()
        for k in d.files:
            if k.startswith(("vce_", "logme_", "ep_")):
                D.setdefault(k, []).append(d[k])
    R = {a: np.array(v) for a, v in R.items()}
    D = {k: np.concatenate(v) for k, v in D.items()}
    print(f"2015 OUT, {n} repeats, {len(y)} patients (N/PD/ET = "
          f"{[int((y == c).sum()) for c in (0, 1, 2)]})\n")
    print(f"{'arm':>11}" + "".join(f"{c:>9}" for c in NAMES) + "  ET preds/rep")
    for a in ARMS:
        print(f"{a:>11}" + "".join(f"{v:>9.3f}" for v in R[a].mean(0))
              + f"  {etp[a] / n:.1f}")
    print("\nrepeat-level paired bootstrap vs ft (4000 draws), [95% CI], win rate")
    for a in ARMS[1:]:
        dd = R[a] - R["ft"]
        print(f"  {a} - ft")
        for i, nm in enumerate(NAMES):
            bs = [np.random.default_rng(s).choice(dd[:, i], n).mean() for s in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"    {nm:>8} {dd[:, i].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"{'*' if lo > 0 or hi < 0 else ' '} win {np.mean(dd[:, i] > 0):.2f}"
                  f" tie {np.mean(dd[:, i] == 0):.2f}")
    print("\nchosen pretraining epochs (per member x seed x fold x repeat)")
    for name in MEMBERS:
        for a in ("stop_val", "stop_logme"):
            e = D[f"ep_{a}_{name}"]
            q = np.percentile(e, [10, 25, 50, 75, 90])
            print(f"  {name:>4} {a:<11} n={len(e)} p10/25/50/75/90 = "
                  + "/".join(f"{v:.0f}" for v in q)
                  + f"  epoch 200 in {np.mean(e == 200):.0%}, <=50 in {np.mean(e <= 50):.0%}")
            h, _ = np.histogram(e, bins=np.arange(0, 201, 25) + 0.5)
            print(f"       counts per 25-epoch bin (1-25, ..., 176-200): {h.tolist()}")
        vc, lm = D[f"vce_{name}"].mean(0), D[f"logme_{name}"].mean(0)
        print(f"  {name:>4} mean val-CE curve: min at epoch {SNAPS[vc.argmin()]} "
              f"({vc.min():.3f}), at 200 {vc[-1]:.3f};  mean LogME curve: max at "
              f"epoch {SNAPS[lm.argmax()]} ({lm.max():.4f}), at 200 {lm[-1]:.4f}")
        print(f"       val-CE at 20/50/100/150/200: "
              + " ".join(f"{vc[SNAPS == e][0]:.3f}" for e in (20, 50, 100, 150, 200))
              + "   LogME: " + " ".join(f"{lm[SNAPS == e][0]:.4f}"
                                        for e in (20, 50, 100, 150, 200)))
    from experiments.patient_bootstrap_2015 import bootstrap_dir
    print("\npatient-level (experiments.patient_bootstrap_2015.bootstrap_dir)")
    for a in ARMS[1:]:
        bootstrap_dir(OUT_DIR, a, "ft", tag=f"pretrain_stop_150")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["report"]:
        report()
    else:
        a, b = map(int, os.environ.get("REPS", "0-20").split("-"))
        run(range(a, b))
