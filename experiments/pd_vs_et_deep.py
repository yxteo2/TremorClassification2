"""Deep models on the binary PD-vs-ET problem.

`pd_vs_et.py` established the linear baselines and the per-cohort best feature
sets, training on tremor patients only:

  PADS      (28 ET)   spectrum          AUC 0.790   precET 0.268
  in-house  (21 ET)   axes              AUC 0.625   precET 0.291
  merged    (49 ET)   axes + stability  AUC 0.728   precET 0.218

Two things there are worth carrying in. The binary framing already beats reading
PD-vs-ET out of the 3-class model (in-house ET precision 0.291 against 0.193),
and `axes + stability` on the merged cohort is the **first feature union in this
project to beat both its members** (0.728 against 0.627 and 0.659) -- spatial
shape and temporal steadiness are independent properties, so they add.

This asks whether a deep model beats those linear baselines on the same folds.
Every model is binary (2 outputs), trained on tremor patients only:

``logreg``       the winning linear baseline per cohort
``MLPHead``      non-linear boundary on the same hand features
``Spectrum1DCNN`` learned features from the spectrum
``TwoStream``    spectrum CNN + IF-trajectory TCN, the merged model's architecture
``TwoStream+hf`` the same, with the winning hand features joined at the head

Run: ``python -m experiments.pd_vs_et_deep``
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.cohorts import asym_for, desc_table, logbin
from common.protocol import NBIN, N_ASYM
from experiments.final_model import method_table
from frequency.tables import spectrum_table
from models.architectures import (MLPHead, ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)
from signal_processing.stability import patient_table as stab_table
from signal_processing.stability import trajectory_table
from signal_processing.tremor_physics import FAMILIES, FEATURE_NAMES
from signal_processing.tremor_physics import patient_table as physics_table

REPEATS, TL = 10, 64
AX = [FEATURE_NAMES.index(n) for n in FAMILIES["axes"]]


def build():
    from common.load_2025 import SIDE, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    side_new = lambda r: SIDE.get(os.path.basename(r.path)[:2])
    side_pads = lambda r: ("left" if "LeftWrist" in str(r.path)
                           else ("right" if "RightWrist" in str(r.path) else None))
    out = {}
    for tag, recs, ch, side in (
            ("2015", load_quaternion_recordings("Data", action="OUT",
                                                mode="angular_velocity"),
             slice(3, 6), None),
            ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6), side_new),
            ("PADS", load_pads_extracted("pads_stretchhold"), slice(0, 3),
             side_pads)):
        sp = spectrum_table(recs, ch=ch)
        ph = physics_table(recs, ch=ch)[0]
        tr = trajectory_table(recs, ch=ch, n_out=TL)[0]
        if side is not None:
            a, h = asym_for(recs, side, ch, sp[2])
            asym = np.hstack([a, h[:, None]])
        else:
            asym = np.zeros((len(sp[1]), N_ASYM + 1))
        out[tag] = dict(
            spectrum=logbin(method_table(recs, "multitaper", ch)[0]),
            descriptors=desc_table(recs, ch),
            stability=stab_table(recs, ch=ch)[0],
            axes=ph[:, AX],
            asymmetry=asym,
            traj=tr.reshape(len(tr), -1),
            y=sp[1],
        )
    return out


def train_binary(model_fn, Xtr, ytr, Xva, yva, Xte, seed=0, epochs=200,
                 lr=3e-3, wd=1e-3):
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.long)
    xv, yv = T(Xva), torch.tensor(yva, dtype=torch.long)
    m = model_fn()
    c = np.bincount(ytr, minlength=2).astype(float)
    w = torch.tensor(c.sum() / (2 * np.maximum(c, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        lf(m(xt), yt).backward(); opt.step(); sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lf(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(T(Xte)), 1).numpy()[:, 1]


def run(name, X, y, mk, k=5, repeats=REPEATS, deep=True):
    """Repeated stratified CV; inner split carves validation from train."""
    rows = []
    for rep in range(repeats):
        prob = np.zeros(len(y))
        cv = StratifiedKFold(k, shuffle=True, random_state=rep)
        for tr_all, te in cv.split(X, y):
            if deep:
                inner = StratifiedKFold(4, shuffle=True, random_state=rep)
                tr, va = next(inner.split(X[tr_all], y[tr_all]))
                tr, va = tr_all[tr], tr_all[va]
                mu = X[tr].mean(0, keepdims=True)
                sd = X[tr].std(0, keepdims=True) + 1e-8
                prob[te] = np.mean([train_binary(mk, (X[tr]-mu)/sd, y[tr],
                                                 (X[va]-mu)/sd, y[va],
                                                 (X[te]-mu)/sd, seed=s)
                                    for s in (0, 1, 2)], 0)
            else:
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000,
                                                     class_weight="balanced"))
                prob[te] = m.fit(X[tr_all], y[tr_all]).predict_proba(X[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        P, R, _, _ = precision_recall_fscore_support(y, pred, labels=[0, 1],
                                                     zero_division=0)
        rows.append([roc_auc_score(y, prob), P[0], P[1], 0.5 * (R[0] + R[1])])
    a = np.array(rows); mu, sd = a.mean(0), a.std(0)
    print(f"{name:>26}{X.shape[1]:>5}"
          + "".join(f"{mu[i]:>10.3f} +/-{sd[i]:<4.3f}" for i in range(4)),
          flush=True)
    return a


def paired(a, b, name):
    d = a - b
    print(f"  {name}:")
    for i, nm in enumerate(("AUC", "precPD", "precET", "bal-acc")):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def main():
    torch.set_num_threads(1)
    data = build()
    settings = [
        ("in-house (2015+NewData)", ["2015", "NewData"], ["axes"], 3),
        ("MERGED (all three)", ["2015", "NewData", "PADS"],
         ["axes", "stability"], 5),
        ("PADS", ["PADS"], ["spectrum"], 5),
    ]
    for name, tags, best_keys, k in settings:
        B = {q: np.vstack([data[t][q] for t in tags])
             for q in ("spectrum", "descriptors", "stability", "axes",
                       "asymmetry", "traj")}
        y3 = np.concatenate([data[t]["y"] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        hf = np.nan_to_num(np.hstack([B[q] for q in best_keys])[m])
        sp = B["spectrum"][m]
        tj = B["traj"][m]
        nd = hf.shape[1]
        print(f"\n{'='*84}")
        print(f"{name}   PD vs ET   n={int(m.sum())} PD={int((y==0).sum())} "
              f"ET={int(y.sum())}   best linear features: {'+'.join(best_keys)}")
        print(f"{'='*84}")
        print(f"{'model':>26}{'dim':>5}{'AUC':>16}{'precPD':>16}{'precET':>16}"
              f"{'bal-acc':>16}")
        res = {}
        res["logreg (baseline)"] = run("logreg (baseline)", hf, y, None, k=k,
                                       deep=False)
        res["MLPHead"] = run("MLPHead h=16", hf, y,
                             lambda: MLPHead(nd, num_classes=2, hidden=16), k=k)
        res["CNN spectrum"] = run("Spectrum1DCNN", sp, y,
                                  lambda: Spectrum1DCNN(NBIN, num_classes=2,
                                                        ch=8), k=k)
        res["ResidualTCN"] = run("ResidualTCN", sp, y,
                                 lambda: ResidualTCN(NBIN, num_classes=2,
                                                     ch=16), k=k)
        packed = np.hstack([sp, hf, tj])
        res["TwoStream+hf"] = run("TwoStream + hand feats", packed, y,
                                  lambda: TwoStreamNet(
                                      Spectrum1DCNN(NBIN, 2, ch=8),
                                      TRUNKS["cnn"], 8 * 2 * 4, NBIN, nd, TL,
                                      num_classes=2), k=k)
        print(f"\npaired vs logreg baseline, {REPEATS} repeats:")
        for q in ("MLPHead", "CNN spectrum", "ResidualTCN", "TwoStream+hf"):
            paired(res[q], res["logreg (baseline)"], q)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
