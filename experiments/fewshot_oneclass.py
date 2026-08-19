"""Three techniques from the small-data literature not yet tried here.

A literature check on small imbalanced medical datasets surfaces several
standard techniques. Most are already covered in this project -- loss
reweighting, MixUp (failed), SMOTE variants (SVMSMOTE helps), and MCC /
balanced accuracy reporting. Three are not, and two of them target exactly the
constraint measured here: 21-49 ET patients.

**One-class classification.** Rather than learning a PD/ET boundary from 28 ET
examples, model the *abundant* class thoroughly -- 276 PD patients -- and flag
deviations as ET. This inverts the scarcity problem: the model learns from the
class there is plenty of, and the rare class is never fitted at all. Tested with
IsolationForest, One-Class SVM and a Mahalanobis distance to the PD centroid.

**Metric learning.** Learn a distance rather than a boundary. 28 ET patients
give ~378 within-class pairs, so the effective training signal grows
quadratically rather than linearly with the rare class. Tested as a
nearest-centroid / kNN classifier in a learned (NCA) metric space.

**Class-balanced loss.** Reweighting by the *effective number* of samples,
(1-beta^n)/(1-beta), rather than by inverse frequency. Sharper than the
`class_weight="balanced"` used everywhere else here.

Run: ``python -m experiments.fewshot_oneclass``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from experiments.pd_vs_et import build

REPEATS = 10


# --------------------------------------------------------------------------- #
# One-class: fit on PD only, score ET as deviation
# --------------------------------------------------------------------------- #
class OneClassScorer:
    """Fit on the MAJORITY class only; higher score = more ET-like.

    The rare class is never fitted, so its size stops being the constraint.
    """

    def __init__(self, kind="mahalanobis"):
        self.kind = kind

    def fit(self, X, y):
        Xp = X[y == 0]                      # PD only
        self.scaler = StandardScaler().fit(Xp)
        Z = self.scaler.transform(Xp)
        if self.kind == "mahalanobis":
            try:
                self.cov = MinCovDet(random_state=0).fit(Z)
            except Exception:
                self.cov = EmpiricalCovariance().fit(Z)
        elif self.kind == "isoforest":
            self.m = IsolationForest(n_estimators=300, random_state=0,
                                     contamination="auto").fit(Z)
        elif self.kind == "ocsvm":
            self.m = OneClassSVM(kernel="rbf", nu=0.2, gamma="scale").fit(Z)
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        if self.kind == "mahalanobis":
            return self.cov.mahalanobis(Z)
        return -self.m.score_samples(Z)      # higher = more anomalous = more ET


# --------------------------------------------------------------------------- #
# Class-balanced loss (effective number of samples)
# --------------------------------------------------------------------------- #
def cb_weights(y, beta=0.999, nc=2):
    n = np.bincount(y, minlength=nc).astype(float)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-12)
    return w / w.sum() * nc


class SmallMLP(nn.Module):
    def __init__(self, d, hidden=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def train_loss_variant(Xtr, ytr, Xte, kind="cb", seed=0, epochs=300, lr=3e-3):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    m = SmallMLP(Xtr.shape[1])
    if kind == "cb":
        w = torch.tensor(cb_weights(ytr), dtype=torch.float32)
    else:
        c = np.bincount(ytr, minlength=2).astype(float)
        w = torch.tensor(c.sum() / (2 * np.maximum(c, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        logits = m(xt)
        if kind == "focal":
            logp = torch.log_softmax(logits, 1)
            p = logp.exp()
            pt = p.gather(1, yt[:, None]).squeeze(1)
            ce = -logp.gather(1, yt[:, None]).squeeze(1)
            loss = (w[yt] * (1 - pt) ** 2.0 * ce).mean()
        else:
            loss = nn.CrossEntropyLoss(weight=w)(logits, yt)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(xe), 1).numpy()[:, 1]


def evaluate(name, scorer, X, y, k, repeats=REPEATS):
    rows = []
    for rep in range(repeats):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(k, shuffle=True,
                                      random_state=rep).split(X, y):
            s[te] = scorer(X[tr], y[tr], X[te], rep)
        # threshold at the training prevalence quantile, so all methods are
        # compared at a comparable operating point rather than an arbitrary 0.5
        thr = np.quantile(s, 1 - y.mean())
        pred = (s >= thr).astype(int)
        sens = recall_score(y, pred, pos_label=1, zero_division=0)
        spec = recall_score(y, pred, pos_label=0, zero_division=0)
        pPD = precision_score(y, pred, pos_label=0, zero_division=0)
        pET = precision_score(y, pred, pos_label=1, zero_division=0)
        rows.append([roc_auc_score(y, s), 0.5 * (sens + spec), sens, spec,
                     pPD, pET, 0.5 * (pPD + pET)])
    a = np.array(rows); mu, sd = a.mean(0), a.std(0)
    # per-class precision is the reporting standard for this project
    print(f"{name:>28}{mu[4]:>9.3f}{mu[5]:>9.3f}{mu[6]:>9.3f}"
          f"{mu[0]:>9.3f}{mu[1]:>9.3f}{mu[2]:>9.3f}{mu[3]:>9.3f}", flush=True)
    return a


def ppv(sens, spec, p):
    return sens * p / (sens * p + (1 - spec) * (1 - p) + 1e-12)


def main():
    torch.set_num_threads(1)
    data = build()
    for name, tags, keys, k in (
            ("PADS", ["PADS"], ["descriptors", "stability"], 5),
            ("in-house", ["2015", "NewData"], ["axes"], 3)):
        blocks = {q: np.vstack([data[t][0][q] for t in tags])
                  for q in data[tags[0]][0]}
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(np.hstack([blocks[q] for q in keys])[m])
        print(f"\n{'='*74}")
        print(f"{name}  PD vs ET  n={len(y)} PD={int((y==0).sum())} "
              f"ET={int(y.sum())}  features {'+'.join(keys)}")
        print(f"{'='*74}")
        print(f"{'method':>28}{'precPD':>9}{'precET':>9}{'macroP':>9}"
              f"{'AUC':>9}{'bal-acc':>9}{'ETsens':>9}{'PDspec':>9}")
        res = {}

        def lr_base(Xtr, ytr, Xte, rep):
            p = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(max_iter=5000,
                                                   class_weight="balanced"))])
            return p.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        res["logreg (baseline)"] = evaluate("logreg (baseline)", lr_base, X, y, k)

        for kind in ("mahalanobis", "isoforest", "ocsvm"):
            def oc(Xtr, ytr, Xte, rep, kind=kind):
                return OneClassScorer(kind).fit(Xtr, ytr).score(Xte)
            res[f"one-class {kind}"] = evaluate(f"one-class {kind}", oc, X, y, k)

        def metric_knn(Xtr, ytr, Xte, rep):
            n_comp = min(Xtr.shape[1], 4)
            p = Pipeline([("s", StandardScaler()),
                          ("nca", NeighborhoodComponentsAnalysis(
                              n_components=n_comp, random_state=rep, max_iter=200)),
                          ("k", KNeighborsClassifier(
                              n_neighbors=min(5, int(ytr.sum())),
                              weights="distance"))])
            return p.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        res["metric learning (NCA+kNN)"] = evaluate("metric learning (NCA+kNN)",
                                                    metric_knn, X, y, k)

        for kind, lab in (("bal", "MLP inverse-freq"), ("cb", "MLP class-balanced"),
                          ("focal", "MLP focal")):
            def f(Xtr, ytr, Xte, rep, kind=kind):
                return np.mean([train_loss_variant(Xtr, ytr, Xte, kind, seed=s)
                                for s in (0, 1, 2)], 0)
            res[lab] = evaluate(lab, f, X, y, k)

        print(f"\n  ET PPV at clinical prevalences (from sens/spec):")
        print(f"{'method':>28}{'p=0.30':>10}{'p=0.50':>10}{'p=0.60':>10}")
        for lab, a in res.items():
            se, sp = a[:, 2].mean(), a[:, 3].mean()
            print(f"{lab:>28}" + "".join(f"{ppv(se, sp, p):>10.3f}"
                                         for p in (0.30, 0.50, 0.60)))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
