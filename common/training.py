"""Training loops for the spectrum models.

Full-batch by default: at ~250 training patients a whole cohort fits in one
batch, and minibatch training was measured WORSE for the residual TCN at
every batch size (see reports/deep_model_improvement.md).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_predict(model_fn, Xtr, ytr, Xte, epochs=300, lr=3e-3, wd=1e-3,
                seed=0, class_weight=True, device="cpu"):
    """Train on a fold and return test probabilities. Small enough for full-batch."""
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    m = model_fn().to(device)
    w = None
    if class_weight:
        cnt = np.bincount(ytr, minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)),
                         dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(m(Xtr_t), ytr_t).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(Xte_t), 1).cpu().numpy()[:, 1]
def fit_predict_proba(model_fn, Xtr, ytr, Xte, num_classes=3, epochs=300,
                      lr=3e-3, wd=1e-3, seed=0, class_weight=True, device="cpu"):
    """Multiclass counterpart of :func:`fit_predict`; returns full probabilities.

    ``fit_predict`` slices ``[:, 1]`` and so only works on a binary axis. The
    three-cohort N/PD/ET problem needs the whole simplex to compute per-class
    precision, which is the metric that actually matters here -- macro F1 hides
    an ET column that is often 0.1-0.2.
    """
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    m = model_fn().to(device)
    w = None
    if class_weight:
        cnt = np.bincount(ytr, minlength=num_classes).astype(float)
        w = torch.tensor(cnt.sum() / (num_classes * np.maximum(cnt, 1)),
                         dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(m(Xtr_t), ytr_t).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(Xte_t), 1).cpu().numpy()
def kfold_proba(X, y, groups, model_fn, num_classes=3, k=5, seeds=(0, 1, 2),
                epochs=200, class_weight=True, **kw):
    """Patient-disjoint stratified k-fold; probabilities averaged over seeds.

    The scaler is fit on each fold's TRAIN split only. k-fold rather than LOSO
    because the deep models are refit per seed per fold and LOSO at ~250
    patients is 750 fits per architecture.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    prob = np.zeros((len(y), num_classes))
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, groups=groups):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        prob[te] = np.mean([fit_predict_proba(
            model_fn, (X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd,
            num_classes=num_classes, seed=s, epochs=epochs,
            class_weight=class_weight, **kw) for s in seeds], 0)
    return prob
def loso_nn(X, y, groups, model_fn, seeds=(0, 1, 2), epochs=150,
            class_weight=False, **kw):
    """Patient-level LOSO with a small net; probability averaged over seeds.

    The scaler is fit on the TRAIN split of each fold only. ``class_weight``
    defaults False: on the frequency BiLSTM it costs precision (0.667 -> 0.600)
    and AUC (0.942 -> 0.870) while buying recall -- sweep both and report both,
    do not assume.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    prob = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        prob[te] = np.mean([fit_predict(model_fn, (X[tr] - mu) / sd, y[tr],
                                        (X[te] - mu) / sd, seed=s, epochs=epochs,
                                        class_weight=class_weight, **kw)
                            for s in seeds], 0)
    return prob
