"""Classifiers for N/PD/ET separation: flat and hierarchical two-stage.

The two-stage model encodes the structure we measured: stage 1 separates
Normal from tremor (the easy axis), stage 2 separates PD from ET among the
tremor cases (the hard axis). Each stage is an independent estimator, so the
hard PD-vs-ET decision is never diluted by the easy N decision.

Estimators are scikit-learn pipelines (median-impute -> scale -> classifier),
so NaN condition columns are handled and everything runs on CPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tremor.data import CLASS_NAMES

N, PD, ET = 0, 1, 2


def make_estimator(name: str = "rf", k_best: int | None = None):
    """A CPU pipeline that tolerates NaN (absent conditions).

    ``k_best`` (if set) selects the top-k univariately-discriminative features
    on the training fold before the classifier — essential when many processed
    features are available but the cohort is small (avoids overfitting).
    """
    clf = {
        "rf": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=0),
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000),
        "lda": LinearDiscriminantAnalysis(),
    }[name]
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if k_best is not None:
        from sklearn.feature_selection import SelectKBest, f_classif
        steps.append(("select", SelectKBest(f_classif, k=k_best)))
    steps.append(("clf", clf))
    return Pipeline(steps)


class TwoStageClassifier:
    """N-vs-tremor, then PD-vs-ET among predicted tremor.

    Stage 2 uses a *tuned* ET probability threshold, chosen by internal
    cross-validation on the training fold to maximize PD-vs-ET ET-F1. This is
    the leakage-free counterpart of the deep model's in-CV threshold selection
    and is what stops ET from collapsing under the PD majority.
    """

    def __init__(self, stage1: str = "rf", stage2: str = "rf",
                 tune_et_threshold: bool = True, cv: int = 4,
                 k_best: int | None = None):
        self.s1 = make_estimator(stage1, k_best=k_best)
        self.s2 = make_estimator(stage2, k_best=k_best)
        self.tune = tune_et_threshold
        self.cv = cv
        self.et_threshold_ = 0.5

    def fit(self, X, y):
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        y = np.asarray(y)
        self.s1.fit(X, (y != N).astype(int))       # 0 = N, 1 = tremor
        tremor = y != N
        self._has_s2 = tremor.sum() >= 2 and len(np.unique(y[tremor])) == 2
        if self._has_s2:
            Xt, yt = X[tremor], (y[tremor] == ET).astype(int)
            self.et_threshold_ = 0.5
            n_et = int(yt.sum())
            if self.tune and n_et >= 3 and (len(yt) - n_et) >= 3:
                k = int(min(self.cv, n_et, len(yt) - n_et))
                if k >= 2:
                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
                    proba = cross_val_predict(self.s2, Xt, yt, cv=skf,
                                              method="predict_proba")[:, 1]
                    best_t, best_f1 = 0.5, -1.0
                    for t in np.linspace(0.1, 0.9, 33):
                        f1 = f1_score(yt, (proba >= t).astype(int), zero_division=0)
                        if f1 > best_f1:
                            best_f1, best_t = f1, float(t)
                    self.et_threshold_ = best_t
            self.s2.fit(Xt, yt)
        return self

    def predict(self, X):
        is_tremor = self.s1.predict(X).astype(bool)
        out = np.full(len(X), N, dtype=int)
        idx = np.flatnonzero(is_tremor)
        if len(idx) and self._has_s2:
            p_et = self.s2.predict_proba(X[idx])[:, 1]
            out[idx] = np.where(p_et >= self.et_threshold_, ET, PD)
        elif len(idx):
            out[is_tremor] = PD                        # fallback if no stage-2
        return out


class FlatClassifier:
    """Single 3-class estimator, for comparison against the two-stage model."""

    def __init__(self, name: str = "rf"):
        self.est = make_estimator(name)

    def fit(self, X, y):
        self.est.fit(X, np.asarray(y))
        return self

    def predict(self, X):
        return self.est.predict(X)
