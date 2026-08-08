"""The chosen study design: train on 2015 + NewData, validate on PADS.

Design constraints, all forced by what the cohorts actually contain:

* **One sensor.** PADS is a single wrist unit, so everything uses the
  wrist-equivalent channel only (2015/NewData ``lower_arm``, PADS wrist).
* **Gyro-derived descriptors only.** PADS ships no orientation stream, so
  ``log_map`` and gravity-referenced features cannot be computed there. Any
  feature used for the PADS validation must survive that restriction.
* **NewData is segmented.** Its exports are ~38 s free-form captures with an
  empty Annotations table; unsegmented, only 9.9 % of power lands in the tremor
  band. ``load_2025`` now segments by default.
* **Device probe before pooling.** Judged on ``|AUC - 0.5|`` -- an AUC of 0.000
  is maximally separable with LOO-inverted labels, not "safe".

Task-matching caveat that limits the external step: only PADS **StretchHold**
has been extracted, which corresponds to ``OUT``. A model trained on ``REST``
(the stronger condition on the 2015 cohort) therefore has **no matched PADS
task** to validate against until PADS ``Relaxed`` is extracted. Both are
reported, and the mismatch is labelled rather than hidden.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfbench.descriptors import DESCRIPTOR_NAMES, describe
from tfbench.transforms import METHODS
from tremor.stats import bootstrap_subject_ci

WRIST_2015 = slice(3, 6)          # lower_arm ~ wrist


def descriptor_table(recs, method, ch=WRIST_2015):
    """(X, y, patients) of the 10 descriptors for one method."""
    fn = METHODS[method]
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        d = describe(*fn(x))
        rows[r.subject].append([d[c] for c in DESCRIPTOR_NAMES])
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[p], 0) for p in pats])),
            np.array([lab[p] for p in pats]), np.array(pats))


def _clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def bal_acc(y, p):
    return 0.5 * (recall_score(y, p, pos_label=1, zero_division=0)
                  + recall_score(y, p, pos_label=0, zero_division=0))


def device_probe(Xa, ga, Xb, gb):
    """Cohort separability. Returns (auc, deviation) -- judge on DEVIATION."""
    X = np.vstack([Xa, Xb])
    d = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    g = np.r_[ga, gb]
    p = cross_val_predict(_clf(), X, d, groups=g, cv=LeaveOneGroupOut(),
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(d, p)
    return auc, abs(auc - 0.5)


def loso(X, y, g):
    p = cross_val_predict(_clf(), X, y, groups=g, cv=LeaveOneGroupOut(),
                          method="predict_proba")[:, 1]
    return p, (p >= 0.5).astype(int)


def merge(local, new, method, ch=WRIST_2015):
    """Pool 2015 with the ET-only NewData cohort. Returns merged table + probe."""
    Xl, yl, gl = descriptor_table(local, method, ch)
    Xn, yn, gn = descriptor_table(new, method, ch)
    auc, dev = device_probe(Xl[yl == 2], gl[yl == 2], Xn, gn)
    X = np.vstack([Xl, Xn])
    y = np.r_[yl, yn]
    g = np.r_[gl, gn]
    return (X, y, g), {"identity_auc": auc, "deviation": dev,
                       "n_local": len(yl), "n_new": len(yn)}


def report(y_true, y_prob, y_pred, groups, tag, n_boot=2000):
    e = bootstrap_subject_ci(y_true, y_pred, groups, ["neg", "pos"],
                             n_boot=n_boot, seed=0)["pos"]
    maj = max(np.mean(y_true == 1), np.mean(y_true == 0))
    print(f"{tag:>46} n={len(y_true):>3} pos={int(y_true.sum()):>3} "
          f"| bal-acc {bal_acc(y_true, y_pred):.3f} | AUC {roc_auc_score(y_true, y_prob):.3f} "
          f"| F1 {f1_score(y_true, y_pred):.3f} [{e.lo:.2f},{e.hi:.2f}] | maj {maj:.3f}",
          flush=True)
    return bal_acc(y_true, y_pred)


def external_validate(train_X, train_y, test_X, test_y, test_g, tag):
    """Fit on the merged cohort, score once on the held-out external cohort."""
    m = _clf().fit(train_X, train_y)
    prob = m.predict_proba(test_X)[:, 1]
    return report(test_y, prob, (prob >= 0.5).astype(int), test_g, tag)
