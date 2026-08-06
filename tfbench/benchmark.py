"""Stage 1 — rank signal-processing methods by how well simple frequency
descriptors discriminate tremor.

Two levels of evidence, both required before calling a method "best":

* **Univariate** — every (method, descriptor) pair scored by AUC and
  Mann-Whitney effect, with **Benjamini-Hochberg q-values** over the whole grid.
  A raw p from a grid this size means nothing on its own; this repo has already
  been burned by exactly that (`reports/handedness_does_not_survive.md`).
* **Multivariate** — each method's full descriptor set through patient-level
  LOSO, compared against the reference method with a **paired subject-level
  bootstrap** of the difference. That paired CI, not the point estimate, decides
  whether one method beats another.

Both axes are reported separately: N-vs-Tremor and PD-vs-ET. They behave very
differently and a single "accuracy" hides that.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tfbench.descriptors import DESCRIPTOR_NAMES, describe
from tfbench.transforms import METHODS

AXES = {"N_vs_Tremor": lambda y: (y != 0).astype(int),
        "PD_vs_ET": None}          # PD_vs_ET drops N; handled in _axis_data


def _axis_data(X, y, g, axis):
    if axis == "N_vs_Tremor":
        return X, (y != 0).astype(int), g
    m = y != 0
    return X[m], (y[m] == 2).astype(int), g[m]


def patient_table(recs, method, fs=100.0, **kw):
    """(X, y, patients) of descriptors for one method, averaged per patient."""
    fn = METHODS[method]
    per, lab = defaultdict(list), {}
    for r in recs:
        f, P = fn(r.x, fs=fs, **kw)
        d = describe(f, P)
        per[r.subject].append([d[c] for c in DESCRIPTOR_NAMES])
        lab[r.subject] = r.y
    pats = sorted(per)
    X = np.nan_to_num(np.array([np.mean(per[p], axis=0) for p in pats]))
    return X, np.array([lab[p] for p in pats]), np.array(pats)


def build_all(recs, methods=None, fs=100.0, verbose=True, **kw):
    """Compute the patient table for every method once, and reuse it."""
    tables = {}
    for m in (methods or list(METHODS)):
        try:
            tables[m] = patient_table(recs, m, fs=fs, **kw)
            if verbose:
                print(f"  {m:>15}  {tables[m][0].shape}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  {m:>15}  FAILED {type(e).__name__}: {e}", flush=True)
    return tables


# --------------------------------------------------------------------------- #
def screen(tables, axis="PD_vs_ET", top=20):
    """Univariate (method, descriptor) screen with BH q-values."""
    rows = []
    for m, (X, y, g) in tables.items():
        Xa, ya, _ = _axis_data(X, y, g, axis)
        for j, d in enumerate(DESCRIPTOR_NAMES):
            a, b = Xa[ya == 0, j], Xa[ya == 1, j]
            if len(a) < 3 or len(b) < 3 or (a.std() + b.std()) == 0:
                continue
            u, p = mannwhitneyu(a, b)
            rows.append([m, d, roc_auc_score(ya, Xa[:, j]),
                         2 * u / (len(a) * len(b)) - 1, p])
    rows.sort(key=lambda r: r[4])
    n = len(rows)
    for i, r in enumerate(rows):                    # Benjamini-Hochberg
        r.append(min(1.0, r[4] * n / (i + 1)))
    print(f"\n=== univariate screen: {axis} ===  ({n} tests, BH-corrected)")
    print(f"{'method':>15}{'descriptor':>18}{'AUC':>7}{'effect':>9}{'raw p':>10}{'BH q':>9}")
    for m, d, auc, eff, p, q in rows[:top]:
        flag = " *" if q < 0.05 else ""
        print(f"{m:>15}{d:>18}{auc:>7.3f}{eff:>9.3f}{p:>10.4f}{q:>9.4f}{flag}")
    sig = sum(1 for r in rows if r[5] < 0.05)
    print(f"  surviving BH q<0.05: {sig} of {n}")
    return rows


def _loso(X, y, g):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    return cross_val_predict(clf, X, y, groups=g, cv=LeaveOneGroupOut())


def _bal(y, p):
    return 0.5 * (recall_score(y, p, pos_label=1, zero_division=0)
                  + recall_score(y, p, pos_label=0, zero_division=0))


def rank_methods(tables, axis="PD_vs_ET", reference="welch", n_boot=3000, seed=0):
    """Per-method LOSO, with a paired bootstrap CI against ``reference``."""
    rng = np.random.default_rng(seed)
    preds, ys = {}, {}
    for m, (X, y, g) in tables.items():
        Xa, ya, ga = _axis_data(X, y, g, axis)
        preds[m], ys[m] = _loso(Xa, ya, ga), ya

    ref = preds.get(reference)
    print(f"\n=== method ranking: {axis} ===  (paired vs '{reference}')")
    print(f"{'method':>15}{'bal-acc':>9}{'diff':>8}{'95% CI of diff':>21}{'p(<=0)':>9}")
    out = []
    for m in sorted(preds, key=lambda k: -_bal(ys[k], preds[k])):
        y, p = ys[m], preds[m]
        bal = _bal(y, p)
        if ref is None or m == reference:
            print(f"{m:>15}{bal:>9.3f}{'—':>8}{'—':>21}{'—':>9}")
            out.append((m, bal, np.nan, np.nan, np.nan)); continue
        # Bootstrap the BALANCED accuracy difference. Using raw accuracy here
        # would be the classic trap on this axis: with 75 PD / 15 ET a model
        # that simply predicts PD more often scores a large positive raw-accuracy
        # difference while getting worse at the minority class.
        idx = np.arange(len(y))
        d = []
        for _ in range(n_boot):
            bi = rng.choice(idx, len(idx), True)
            yb = y[bi]
            if len(np.unique(yb)) < 2:
                continue
            d.append(_bal(yb, p[bi]) - _bal(yb, ref[bi]))
        d = np.array(d)
        if d.size == 0:
            print(f"{m:>15}{bal:>9.3f}{'n/a':>8}{'n/a':>21}{'n/a':>9}")
            out.append((m, bal, np.nan, np.nan, np.nan)); continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        star = " *" if lo > 0 else ""
        print(f"{m:>15}{bal:>9.3f}{d.mean():>+8.3f}   [{lo:>+.3f}, {hi:>+.3f}]"
              f"{(d <= 0).mean():>9.4f}{star}")
        out.append((m, bal, d.mean(), lo, hi))
    return out
