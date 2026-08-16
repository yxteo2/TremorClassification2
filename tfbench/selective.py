"""Selective prediction: precision at reduced coverage, with an abstain option.

Every precision figure in this repo is measured at **100 % coverage** -- the
model is forced to label every patient. Precision is threshold-dependent, so
that is the single hardest operating point.

A target of >0.90 macro precision on all three classes at full coverage is not
reachable with 404 patients and 49 ET; the best measured is 0.675, and the
published state of the art on PADS (a transformer with self-supervised
pretraining, 469 patients) reports 87 % on the *two-class* PD-vs-DD problem.

What is reachable, and is a normal clinical framing, is high precision with an
**abstain** option: the model answers confidently for some fraction of patients
and refers the rest for specialist review. This module measures the trade
directly.

Two abstention rules:

``max_prob``  abstain when the top class probability is below a threshold.
``margin``   abstain when the gap between the top two class probabilities is
             below a threshold. Better suited here, because the failure mode is
             confusion between two specific classes (PD vs ET) rather than
             uniform uncertainty.

Reported as precision-at-coverage: for each target coverage, the macro and
per-class precision over the patients the model chose to answer for.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def selective_scores(prob, y, coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
                     rule="margin", n_classes=3):
    """Per-class and macro precision at each target coverage.

    ``prob`` is (n, n_classes) predicted probability, ``y`` the true labels.
    Patients are ranked by confidence and the least-confident are abstained on
    until the target coverage is met.
    """
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y)
    pred = prob.argmax(1)
    if rule == "max_prob":
        conf = prob.max(1)
    elif rule == "margin":
        s = np.sort(prob, axis=1)
        conf = s[:, -1] - s[:, -2]
    else:
        raise ValueError(rule)

    order = np.argsort(-conf)          # most confident first
    out = []
    for cov in coverages:
        k = max(int(round(cov * len(y))), n_classes)
        idx = order[:k]
        P, R, F, S = precision_recall_fscore_support(
            y[idx], pred[idx], labels=list(range(n_classes)), zero_division=0)
        # macro precision over classes actually PRESENT among the answered
        present = S > 0
        out.append({
            "coverage": len(idx) / len(y),
            "n": len(idx),
            "precN": P[0], "precPD": P[1], "precET": P[2],
            "macroP": float(P[present].mean()) if present.any() else float("nan"),
            "macroF1": float(F[present].mean()) if present.any() else float("nan"),
            "n_per_class": S.tolist(),
        })
    return out


def coverage_at_precision(prob, y, target=0.90, rule="margin", n_classes=3,
                          grid=None):
    """Highest coverage whose macro precision still reaches ``target``.

    Returns (coverage, macroP, n_answered) or (0.0, nan, 0) if the target is
    never met at any coverage down to 20 %.
    """
    grid = grid if grid is not None else np.linspace(1.0, 0.2, 33)
    best = (0.0, float("nan"), 0)
    for row in selective_scores(prob, y, coverages=grid, rule=rule,
                                n_classes=n_classes):
        if row["macroP"] >= target and row["coverage"] > best[0]:
            best = (row["coverage"], row["macroP"], row["n"])
    return best


def report(prob, y, tag="", rule="margin", target=0.90, class_names=None):
    """``class_names`` labels the per-class columns.

    Defaults to N/PD/ET. Pass the actual names for binary axes -- the columns
    are positional, so a binary PD-vs-ET call previously printed PD precision
    under a 'precN' header.
    """
    n_classes = prob.shape[1]
    names = class_names or (("N", "PD", "ET")[:n_classes] if n_classes <= 3
                            else tuple(f"c{i}" for i in range(n_classes)))
    rows = selective_scores(prob, y, rule=rule, n_classes=n_classes)
    print(f"\n{tag}  (abstention rule: {rule})")
    print(f"{'coverage':>9}{'n':>6}" + "".join(f"{'prec'+c:>9}" for c in names)
          + f"{'macroP':>9}{'macroF1':>9}   n per class")
    keys = ["precN", "precPD", "precET"][:n_classes]
    for r in rows:
        print(f"{r['coverage']:>9.2f}{r['n']:>6}"
              + "".join(f"{r[k]:>9.3f}" for k in keys)
              + f"{r['macroP']:>9.3f}{r['macroF1']:>9.3f}   {r['n_per_class']}")
    cov, mp, n = coverage_at_precision(prob, y, target=target, rule=rule,
                                       n_classes=n_classes)
    if cov > 0:
        print(f"  -> macro precision >= {target:.2f} at coverage {cov:.0%} "
              f"({n} of {len(y)} patients answered, macroP {mp:.3f})")
    else:
        print(f"  -> macro precision never reaches {target:.2f}, "
              f"even abstaining on 80 % of patients")
    return rows
