"""Actually train (and fine-tune) networks, scored the same way as the
classical two-stage models.

``two_stage_comparison.ipynb`` previously compared only scikit-learn models on
precomputed features -- nothing in it trained a network. This module supplies
the missing piece: BiLSTMs trained on TF images under patient-level grouped CV,
with per-recording probabilities aggregated to patient level so the numbers sit
in the SAME table as the classical results (same patients, same bootstrap CI,
same permutation test).

Two modes:
  ``3class``    one 3-class network.
  ``two_stage`` a N-vs-tremor network plus a DEDICATED PD-vs-ET network, with
                the ET threshold tuned on the fold's validation subjects only.

Fine-tuning: pass ``pretrain=`` a state dict (or use :func:`pretrain_stage`) to
warm-start each fold instead of training from scratch -- e.g. pretrain the
PD-vs-ET stage on PADS, then fine-tune per fold on the local cohort.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score

from tremor.data import CLASS_NAMES
from tremor.evaluate import classification_report, softmax
from tremor.stats import bootstrap_subject_ci, permutation_test
from pdetn.deep_crossdataset import DEVICE, predict_logits, remap, train_bilstm


def _patient_split(recs, seed=0, val_frac=0.25):
    """Split recordings into train/val by PATIENT, keeping class balance.

    Never splits within a patient -- that would leak.
    """
    subs, ys = [], []
    seen = {}
    for r in recs:
        if r.subject not in seen:
            seen[r.subject] = r.y
    subs = np.array(sorted(seen))
    ys = np.array([seen[s] for s in subs])
    n_splits = max(2, int(round(1 / val_frac)))
    n_splits = min(n_splits, min(np.bincount(ys)[np.bincount(ys) > 0]))
    if n_splits < 2:
        return recs, recs                       # too few subjects to hold out
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_i, vl_i = next(skf.split(subs, ys))
    tr_s, vl_s = set(subs[tr_i]), set(subs[vl_i])
    return ([r for r in recs if r.subject in tr_s],
            [r for r in recs if r.subject in vl_s])


def _tune_et_threshold(p_et, y_et, grid=np.linspace(0.1, 0.9, 33)):
    best_t, best = 0.5, -1.0
    for t in grid:
        f = f1_score(y_et, (p_et >= t).astype(int), zero_division=0)
        if f > best:
            best, best_t = f, float(t)
    return best_t


def _fold_probs(train_recs, test_recs, mode, target_length, seed, tune_et,
                train_kw, pretrain):
    """Fit on a fold and return (P, threshold) with P = (n_test, 3) 3-class
    probabilities."""
    tr, vl = _patient_split(train_recs, seed=seed)
    if mode == "3class":
        m = train_bilstm(tr, vl, 3, target_length, seed=seed,
                         init_state=pretrain.get("3class") if pretrain else None,
                         **train_kw)
        return softmax(predict_logits(m, test_recs, target_length)), 0.5

    s1 = train_bilstm(remap(tr, {0: 0, 1: 1, 2: 1}), remap(vl, {0: 0, 1: 1, 2: 1}),
                      2, target_length, seed=seed,
                      init_state=pretrain.get("s1") if pretrain else None, **train_kw)
    s2 = train_bilstm(remap(tr, {1: 0, 2: 1}), remap(vl, {1: 0, 2: 1}),
                      2, target_length, seed=seed,
                      init_state=pretrain.get("s2") if pretrain else None, **train_kw)

    thr = 0.5
    if tune_et:
        s2_vl = remap(vl, {1: 0, 2: 1})
        if s2_vl and len({r.y for r in s2_vl}) == 2:
            p = softmax(predict_logits(s2, s2_vl, target_length))[:, 1]
            thr = _tune_et_threshold(p, np.array([r.y for r in s2_vl]))

    p_tremor = softmax(predict_logits(s1, test_recs, target_length))[:, 1]
    p_et = softmax(predict_logits(s2, test_recs, target_length))[:, 1]
    # rescale so the tuned threshold sits at 0.5 in the composed probability,
    # keeping argmax over the 3 columns equivalent to the staged decision rule
    p_et_adj = np.clip(0.5 * p_et / max(thr, 1e-6), 0.0, 1.0)
    P = np.stack([1 - p_tremor, p_tremor * (1 - p_et_adj), p_tremor * p_et_adj], 1)
    return P, thr


def deep_grouped_cv(recs, mode="two_stage", n_splits=5, target_length=None,
                    seed=0, tune_et=True, pretrain=None, verbose=True,
                    n_boot=1000, n_perm=1000, **train_kw):
    """Patient-level grouped CV over a deep model; patient-level scoring.

    Args:
        recs: 3-class ``Recording`` list (y in {0, 1, 2}).
        mode: ``'two_stage'`` or ``'3class'``.
        n_splits: patient-grouped folds. Every patient is tested exactly once.
        target_length: samples per recording fed to the TFD (default: the 25th
            percentile length, so truncation keeps most recordings intact).
        pretrain: dict of warm-start state dicts, keys ``'s1'``/``'s2'``/
            ``'3class'`` -- makes this a FINE-TUNING run.
        **train_kw: forwarded to :func:`train_bilstm` (``epochs``, ``lr``,
            ``focal_gamma``, ``hidden``, ``dropout``, ``tfd_method``,
            ``nperseg``, ...).

    Returns the same dict shape as :func:`pdetn.evaluate.evaluate`, so
    ``print_result`` and the comparison table work unchanged.
    """
    y_rec = np.array([r.y for r in recs])
    groups = np.array([r.subject for r in recs])
    if target_length is None:
        target_length = int(np.percentile([r.x.shape[1] for r in recs], 25))

    P = np.zeros((len(recs), 3))
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr_i, te_i) in enumerate(gkf.split(np.zeros(len(recs)), y_rec, groups)):
        tr = [recs[i] for i in tr_i]
        te = [recs[i] for i in te_i]
        P[te_i], thr = _fold_probs(tr, te, mode, target_length, seed + k,
                                   tune_et, train_kw, pretrain)
        if verbose:
            print(f"  fold {k + 1}/{n_splits}: {len(tr)} train / {len(te)} test "
                  f"recordings, {len(set(groups[te_i]))} test patients"
                  + (f", ET thr {thr:.2f}" if mode == "two_stage" else ""))

    # aggregate recordings -> patients (mean probability), then decide once
    pats = np.array(sorted(set(groups)))
    Pp = np.array([P[groups == p].mean(0) for p in pats])
    yp = np.array([y_rec[groups == p][0] for p in pats])
    y_pred = Pp.argmax(1)

    rep = classification_report(np.log(Pp + 1e-9), yp, CLASS_NAMES)
    ci = bootstrap_subject_ci(yp, y_pred, pats, CLASS_NAMES, n_boot=n_boot, seed=seed)
    perm = permutation_test(yp, y_pred, pats, CLASS_NAMES, n_perm=n_perm, seed=seed)
    m = (yp != 0) & (y_pred != 0)
    return {
        "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
        "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
        "confusion_matrix": rep["confusion_matrix"],
        "ci": {k: {"point": e.point, "lo": e.lo, "hi": e.hi} for k, e in ci.items()},
        "permutation_p": perm["p_value"],
        "n_vs_tremor_acc": float(((y_pred != 0) == (yp != 0)).mean()),
        "pd_vs_et_acc": (float((y_pred[m] == yp[m]).mean()) if m.any() else float("nan")),
        "y_true": yp.tolist(), "y_pred": y_pred.tolist(),
        "patients": pats.tolist(), "proba": Pp.tolist(),
    }


def pretrain_stage(recs, stage="s2", target_length=None, seed=0, **train_kw):
    """Train one stage on an EXTERNAL cohort and return its state dict, for use
    as ``pretrain=`` in :func:`deep_grouped_cv`.

    ``stage='s1'`` -> N-vs-tremor, ``'s2'`` -> PD-vs-ET, ``'3class'`` -> 3-class.
    The external cohort is never used for scoring, only for initialisation.
    """
    if target_length is None:
        target_length = int(np.percentile([r.x.shape[1] for r in recs], 25))
    mapping = {"s1": {0: 0, 1: 1, 2: 1}, "s2": {1: 0, 2: 1},
               "3class": {0: 0, 1: 1, 2: 2}}[stage]
    sub = remap(recs, mapping)
    n_cls = 3 if stage == "3class" else 2
    tr, vl = _patient_split(sub, seed=seed)
    m = train_bilstm(tr, vl, n_cls, target_length, seed=seed, **train_kw)
    return {stage: {k: v.cpu() for k, v in m.state_dict().items()}}


def multi_seed(recs, seeds=(0, 1, 2, 3), **kw):
    """Repeat :func:`deep_grouped_cv` over seeds -- a single deep run is a draw
    from a distribution, and we have been burned by reporting a lucky one."""
    out = []
    for s in seeds:
        r = deep_grouped_cv(recs, seed=s, verbose=False, **kw)
        out.append(r)
        print(f"  seed {s}: macro-F1 {r['macro_f1']:.3f}  "
              f"ET-F1 {r['per_class_f1']['ET']:.3f}  "
              f"N-vs-T {r['n_vs_tremor_acc']:.3f}  PD-vs-ET {r['pd_vs_et_acc']:.3f}")
    for key, get in [("macro_f1", lambda r: r["macro_f1"]),
                     ("ET_f1", lambda r: r["per_class_f1"]["ET"]),
                     ("n_vs_tremor", lambda r: r["n_vs_tremor_acc"]),
                     ("pd_vs_et", lambda r: r["pd_vs_et_acc"])]:
        v = np.array([get(r) for r in out])
        print(f"{key:>12}: {v.mean():.3f} +/- {v.std():.3f}  (n={len(v)} seeds)")
    return out
