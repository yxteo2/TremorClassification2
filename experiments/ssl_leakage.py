"""Is the self-supervised gain real, or did the encoder see the test patients?

``experiments/masked_pretrain.py`` measured the largest deep-learning gain in
this project: a masked-spectrum encoder pretrained on 3081 unlabelled
recordings, then **frozen** with a linear head, reaches PADS PD-vs-ET
precET 0.389 / AUC 0.803 against 0.229 / 0.625 from the same architecture at
random init -- paired +0.161 [+0.111, +0.218] precET.

That result has a design hole. The unlabelled corpus contains
``pads_stretchhold`` and ``pads_relaxed``, which are the very recordings the
test folds are drawn from. No labels leak -- the pretext task is bin
reconstruction -- but the encoder has *seen the test spectra*. That is
transductive SSL. It is common and defensible when stated, but it cannot
support "pretraining transfers to new patients", which is the claim worth
publishing.

Four arms, all evaluated on PADS PD-vs-ET with the encoder frozen (the
configuration that won):

  1. random init                  -- baseline, no pretraining
  2. SSL, full corpus             -- the transductive reference (reproduces 0.389)
  3. SSL, PADS excluded           -- pretrained on 2015 + NewData ONLY, so the
                                     encoder has never seen a PADS recording of
                                     any patient. A cross-cohort transfer claim.
  4. SSL, test patients excluded  -- pretrained per fold on every cohort minus
                                     the recordings of that fold's test
                                     patients. The clean within-cohort claim.

Arm 4 is the one that decides it. If arm 4 collapses to arm 1, the gain was
transduction and must be reported as such. If arm 4 holds, the finding stands.

Input normalisation is taken from each arm's OWN pretraining corpus, so arms 3
and 4 touch no test data at any stage; the random-init arm normalises on the
training fold.

Two reference rows are printed alongside: logistic regression on the ten
spectral descriptors (the standing PADS PD-vs-ET best, AUC 0.807) and on the
same log-binned spectrum the encoder consumes, so the encoder's AUC is read
against something rather than in isolation.

Run: ``python -m experiments.ssl_leakage``
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.cohorts import logbin
from experiments.masked_pretrain import NBIN, SpectrumEncoder, finetune, pretrain
from experiments.pd_vs_et import build as build_labelled

REPEATS, KFOLD = 5, 5


# --------------------------------------------------------------------------- #
# Unlabelled corpus, but keyed by patient so folds can be excluded
# --------------------------------------------------------------------------- #
def unlabelled_keyed():
    """(X, cohort, subject) for every unlabelled recording in every cohort.

    ``cohort`` is "PADS" or "in-house"; ``subject`` is the patient id, which for
    PADS matches the ids the labelled table is built from (both come from
    ``Recording.subject``), so a fold's test patients can be removed exactly.
    """
    from scipy.signal import welch

    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    chunks = []
    for action in ("OUT", "REST", "WING"):
        try:
            chunks.append(("in-house", load_quaternion_recordings(
                "Data", action=action, mode="angular_velocity"), slice(3, 6)))
        except Exception:
            pass
    for task in ALL_TASKS_2025:
        try:
            r = load_2025_all(conditions=(task,))
            if r:
                chunks.append(("in-house", r, slice(3, 6)))
        except Exception:
            pass
    for folder in ("pads_stretchhold", "pads_relaxed"):
        if os.path.isdir(folder):
            try:
                chunks.append(("PADS", load_pads_extracted(folder), slice(0, 3)))
            except Exception:
                pass

    rows, coh, subj = [], [], []
    for tag, recs, ch in chunks:
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            f, P = welch(np.atleast_2d(x), fs=100.0,
                         nperseg=min(512, x.shape[-1]), axis=-1)
            P = P.mean(0)
            v = P[(f >= 3.0) & (f <= 15.0)]
            if v.sum() <= 0:
                continue
            rows.append(v / v.sum())
            coh.append(tag)
            subj.append(str(r.subject))
    X = np.nan_to_num(logbin(np.array(rows)).astype(np.float32))
    return X, np.array(coh), np.array(subj)


# --------------------------------------------------------------------------- #
def scores(y, p):
    pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
    se = recall_score(y, pr, pos_label=1, zero_division=0)
    sp = recall_score(y, pr, pos_label=0, zero_division=0)
    pPD = precision_score(y, pr, pos_label=0, zero_division=0)
    pET = precision_score(y, pr, pos_label=1, zero_division=0)
    return [roc_auc_score(y, p), pPD, pET, 0.5 * (pPD + pET), se, sp]


COLS = ("AUC", "precPD", "precET", "macroP", "ETsens", "PDspec")


def paired(a, b, n=4000):
    """Bootstrap CI on the per-repeat difference a - b."""
    d = a - b
    out = []
    for i in range(len(COLS)):
        boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                        replace=True))
                for s in range(n)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append((d[:, i].mean(), lo, hi))
    return out


def main():
    torch.set_num_threads(1)

    print("building keyed unlabelled corpus ...", flush=True)
    U, Ucoh, Usub = unlabelled_keyed()
    print(f"  {len(U)} recordings   PADS {int((Ucoh=='PADS').sum())}   "
          f"in-house {int((Ucoh=='in-house').sum())}   {U.shape[1]} bins\n",
          flush=True)

    # labelled PADS PD-vs-ET, with patient ids in table-row order
    from common.loaders import load_pads_extracted
    recs = load_pads_extracted("pads_stretchhold")
    pats = np.array(sorted({str(r.subject) for r in recs}))
    data = build_labelled()
    blocks, y3 = data["PADS"]
    keep = y3 != 0
    y = (y3[keep] == 2).astype(int)
    X = np.nan_to_num(blocks["spectrum"][keep])
    Xdesc = np.nan_to_num(blocks["descriptors"][keep])
    pats = pats[keep]
    print(f"PADS PD vs ET  n={len(y)}  ET={int(y.sum())}  "
          f"prevalence {y.mean():.3f}\n", flush=True)

    # ---- pretraining corpora that do not depend on the fold ---------------- #
    t0 = time.time()
    enc_full, norm_full = pretrain(U)
    dt = time.time() - t0
    print(f"  [full corpus pretrain: {dt:.0f}s]", flush=True)
    st_full = {k: v.clone() for k, v in enc_full.state_dict().items()}

    m_noPADS = Ucoh != "PADS"
    enc_np, norm_np = pretrain(U[m_noPADS])
    st_np = {k: v.clone() for k, v in enc_np.state_dict().items()}
    print(f"  [PADS-excluded pretrain on {int(m_noPADS.sum())} recordings]\n",
          flush=True)
    print(f"  arm 4 needs {REPEATS*KFOLD} more pretrains "
          f"(~{REPEATS*KFOLD*dt/60:.0f} min)\n", flush=True)

    # ---- evaluate ---------------------------------------------------------- #
    res = {k: [] for k in ("random init", "SSL full corpus (transductive)",
                           "SSL, PADS excluded", "SSL, test patients excluded",
                           "logreg descriptors", "logreg spectrum")}

    for rep in range(REPEATS):
        folds = list(StratifiedKFold(KFOLD, shuffle=True,
                                     random_state=rep).split(X, y))
        pred = {k: np.zeros(len(y)) for k in res}
        for fi, (tr, te) in enumerate(folds):
            # baseline normalisation from the TRAINING fold only
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            pred["random init"][te] = np.mean(
                [finetune(None, (mu, sd), X[tr], y[tr], X[te], seed=s,
                          freeze=True) for s in (0, 1)], 0)

            for key, st, nm in (("SSL full corpus (transductive)", st_full,
                                 norm_full),
                                ("SSL, PADS excluded", st_np, norm_np)):
                pred[key][te] = np.mean(
                    [finetune(st, nm, X[tr], y[tr], X[te], seed=s, freeze=True)
                     for s in (0, 1)], 0)

            # arm 4: pretrain excluding this fold's test patients
            drop = set(pats[te])
            m_fold = ~np.isin(Usub, list(drop))
            enc_f, norm_f = pretrain(U[m_fold])
            st_f = {k: v.clone() for k, v in enc_f.state_dict().items()}
            pred["SSL, test patients excluded"][te] = np.mean(
                [finetune(st_f, norm_f, X[tr], y[tr], X[te], seed=s, freeze=True)
                 for s in (0, 1)], 0)

            for key, F in (("logreg descriptors", Xdesc),
                           ("logreg spectrum", X)):
                lr = make_pipeline(StandardScaler(),
                                   LogisticRegression(max_iter=5000,
                                                      class_weight="balanced"))
                lr.fit(F[tr], y[tr])
                pred[key][te] = lr.predict_proba(F[te])[:, 1]

            print(f"    repeat {rep} fold {fi} done "
                  f"({int(m_fold.sum())} pretrain recordings, "
                  f"{len(drop)} patients held out)", flush=True)

        for k in res:
            res[k].append(scores(y, pred[k]))
        print(f"  repeat {rep} complete", flush=True)

    for k in res:
        res[k] = np.array(res[k])

    print(f"\n{'='*86}")
    print(f"PADS  PD vs ET  frozen encoder + linear head   "
          f"{REPEATS} repeats x {KFOLD} folds")
    print(f"{'='*86}")
    print(f"{'arm':>32}" + "".join(f"{c:>9}" for c in COLS))
    for k, a in res.items():
        mu_ = a.mean(0)
        print(f"{k:>32}" + "".join(f"{v:>9.3f}" for v in mu_))

    print(f"\n  paired vs random init:")
    for k in ("SSL full corpus (transductive)", "SSL, PADS excluded",
              "SSL, test patients excluded"):
        for (d, lo, hi), c in zip(paired(res[k], res["random init"]), COLS):
            if c in ("AUC", "precET", "macroP"):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {k:>32} {c:>7} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")

    print(f"\n  paired vs the transductive arm (does leakage inflate it?):")
    for k in ("SSL, PADS excluded", "SSL, test patients excluded"):
        for (d, lo, hi), c in zip(
                paired(res[k], res["SSL full corpus (transductive)"]), COLS):
            if c in ("AUC", "precET", "macroP"):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {k:>32} {c:>7} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")

    print(f"\n  paired vs logreg descriptors (the standing best):")
    for k in ("SSL, test patients excluded", "SSL, PADS excluded"):
        for (d, lo, hi), c in zip(paired(res[k], res["logreg descriptors"]),
                                  COLS):
            if c in ("AUC", "precET", "macroP"):
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {k:>32} {c:>7} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")

    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
