"""The re-adjudication list: patients the adopted model is consistently, confidently wrong about.

`self_consistency_gate.md` bounded the label-noise share of errors at ~54 %, and
`selective_90.md` showed those errors **cannot be abstained away** — the model is
confident on them, so no confidence threshold removes them. Every computational
route to the ceiling has now been closed; the largest remaining lever is the
labels themselves, and that needs a clinician.

What a clinician needs is a short, ranked list. The gate measured the population
but never wrote down *who* is in it. This does.

## The criterion

For each patient, across every split in which they fall in the test fold, using
the **adopted 9-member ensemble** (`diverse_ensemble.md`: the 6 incumbents + 3
`SpectrumTransformer` seeds, pooled as family means):

    wrong        the patient-level prediction disagrees with the label
    consistent   every one of the patient's recordings, scored separately, gets
                 that same wrong class

A patient is **flagged** if they were consistently wrong in **at least two thirds
of their test appearances, and in at least two**. One appearance is too few to
separate a stable reading from a bad split.

For 2015 and NewData the recordings are repeats of the same arm; for PADS they are
the left and right wrists, so "consistent" there means both limbs read as the
same wrong class — still the signature, and arguably a stronger one.

## What the list is and is not

**Flagged = mislabelled ∪ genuinely atypical.** Nothing here separates the two;
that is the clinician's job, and each answer is informative either way. A
confirmed wrong label raises the ceiling directly. A confirmed atypical patient
is a correct label on an unusual presentation — worth knowing, and not something
to drop (`prune_training.md` is retracted, so dropping hard patients is neither
shown to help nor to hurt).

**Nothing in this experiment changes a label.** It writes a CSV for review.

## Prediction, recorded before the run

**40–70 patients are flagged; ET-labelled patients are over-represented relative
to their 12 % share of the cohort; and the dominant flagged confusion is PD↔ET,
not either-vs-N.** The size follows from the gate (~11 wrong per test fold, 73 %
of them self-consistent on same-arm repeats); the ET and PD↔ET claims follow from
the contested set being overwhelmingly a PD-vs-ET problem, since N-vs-Tremor
already reaches 0.91–0.92 precision on its own.

20 splits, checkpointed. Run: ``python -m experiments.readjudication_list``
Writes ``reports/readjudication_list.csv``.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments._resume import _DIR
from experiments.final_model import TL
from experiments.self_consistency_gate import build_recording_tables
from models.architectures import (TRUNKS, ResidualTCN, Spectrum1DCNN,
                                  SpectrumTransformer, TwoStreamNet)

SPLITS = 20
SEEDS = (0, 1, 2)
CLS = ("N", "PD", "ET")
OUT = Path("reports/readjudication_list.csv")
CKPT = _DIR / "readjudication_list.json"


def fit_9(spec, packed, Rspec, Rpacked, y, tr, va, te, nd):
    """The adopted 9-member ensemble, pooled as the mean of three family means,
    scoring patient val/test rows and every recording row."""
    fams = (
        (packed, Rpacked, lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8),
                                               TRUNKS["cnn"], 8 * 2 * 4, NBIN,
                                               nd, TL)),
        (spec, Rspec, lambda: ResidualTCN(NBIN, num_classes=3, ch=16)),
        (packed, Rpacked, lambda: TwoStreamNet(SpectrumTransformer(NBIN, 3,
                                                                   d=32),
                                               TRUNKS["trunk"], 32, NBIN, nd,
                                               TL)),
    )
    V, T, R = [], [], []
    for X, XR, mk in fams:
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        o = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                   [(X[va] - mu) / sd, (X[te] - mu) / sd, (XR - mu) / sd],
                   seed=s) for s in SEEDS]
        V.append(np.mean([a[0] for a in o], 0))
        T.append(np.mean([a[1] for a in o], 0))
        R.append(np.mean([a[2] for a in o], 0))
    return np.mean(V, 0), np.mean(T, 0), np.mean(R, 0)


def main():
    torch.set_num_threads(1)
    T_ = build_recording_tables()
    y, key, SPEC, D, TR = (T_[k] for k in ("y", "key", "SPEC", "D", "TR"))
    Rspec, Rdesc, Rtraj = T_["Rspec"], T_["Rdesc"], T_["Rtraj"]
    rec2pat, Rcoh, pat_ids = T_["rec2pat"], T_["Rcoh"], T_["pat_ids"]
    packed = np.hstack([SPEC, D, TR])
    Rpacked = np.hstack([Rspec, Rdesc, Rtraj])
    nd = D.shape[1]
    coh_of = {int(p): str(c) for p, c in zip(rec2pat, Rcoh)}
    n_rec = np.bincount(rec2pat, minlength=len(y))
    print(f"n={len(y)} patients, {len(rec2pat)} recordings, 9-member ensemble")
    print("prediction on record: 40-70 flagged; ET over-represented vs its 12 %"
          " share; dominant confusion PD<->ET\n", flush=True)

    # per-patient tallies, checkpointed per split
    st = {"done": [], "n_test": [0] * len(y), "n_wrong": [0] * len(y),
          "n_cons": [0] * len(y), "pred": [[] for _ in y],
          "conf": [[] for _ in y]}
    if CKPT.exists():
        st = json.loads(CKPT.read_text())
        print(f"[resume] {len(st['done'])} split(s) recovered", flush=True)

    for sp in range(SPLITS):
        if sp in st["done"]:
            continue
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(y, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(y[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv, pt, pr = fit_9(SPEC, packed, Rspec, Rpacked, y, tr, va, te, nd)
        off = tune_offsets(pv, y[va])
        pat_pred = (np.log(pt + 1e-12) + off).argmax(1)
        p_adj = np.exp(np.log(pt + 1e-12) + off)
        p_adj /= p_adj.sum(1, keepdims=True)
        rec_pred = (np.log(pr + 1e-12) + off).argmax(1)
        for j, p in enumerate(te):
            p = int(p)
            st["n_test"][p] += 1
            wrong = int(pat_pred[j]) != int(y[p])
            ks = np.flatnonzero(rec2pat == p)
            cons = (wrong and len(ks) >= 2
                    and all(int(rec_pred[k]) == int(pat_pred[j]) for k in ks))
            st["n_wrong"][p] += int(wrong)
            st["n_cons"][p] += int(cons)
            st["pred"][p].append(int(pat_pred[j]))
            st["conf"][p].append(float(p_adj[j].max()))
        st["done"].append(sp)
        tmp = CKPT.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(CKPT)
        print(f"  split {sp + 1}/{SPLITS}", flush=True)

    rows = []
    for p in range(len(y)):
        nt, nc = st["n_test"][p], st["n_cons"][p]
        if nt >= 2 and nc >= 2 and nc / nt >= 2 / 3:
            wrong_preds = [c for c in st["pred"][p] if c != int(y[p])]
            modal = Counter(wrong_preds).most_common(1)[0][0]
            rows.append({
                "subject": str(pat_ids[p]), "cohort": coh_of.get(p, "?"),
                "label": CLS[int(y[p])], "model_says": CLS[modal],
                "test_appearances": nt, "consistently_wrong": nc,
                "rate": round(nc / nt, 2), "n_recordings": int(n_rec[p]),
                "mean_confidence": round(float(np.mean(st["conf"][p])), 3),
            })
    rows.sort(key=lambda r: (-r["rate"], -r["consistently_wrong"],
                             -r["mean_confidence"]))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                           ["subject"])
        w.writeheader()
        w.writerows(rows)

    lab = Counter(r["label"] for r in rows)
    pair = Counter(f"{r['label']}->{r['model_says']}" for r in rows)
    coh = Counter(r["cohort"] for r in rows)
    base = Counter(CLS[int(c)] for c in y)
    print(f"\nflagged: {len(rows)} of {len(y)} patients "
          f"({len(rows) / len(y):.0%})   written to {OUT}")
    print("\nby label (flagged share vs cohort share):")
    for c in CLS:
        print(f"  {c:>3}: {lab[c]:>3} flagged = {lab[c] / max(len(rows), 1):5.0%}"
              f"   vs {base[c] / len(y):5.0%} of cohort")
    print("\nconfusions (label -> model says):")
    for k, v in pair.most_common():
        print(f"  {k:>8}: {v}")
    print("\nby cohort:", dict(coh))
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
