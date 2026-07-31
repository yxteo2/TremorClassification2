"""Deep BiLSTM cross-dataset experiment (local + PADS), PD-vs-ET framing.

Compares two deep variants -- 3-class vs two-stage -- under two protocols:
  P1  train LOCAL -> test PADS   (generalisation)
  P2  pooled GroupKFold (5-fold, subject-grouped) over LOCAL+PADS   (n-fix)

Run on your machine (needs torch + PADS; the dev sandbox can't download PADS):
  python -m pdetn.pads_deep_experiment --data-root Data --pads-root PADS

Without --pads-root: a LOCAL-only dry run (internal split) to prove it works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from pdetn.crossdataset import load_local_sensor
from pdetn.deep_crossdataset import (
    DeepTwoStage, pd_vs_et_metrics, predict_3class, train_3class,
)


def _target_length(recs):
    return int(min(r.x.shape[1] for r in recs))


def _split_val(recs, frac=0.2, seed=0):
    subs = sorted({r.subject for r in recs})
    rng = np.random.default_rng(seed)
    val = set(rng.choice(subs, max(1, int(frac * len(subs))), replace=False).tolist())
    tr = [r for r in recs if r.subject not in val]
    vl = [r for r in recs if r.subject in val]
    return tr, vl


def _run_variant(name, train_recs, val_recs, test_recs, tl, epochs):
    if name == "3class":
        m = train_3class(train_recs, val_recs, tl, epochs=epochs)
        yp = predict_3class(m, test_recs, tl)
    else:
        m = DeepTwoStage(tl, epochs=epochs).fit(train_recs, val_recs)
        yp = m.predict(test_recs)
    yt = [r.y for r in test_recs]
    return pd_vs_et_metrics(yt, yp)


def protocol_p1(local, pads, epochs):
    tl = _target_length(local)
    tr, vl = _split_val(local)
    out = {}
    for v in ("3class", "two_stage"):
        out[v] = _run_variant(v, tr, vl, pads, tl, epochs)
    return out


def protocol_p2(allrecs, epochs, n_splits=5):
    tl = _target_length(allrecs)
    subj = np.array([r.subject for r in allrecs])
    y = np.array([r.y for r in allrecs])
    gkf = GroupKFold(n_splits=n_splits)
    pooled = {v: (np.zeros(len(allrecs), int)) for v in ("3class", "two_stage")}
    for tr_i, te_i in gkf.split(allrecs, y, groups=subj):
        train_all = [allrecs[i] for i in tr_i]
        tr, vl = _split_val(train_all)
        test = [allrecs[i] for i in te_i]
        for v in ("3class", "two_stage"):
            if v == "3class":
                m = train_3class(tr, vl, tl, epochs=epochs)
                pooled[v][te_i] = predict_3class(m, test, tl)
            else:
                m = DeepTwoStage(tl, epochs=epochs).fit(tr, vl)
                pooled[v][te_i] = m.predict(test)
    return {v: pd_vs_et_metrics(y, pooled[v]) for v in pooled}


def _show(tag, r):
    pc = r["per_class_f1"]
    print(f"  {tag:>22}: PD-vs-ET acc={r['pd_vs_et_acc']:.3f}  macroF1={r['macro_f1']:.3f}  "
          f"N={pc['N']:.2f} PD={pc['PD']:.2f} ET={pc['ET']:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Data")
    p.add_argument("--pads-root", default=None)
    p.add_argument("--action", default="OUT")
    p.add_argument("--pads-condition", default="OUT")
    p.add_argument("--sensor", default="lower_arm")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--output", default="artifacts/pads_deep")
    args = p.parse_args()

    local = load_local_sensor(args.data_root, action=args.action, sensor=args.sensor)
    print(f"[local] {len(local)} recordings ({args.sensor})")

    if args.pads_root is None:
        tl = _target_length(local)
        tr, vl = _split_val(local, frac=0.25)
        print("[dry-run] LOCAL internal split, both variants (proves it runs):")
        for v in ("3class", "two_stage"):
            _show(f"dry {v}", _run_variant(v, tr, vl, vl, tl, args.epochs))
        print("\nProvide --pads-root for the real P1/P2 cross-dataset comparison.")
        return

    from tremor.pads_data import load_pads_recordings
    pads = load_pads_recordings(args.pads_root, conditions=[args.pads_condition])
    print(f"[PADS] {len(pads)} recordings (wrist)")

    out = {}
    print("\n=== P1 train-LOCAL / test-PADS ===")
    out["p1"] = protocol_p1(local, pads, args.epochs)
    for v, r in out["p1"].items():
        _show(f"P1 {v}", r)
    print("\n=== P2 pooled GroupKFold (LOCAL+PADS) ===")
    out["p2"] = protocol_p2(local + pads, args.epochs)
    for v, r in out["p2"].items():
        _show(f"P2 {v}", r)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {args.output}/results.json")


if __name__ == "__main__":
    main()
