"""Cross-dataset PD/N/ET experiment: local data + PADS.

Runs on YOUR machine (the dev sandbox cannot reach PhysioNet/Kaggle). Steps:
  1. Download PADS (PhysioNet DOI 10.13026/m0w9-zx22 or the Kaggle mirror).
  2. Confirm the four VERIFY constants in tremor/pads_data.py against the files.
  3. python -m pdetn.pads_experiment --data-root Data --pads-root /path/to/PADS

Reports P1 (train-local/test-PADS + reverse), P2 (pooled LOSO, the n-fix), and a
dataset-identity probe. Without --pads-root it runs a LOCAL-ONLY dry run to prove
the pipeline works before you have the data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pdetn.crossdataset import (
    build_features, dataset_identity_probe, load_local_hand, protocol_p1,
    protocol_p2_pooled_loso,
)


def _summ(tag, m):
    pc = m["per_class_f1"]; ci = m.get("ET_ci") or m["ci"]["ET"]
    et = pc["ET"]
    print(f"  {tag:>26}: macroF1={m['macro_f1']:.3f}  "
          f"N={pc['N']:.2f} PD={pc['PD']:.2f} ET={et:.3f} "
          f"[{ci['lo']:.2f},{ci['hi']:.2f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Data")
    p.add_argument("--pads-root", default=None,
                   help="PADS dataset root. Omit for a local-only dry run.")
    p.add_argument("--action", default="OUT", help="local action (~ PADS StretchHold)")
    p.add_argument("--pads-condition", default="OUT",
                   help="which PADS condition to use (mapped in pads_data.py)")
    p.add_argument("--output", default="artifacts/pads_experiment")
    args = p.parse_args()

    local = load_local_hand(args.data_root, action=args.action)
    print(f"[local] {len(local)} recordings (hand only)")

    if args.pads_root is None:
        Xl, yl, sl, _ = build_features(local)
        print("[dry-run] LOCAL-ONLY pooled LOSO (proves the pipeline runs):")
        _summ("local pooled LOSO", protocol_p2_pooled_loso(Xl, yl, sl))
        print("\nProvide --pads-root to run the real cross-dataset protocols.")
        return

    from tremor.pads_data import load_pads_recordings
    pads = load_pads_recordings(args.pads_root, conditions=[args.pads_condition])
    print(f"[PADS] {len(pads)} recordings (wrist gyro)")

    Xl, yl, sl, _ = build_features(local)
    Xp, yp, sp, _ = build_features(pads)
    Xa, ya, sa, da = build_features(local + pads)

    out = {}
    print("\n=== P1 external generalisation ===")
    out["p1_local_to_pads"] = protocol_p1(Xl, yl, Xp, yp, sp); _summ("train LOCAL -> test PADS", out["p1_local_to_pads"])
    out["p1_pads_to_local"] = protocol_p1(Xp, yp, Xl, yl, sl); _summ("train PADS -> test LOCAL", out["p1_pads_to_local"])

    print("\n=== P2 pooled LOSO (44 ET) ===")
    out["p2_pooled"] = protocol_p2_pooled_loso(Xa, ya, sa)
    _summ("pooled LOSO", out["p2_pooled"])
    auc = dataset_identity_probe(Xa, da)
    print(f"  dataset-identity probe AUC = {auc:.3f}  "
          f"({'strong domain shift - pooled results confounded' if auc > 0.85 else 'acceptable'})")
    out["dataset_identity_auc"] = auc

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {args.output}/results.json")


if __name__ == "__main__":
    main()
