"""End-to-end N/PD/ET separation experiment (condition-aware, CPU, no GPU).

Loads the multi-condition data, builds per-patient condition-aware feature
vectors, and evaluates flat vs two-stage classifiers under leave-one-patient-out
with subject bootstrap CIs.

    python -m pdetn.run --data-root Data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tremor.quaternion_data import load_quaternion_recordings_multi

from pdetn.evaluate import evaluate, print_result
from pdetn.features import CONDITIONS, build_patient_table
from pdetn.model import FlatClassifier, TwoStageClassifier


def load_features(data_root, conditions=CONDITIONS, fs=100.0):
    recs = load_quaternion_recordings_multi(
        data_root, actions=list(conditions), fs=fs, mode="angular_velocity",
    )
    return build_patient_table(recs, conditions=conditions, fs=fs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="Data", type=Path)
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--estimator", default="rf", choices=["rf", "logreg", "lda"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--output", type=Path, default=Path("artifacts/pdetn"))
    args = p.parse_args()
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    X, y, subjects, names = load_features(args.data_root, conditions)
    print(f"[pdetn] {len(subjects)} patients, {X.shape[1]} features, "
          f"conditions={conditions}")

    results = {}
    results["flat"] = evaluate(
        lambda: FlatClassifier(args.estimator), X, y, subjects, n_boot=args.n_boot)
    print_result(f"FLAT ({args.estimator})", results["flat"])

    results["two_stage"] = evaluate(
        lambda: TwoStageClassifier(args.estimator, args.estimator),
        X, y, subjects, n_boot=args.n_boot)
    print_result(f"TWO-STAGE ({args.estimator})", results["two_stage"])

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}/results.json")


if __name__ == "__main__":
    main()
