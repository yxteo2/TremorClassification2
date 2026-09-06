"""Paired patient-level test of scattering against the current architecture.

Default: PADS postural, full strict cohort, FIVE outer folds, fixed seed 0.
This is not the old merged 40-split headline and must not be compared to it as
if the population/evaluation were unchanged. All arms use identical train,
validation and test patients. Scattering C is selected by 3-fold CV on TRAIN
only; offsets and neural early stopping use validation only. No test-driven
seed, mixture, or model selection. Four predeclared arms are all reported.

Run: python -m experiments.scattering_benchmark --output artifacts/scattering_pads
Optional --cohort merged requires Data/, NewData/, and pads_stretchhold/.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             average_precision_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARMS = ("reference", "scattering_first", "scattering_second", "fusion_half")
C_GRID = (0.01, 0.1, 1.0)


def make_splits(y, patients, strata, folds=5, seed=0):
    """Unique patient rows; fail closed rather than silently splitting recordings."""
    y, patients, strata = map(np.asarray, (y, patients, strata))
    if not (len(y) == len(patients) == len(strata)):
        raise ValueError("Patient, label and stratum lengths must match")
    if len(set(patients.tolist())) != len(patients):
        raise ValueError("Expected exactly one row per patient")
    if min(Counter(strata).values()) < folds:
        raise ValueError("Not enough patients per stratum for requested outer folds")
    if set(y.tolist()) != {0, 1, 2}:
        raise ValueError("All three N/PD/ET classes are required")
    outer = StratifiedKFold(folds, shuffle=True, random_state=seed)
    result = []
    for fold, (tv, te) in enumerate(outer.split(np.zeros(len(y)), strata)):
        a, b = next(StratifiedShuffleSplit(1, test_size=0.2,
                    random_state=seed + fold).split(tv, strata[tv]))
        tr, va = tv[a], tv[b]
        if min(np.bincount(y[tr], minlength=3)) < 3:
            raise ValueError("Need at least three training patients per class")
        if set(y[va]) != {0, 1, 2}:
            raise ValueError("Validation fold is missing a class")
        result.append((tr, va, te))
    return result


def fit_scattering(X, y, tr, va, te):
    # Pipeline is essential: scaler is refit inside EVERY inner fold.
    grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=3000, solver="lbfgs")),
        {"logisticregression__C": C_GRID}, scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        n_jobs=1, error_score="raise")
    grid.fit(X[tr], y[tr])
    return grid.predict_proba(X[va]), grid.predict_proba(X[te]), float(
        grid.best_params_["logisticregression__C"])


def reference(d, tr, va, te, epochs=200):
    """Same two-stream + TCN, three-seed ensemble as final_model.evaluate."""
    from common.protocol import train, NBIN
    from models.architectures import Spectrum1DCNN, ResidualTCN, TwoStreamNet, TRUNKS
    from experiments.final_model import TL
    spec, desc, traj, y = (d[k] for k in ("spec", "desc", "traj", "y"))
    packed = np.hstack([spec, desc, traj])
    builders = [
        (packed, lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                                      8 * 2 * 4, NBIN, desc.shape[1], TL)),
        (spec, lambda: ResidualTCN(NBIN, num_classes=3, ch=16)),
    ]
    outputs = []
    for X, factory in builders:
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        z = (X - mu) / sd
        for seed in (0, 1, 2):
            outputs.append(train(factory, z[tr], y[tr], z[va], y[va],
                                 [z[va], z[te]], seed=seed, epochs=epochs))
    return (np.mean([v[0] for v in outputs], axis=0),
            np.mean([v[1] for v in outputs], axis=0))


def build(cohort):
    from common.loaders import load_pads_extracted
    from common.cohorts import asym_for, desc_table, logbin, N_ASYM
    from experiments.final_model import method_table, TL
    from signal_processing.stability import trajectory_table
    from signal_processing.scattering_features import patient_table

    sources = [("PADS", load_pads_extracted("pads_stretchhold"), slice(0, 3),
                lambda r: "left" if "LeftWrist" in str(r.path) else "right")]
    if cohort == "merged":
        import os
        from common.quaternion_data import load_quaternion_recordings
        from common.load_2025 import load_2025_all, SIDE
        sources = [
            ("2015", load_quaternion_recordings("Data", action="OUT", mode="angular_velocity"),
             slice(3, 6), None),
            ("NewData", load_2025_all(conditions=("OUT",)), slice(3, 6),
             lambda r: SIDE.get(os.path.basename(r.path)[:2])),
        ] + sources
    parts, exclusions = [], {}
    for name, recs, ch, side in sources:
        if not recs:
            raise ValueError(f"No recordings loaded for {name}")
        s1, s2, sy, sp, dropped = patient_table(recs, ch)
        exclusions[name] = dropped
        # Do not silently select a cleaner subset for the candidate than control.
        if dropped:
            raise ValueError(f"{name}: {len(dropped)} excluded recordings; inspect quality before comparison")
        spec, y, p = method_table(recs, "multitaper", ch)
        traj, ty, tp = trajectory_table(recs, ch=ch, n_out=TL)
        if not (np.array_equal(p, sp) and np.array_equal(p, tp)
                and np.array_equal(y, sy) and np.array_equal(y, ty)):
            raise ValueError(f"Patient alignment failed in {name}")
        desc = desc_table(recs, ch)
        if side is None:
            asym, have = np.zeros((len(y), N_ASYM)), np.zeros(len(y))
        else:
            asym, have = asym_for(recs, side, ch, p)
        keep = np.arange(len(y))
        if cohort == "merged" and name == "PADS":
            rng = np.random.default_rng(0)
            keep = np.array(sorted(np.concatenate([rng.choice(np.flatnonzero(y == c),
                min(90, (y == c).sum()), replace=False) for c in (0, 1, 2)])))
        parts.append(dict(spec=logbin(spec)[keep],
            desc=np.hstack([desc, asym, have[:, None]])[keep],
            traj=traj.reshape(len(y), -1)[keep], first=s1[keep], second=s2[keep],
            y=y[keep], patients=np.array([f"{name}:{v}" for v in p[keep]]),
            cohorts=np.repeat(name, len(keep))))
        print(f"{name}: {len(keep)} patients; classes={np.bincount(y[keep], minlength=3).tolist()}", flush=True)
    d = {k: np.concatenate([v[k] for v in parts]) for k in parts[0]}
    for k in ("spec", "desc", "traj", "first", "second"):
        if not np.isfinite(d[k]).all():
            raise ValueError(f"Nonfinite features: {k}")
    return d, exclusions


def metrics(y, pred, probs):
    p, r, f, support = precision_recall_fscore_support(y, pred,
        labels=[0, 1, 2], zero_division=0)
    out = dict(accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_precision=float(p.mean()), macro_f1=float(f.mean()),
        precision=p.tolist(), recall=r.tolist(), f1=f.tolist(),
        support=support.tolist(), confusion=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    if 0 < (y == 2).sum() < len(y):
        out.update(et_average_precision=float(average_precision_score(y == 2, probs[:, 2])),
                   et_auc=float(roc_auc_score(y == 2, probs[:, 2])))
    return out


def paired_intervals(y, predictions, bootstraps=2000):
    """Paired patient bootstrap of fixed OOF predictions, NOT training uncertainty.

    One test prediction per patient. Unstratified resampling allows observed
    prevalence to vary; intervals do not cover the entire model-selection history.
    """
    rng = np.random.default_rng(0)
    diffs = {a: [] for a in ARMS[1:]}
    for _ in range(bootstraps):
        ix = rng.integers(0, len(y), len(y))
        def score(pred):
            p, r, f, _ = precision_recall_fscore_support(y[ix], pred[ix],
                labels=[0, 1, 2], zero_division=0)
            return np.array([f.mean(), p[2], r[2]])
        base = score(predictions["reference"])
        for a in diffs:
            diffs[a].append(score(predictions[a]) - base)
    return {a: {m: np.percentile(v, [2.5, 97.5], axis=0)[:, i].tolist()
                for i, m in enumerate(("macro_f1", "et_precision", "et_recall"))}
            for a, v in diffs.items()}


def run(d, output, folds=5, seed=0, epochs=200, bootstraps=2000):
    import torch
    from common.protocol import tune_offsets
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run
    y = d["y"]
    strata = np.array([f"{c}:{v}" for c, v in zip(d["cohorts"], y)])
    splits = make_splits(y, d["patients"], strata, folds, seed)
    manifest, selections = [], []
    for k, (tr, va, te) in enumerate(splits):
        manifest.append({"fold": k, **{name: d["patients"][ix].tolist()
            for name, ix in (("train", tr), ("validation", va), ("test", te))}})
    (output / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    probs = {a: np.zeros((len(y), 3)) for a in ARMS}
    predictions = {a: np.full(len(y), -1) for a in ARMS}
    fold_ids = np.full(len(y), -1)
    for fold, (tr, va, te) in enumerate(splits):
        print(f"Fold {fold + 1}/{folds}: fitting reference ensemble", flush=True)
        pv, pt = {}, {}
        pv["reference"], pt["reference"] = reference(d, tr, va, te, epochs)
        selected = {"fold": fold, "C": {}, "offsets": {}}
        for a, key in (("scattering_first", "first"), ("scattering_second", "second")):
            pv[a], pt[a], selected["C"][a] = fit_scattering(d[key], y, tr, va, te)
        for pp in (pv, pt):
            pp["fusion_half"] = (pp["reference"] + pp["scattering_second"]) / 2
        for a in ARMS:
            offsets = tune_offsets(pv[a], y[va])
            predictions[a][te] = (np.log(pt[a] + 1e-12) + offsets).argmax(1)
            probs[a][te] = pt[a]
            selected["offsets"][a] = offsets.tolist()
        fold_ids[te] = fold
        selections.append(selected)
        print(f"Fold {fold + 1}/{folds} complete", flush=True)
    summary = {a: metrics(y, predictions[a], probs[a]) for a in ARMS}
    summary["per_cohort"] = {c: {a: metrics(y[d["cohorts"] == c],
        predictions[a][d["cohorts"] == c], probs[a][d["cohorts"] == c]) for a in ARMS}
        for c in np.unique(d["cohorts"])}
    summary["paired_95_intervals"] = paired_intervals(y, predictions, bootstraps)
    summary["protocol"] = dict(outer_folds=folds, split_seed=seed,
        reference_seeds=[0, 1, 2], epochs=epochs, C_grid=list(C_GRID),
        primary_comparison="fusion_half minus reference macro_f1",
        exploratory=True, patient_count=len(y),
        feature_shapes={k: list(d[k].shape) for k in ("spec", "desc", "traj", "first", "second")},
        feature_sha256={k: hashlib.sha256(d[k].tobytes()).hexdigest() for k in
                        ("spec", "desc", "traj", "first", "second")},
        versions={p: importlib.metadata.version(p) for p in
                  ("numpy", "scipy", "scikit-learn", "torch", "kymatio")},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        warning="Internal exploratory OOF evaluation; not external clinical validation. CIs condition on fitted models.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "selections.json").write_text(json.dumps(selections, indent=2) + "\n")
    with (output / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "cohort", "fold", "label", "arm", "prediction", "pN", "pPD", "pET"])
        for i in range(len(y)):
            for a in ARMS:
                writer.writerow([d["patients"][i], d["cohorts"][i], fold_ids[i], y[i], a,
                                 predictions[a][i], *probs[a][i]])
    for a in ARMS:
        s = summary[a]
        print(f"{a}: macroF1={s['macro_f1']:.3f}, macroP={s['macro_precision']:.3f}, "
              f"ET precision={s['precision'][2]:.3f}, ET recall={s['recall'][2]:.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("pads", "merged"), default="pads")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()
    if args.epochs < 1 or args.bootstraps < 1:
        parser.error("epochs and bootstraps must be positive")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new run directory")
    d, _ = build(args.cohort)
    run(d, args.output, args.folds, args.seed, args.epochs, args.bootstraps)


if __name__ == "__main__":
    main()
