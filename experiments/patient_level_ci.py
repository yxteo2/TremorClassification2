"""Conditional patient-bootstrap uncertainty for the CURRENT headline pipeline.

Both arms use final_model.evaluate and the same 40 splits as headline_audit.
A patient is drawn once per replicate, with its multiplicity reused in every
split and both arms. Intervals condition on fitted models and observed cohorts;
they omit training-sample and model-selection uncertainty and do not establish
external generalisation. Split resampling only describes split sensitivity.
Neither interval is guaranteed to be wider than the other.

Run: python -m experiments.patient_level_ci --output patient_ci_current.json
The output retains predictions, patient identities and source/data fingerprints.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

NBOOT = 4000
NM = ("precN", "precPD", "precET", "macroP", "macroF1")
N_SPLITS = 40


def evaluate_keep(spec, desc, traj, y, key, splits=N_SPLITS):
    """Use the same trainer as the headline, retaining patient predictions."""
    from experiments.final_model import evaluate
    return evaluate("patient CI", spec, desc, traj, y, key, splits=splits,
                    verbose=False, return_predictions=True)


def metrics_on(y_true, y_pred):
    P, _, F, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2],
                                                 zero_division=0)
    return np.array([P[0], P[1], P[2], P.mean(), F.mean()])


def patient_bootstrap(y, ps_a, ps_b, n_patients, n=NBOOT, seed=0):
    """Paired difference (b - a) resampling PATIENTS, splits held fixed.

    For each replicate a single patient multiset is drawn and used for **both**
    arms and **every** split, so fold assignment and model fitting contribute
    nothing to the spread -- only which patients were sampled.
    """
    y = np.asarray(y)
    if len(y) != n_patients or n_patients < 1 or n < 1:
        raise ValueError("Invalid patient count or bootstrap count")
    if len(ps_a) != len(ps_b) or not len(ps_a):
        raise ValueError("Both arms must have the same nonzero split count")
    for (ta, pa), (tb, pb) in zip(ps_a, ps_b):
        if not np.array_equal(ta, tb):
            raise ValueError("Paired arms must have identical ordered test patients")
        ta = np.asarray(ta)
        if (ta.ndim != 1 or not np.issubdtype(ta.dtype, np.integer)
                or len(ta) == 0 or len(np.unique(ta)) != len(ta)
                or np.any(ta < 0) or np.any(ta >= n_patients)
                or np.shape(pa) != ta.shape or np.shape(pb) != ta.shape):
            raise ValueError("Invalid patient indices or prediction shape")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, n_patients, n_patients),
                          minlength=n_patients)
        da, db = [], []
        for (te, pa), (_, pb) in zip(ps_a, ps_b):
            rep = np.repeat(np.arange(len(te)), cnt[te])
            if not len(rep):
                continue
            # Fixed labels/zero_division policy matches the original metric,
            # including draws missing a class; never filter on observed labels.
            da.append(metrics_on(y[te][rep], np.asarray(pa)[rep]))
            db.append(metrics_on(y[te][rep], np.asarray(pb)[rep]))
        if da:
            out.append(np.mean(db, 0) - np.mean(da, 0))
    if not out:
        raise ValueError("No bootstrap draw contained an evaluated patient")
    return np.array(out)


def save_predictions(path, d, arms):
    """Save rerunnable audit inputs, with code and feature provenance."""
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("common", "experiments", "models", "frequency", "signal_processing")
        for p in sorted((root / folder).glob("*.py"))}
    features = hashlib.sha256()
    for value in [d["y"], d["patient_ids"], d["key"], d["DESC"], d["ASYM"],
                  d["HAVE"], d["TRAJ"], *[d["SPEC"][k] for k in sorted(d["SPEC"])]]:
        a = np.ascontiguousarray(value)
        features.update(str((a.dtype, a.shape)).encode())
        features.update(a.tobytes())
    payload = dict(schema_version=1, revision=revision, source_sha256=source_hashes,
                   feature_sha256=features.hexdigest(), labels=d["y"].tolist(),
                   patient_ids=d["patient_ids"].tolist(), strata=d["key"].tolist(),
                   metrics=list(NM), arms={})
    for name, (rows, splits) in arms.items():
        payload["arms"][name] = dict(metrics=rows.tolist(), splits=[
            dict(test_indices=te.tolist(), predictions=pred.tolist())
            for te, pred in splits])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def split_bootstrap(diff, n=NBOOT):
    out = []
    for i in range(diff.shape[1]):
        b = [np.mean(np.random.default_rng(s).choice(diff[:, i], len(diff),
                                                     replace=True))
             for s in range(n)]
        out.append(np.percentile(b, [2.5, 97.5]))
    return np.array(out)


def main():
    import argparse
    import torch
    from experiments.final_model import build

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="patient_ci_current.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    d = build()
    y, key, SPEC = d["y"], d["key"], d["SPEC"]
    D_desc = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {N_SPLITS} splits, {NBOOT} bootstrap draws\n")

    print("running welch baseline ...", flush=True)
    a_rows, a_ps = evaluate_keep(SPEC["welch"], D_desc, None, y, key)
    print("running multitaper + trajectory ...", flush=True)
    b_rows, b_ps = evaluate_keep(SPEC["multitaper"], D_desc, d["TRAJ"], y, key)

    print(f"\n{'arm':>34}" + "".join(f"{c:>9}" for c in NM))
    print(f"{'welch + desc + asym (baseline)':>34}" +
          "".join(f"{v:>9.3f}" for v in a_rows.mean(0)))
    print(f"{'multitaper + trajectory':>34}" +
          "".join(f"{v:>9.3f}" for v in b_rows.mean(0)))

    save_predictions(args.output, d, {"base": (a_rows, a_ps), "mt_t": (b_rows, b_ps)})
    diff = b_rows - a_rows
    sci = split_bootstrap(diff)
    pb = patient_bootstrap(y, a_ps, b_ps, len(y))
    pci = np.percentile(pb, [2.5, 97.5], axis=0).T

    print(f"\n{'':>10}{'diff':>9}{'split-level 95 %':>22}"
          f"{'patient-level 95 %':>24}{'  width x'}")
    for i, nm in enumerate(NM):
        lo_s, hi_s = sci[i]
        lo_p, hi_p = pci[i]
        s_s = "*" if lo_s > 0 or hi_s < 0 else " "
        s_p = "*" if lo_p > 0 or hi_p < 0 else " "
        w = (hi_p - lo_p) / (hi_s - lo_s + 1e-12)
        print(f"{nm:>10}{diff[:, i].mean():>+9.3f}"
              f"{f'[{lo_s:+.3f}, {hi_s:+.3f}] {s_s}':>22}"
              f"{f'[{lo_p:+.3f}, {hi_p:+.3f}] {s_p}':>24}{w:>9.1f}")

    print("\n* = interval excludes zero, conditional on fitted models.")
    print("Neither column accounts for training-sample or model-selection uncertainty.")
    print("Split intervals measure split sensitivity; patient intervals resample patients.")
    print(f"Predictions and provenance saved to {args.output}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()

