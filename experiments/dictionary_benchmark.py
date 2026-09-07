"""WEASEL v2 transform with patient-pooled logistic regression (adaptation).

Fixed protocol: PADS, same five patient folds as scattering, seed 0, balanced
logistic C=0.1, dictionary budget setting 4096. No test-driven parameter search.
Not the off-the-shelf WEASEL_V2 ridge classifier. Requires aeon==1.5.0.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from experiments.scattering_benchmark import build, make_splits, metrics, reference
from signal_processing.waveform import waveform

ARMS = ("reference", "dictionary", "fusion_half")


def load_waves(patients):
    from common.loaders import load_pads_extracted
    rows, labels = {p: [] for p in patients}, {}
    for r in load_pads_extracted("pads_stretchhold"):
        p = "PADS:" + r.subject
        if p not in rows or (p in labels and labels[p] != r.y):
            raise ValueError("Unexpected patient or conflicting labels")
        labels[p] = r.y
        if r.x.shape[-1] < 960 or not np.isfinite(r.x).all():
            raise ValueError("Short/nonfinite recording; cannot silently drop it")
        w = waveform(r.x)
        if w is None:
            raise ValueError("Degenerate waveform")
        rows[p].append(w)
    if any(not rows[p] for p in patients):
        raise ValueError("Missing patient recordings")
    return [np.asarray(rows[p], dtype=float) for p in patients], np.array([labels[p] for p in patients])


def stack_records(waves, indices):
    counts = np.array([len(waves[i]) for i in indices])
    if not len(indices) or (counts == 0).any():
        raise ValueError("Empty patient or recording set")
    return (np.concatenate([waves[i] for i in indices])[:, None, :],
            np.repeat(np.arange(len(indices)), counts))


def pool_words(words, owners, n_patients):
    """Average within patient, L1 normalize, square root (Hellinger features)."""
    words = sparse.csr_matrix(words, dtype=float)
    counts = np.bincount(owners, minlength=n_patients)
    if len(counts) != n_patients or (counts == 0).any() or words.shape[0] != len(owners):
        raise ValueError("Missing patient or mismatched word rows")
    if (words.data < 0).any() or not np.isfinite(words.data).all():
        raise ValueError("Word counts must be nonnegative and finite")
    pool = sparse.csr_matrix((1. / counts[owners], (owners, np.arange(len(owners)))),
                             shape=(n_patients, len(owners)))
    X = (pool @ words).tocsr()
    norm = np.asarray(X.sum(axis=1)).ravel()
    X = (sparse.diags(1 / np.maximum(norm, 1e-12)) @ X).tocsr()
    X.data = np.sqrt(X.data)
    return X


def fit_dictionary(waves, y, tr, va, te, transformer_factory=None):
    if transformer_factory is None:
        from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
        transformer_factory = lambda: WEASELTransformerV2(
            min_window=4, norm_options=(False,), word_lengths=(7, 8),
            use_first_differences=(True, False), feature_selection="chi2_top_k",
            max_feature_count=4096, n_jobs=1, random_state=0)
    tf = transformer_factory()
    xt, ot = stack_records(waves, tr)
    Xtr = pool_words(tf.fit_transform(xt, y[tr][ot]), ot, len(tr))
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="lbfgs")
    clf.fit(Xtr, y[tr])
    outputs = []
    for ix in (va, te):
        x, owners = stack_records(waves, ix)
        outputs.append(clf.predict_proba(pool_words(tf.transform(x), owners, len(ix))))
    return *outputs, int(Xtr.shape[1])


def intervals(y, predictions, n=2000):
    rng = np.random.default_rng(0)
    values = {a: [] for a in ARMS[1:]}
    for _ in range(n):
        ix = rng.integers(0, len(y), len(y))
        def score(a):
            p, r, f, _ = precision_recall_fscore_support(y[ix], predictions[a][ix],
                labels=[0, 1, 2], zero_division=0)
            return np.array([f.mean(), p[2], r[2]])
        base = score("reference")
        for a in values:
            values[a].append(score(a) - base)
    return {a: {k: np.percentile(v, [2.5, 97.5], axis=0)[:, j].tolist()
                for j, k in enumerate(("macro_f1", "et_precision", "et_recall"))}
            for a, v in values.items()}


def main():
    import torch
    from common.protocol import tune_offsets
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        parser.error("Output exists; choose a new directory")
    torch.set_num_threads(1)
    d, _ = build("pads")
    waves, wy = load_waves(d["patients"])
    y = d["y"]
    np.testing.assert_array_equal(wy, y)
    splits = make_splits(y, d["patients"], y, folds=5, seed=0)
    out.mkdir(parents=True)
    protocol = dict(method="WEASEL v2 transform + patient-pooled logistic (adaptation)",
        max_feature_count=4096, C=0.1, dictionary_seed=0, outer_seed=0,
        neural_seeds=[0, 1, 2], epochs=200,
        primary_comparison="dictionary minus reference macro_f1",
        secondary_comparison="fixed half fusion minus reference",
        counts=np.bincount(y, minlength=3).tolist(), records_per_patient=sorted(set(map(len, waves))),
        versions={p: importlib.metadata.version(p) for p in
                  ("numpy", "scipy", "scikit-learn", "torch", "aeon", "numba")},
        feature_hashes={k: hashlib.sha256(d[k].tobytes()).hexdigest() for k in ("spec", "desc", "traj")},
        note="Exploratory PADS-only; fixed OOF bootstrap excludes training/selection/site uncertainty")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = [{name: d["patients"][ix].tolist() for name, ix in
                 zip(("train", "validation", "test"), split)} for split in splits]
    (out / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n")
    probabilities = {a: np.zeros((len(y), 3)) for a in ARMS}
    predictions = {a: np.full(len(y), -1) for a in ARMS}
    fold_ids = np.full(len(y), -1)
    selections = []
    for fold, (tr, va, te) in enumerate(splits):
        print(f"Fold {fold + 1}/5: dictionary", flush=True)
        dv, dt, n_features = fit_dictionary(waves, y, tr, va, te)
        print(f"Fold {fold + 1}/5: reference; dictionary features={n_features}", flush=True)
        rv, rt = reference(d, tr, va, te)
        pv = dict(reference=rv, dictionary=dv, fusion_half=(rv + dv) / 2)
        pt = dict(reference=rt, dictionary=dt, fusion_half=(rt + dt) / 2)
        offsets = {}
        for a in ARMS:
            offsets[a] = tune_offsets(pv[a], y[va])
            probabilities[a][te] = pt[a]
            predictions[a][te] = (np.log(pt[a] + 1e-12) + offsets[a]).argmax(1)
        fold_ids[te] = fold
        selections.append(dict(fold=fold, features=n_features,
                               offsets={a: v.tolist() for a, v in offsets.items()}))
        np.savez_compressed(out / f"fold_{fold}.npz", patients=d["patients"][te], y=y[te],
                            **{f"p_{a}": pt[a] for a in ARMS},
                            **{f"pred_{a}": predictions[a][te] for a in ARMS})
        print(f"Fold {fold + 1}/5 complete", flush=True)
    summary = {a: metrics(y, predictions[a], probabilities[a]) for a in ARMS}
    summary.update(paired_95_intervals=intervals(y, predictions), protocol=protocol, selections=selections)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "fold", "label", "arm", "prediction", "pN", "pPD", "pET"])
        for i in range(len(y)):
            for a in ARMS:
                writer.writerow([d["patients"][i], fold_ids[i], y[i], a, predictions[a][i], *probabilities[a][i]])
    for a in ARMS:
        s = summary[a]
        print(f"{a}: macroF1={s['macro_f1']:.3f}, ET precision={s['precision'][2]:.3f}, "
              f"ET recall={s['recall'][2]:.3f}", flush=True)


if __name__ == "__main__":
    main()
