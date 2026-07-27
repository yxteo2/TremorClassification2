"""Patient-level ET-LOSO for the condition-aware fusion model.

Trains :class:`tremor.fusion.ConditionAwareFusion` under the same honest
protocol as ``cv_benchmark`` (subject/patient-level LOSO, val-loss model
selection, pooled test predictions scored once with a subject-level bootstrap
CI + permutation p-value), but one sample is a *patient* — the set of that
patient's REST/OUT/WING recordings — rather than a single recording.

Runs one or more fusion modes (``attention``, ``mean``) so the learned,
condition-tagged attention can be compared against trivial mean-pooling.

Usage:
    python -m tremor.fusion_train --data-root Data --actions OUT,REST,WING \
        --quaternion-sensors hand,lower_arm,upper_arm --f-max 15 \
        --fusion-modes attention mean --encoder cnn \
        --epochs 80 --patience 15 --n-boot 2000 --n-perm 2000 \
        --output artifacts/phase2_fusion/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tremor.data import CLASS_NAMES
from tremor.datasets import TremorDataset
from tremor.evaluate import classification_report, softmax
from tremor.fusion import (
    ConditionAwareFusion, PatientFusionDataset, build_encoder, collate_patients,
)
from tremor.losses import build_loss_fn
from tremor.quaternion import select_sensor_channels
from tremor.quaternion_data import load_quaternion_recordings_multi
from tremor.stats import bootstrap_subject_ci, permutation_test
from tremor.train import _seed_everything


def _patient_loso_folds(patients, plabels, target_label, balance, rng_seed):
    """One fold per target-class patient: hold out that patient + `balance`
    patients from each other class. Every target patient tested once."""
    patients = np.asarray(patients)
    plabels = np.asarray(plabels)
    targets = patients[plabels == target_label]
    others = sorted(set(np.unique(plabels).tolist()) - {target_label})
    rng = np.random.default_rng(rng_seed)
    folds = []
    for tp in targets:
        test = {tp}
        for c in others:
            pool = patients[plabels == c]
            pick = rng.choice(pool, size=min(balance, len(pool)), replace=False)
            test.update(pick.tolist())
        train = [p for p in patients if p not in test]
        folds.append((train, sorted(test)))
    return folds


def _run_loader(model, loader, device):
    model.eval()
    logits, ys, subs = [], [], []
    with torch.no_grad():
        for specs, conds, mask, y, subj in loader:
            out = model(specs.to(device), conds.to(device), mask.to(device))
            logits.append(out.cpu().numpy())
            ys.append(y.numpy())
            subs.extend(subj)
    return np.concatenate(logits), np.concatenate(ys), np.array(subs)


def _train_fold(fusion, base_train, base_eval, train_p, val_p, test_p,
                plabel, args, device):
    train_ds = PatientFusionDataset(base_train, train_p)
    val_ds = PatientFusionDataset(base_eval, val_p)
    test_ds = PatientFusionDataset(base_eval, test_p)
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_patients)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_patients)
    xl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_patients)

    model = ConditionAwareFusion(
        num_classes=len(CLASS_NAMES), embed_dim=args.embed_dim,
        fusion=fusion, dropout=args.dropout,
        encoder=build_encoder(args.encoder, args.embed_dim, args.dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    loss_fn = build_loss_fn(
        loss_type=args.loss, train_labels=[plabel[p] for p in train_p],
        num_classes=len(CLASS_NAMES), focal_gamma=args.focal_gamma,
        label_smoothing=0.0, device=device,
    )

    best_val, best_state, patience = math.inf, None, 0
    for _ in range(args.epochs):
        model.train()
        for specs, conds, mask, y, _ in tl:
            opt.zero_grad()
            out = model(specs.to(device), conds.to(device), mask.to(device))
            loss = loss_fn(out, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            v, n = 0.0, 0
            for specs, conds, mask, y, _ in vl:
                out = model(specs.to(device), conds.to(device), mask.to(device))
                v += float(loss_fn(out, y.to(device))) * len(y)
                n += len(y)
            v /= max(n, 1)
        if v < best_val:
            best_val, patience = v, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            patience += 1
        if patience >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return _run_loader(model, xl, device)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--actions", default="OUT,REST,WING")
    p.add_argument("--feature", default="raw_quaternion")
    p.add_argument("--quaternion-sensors", default="hand,lower_arm,upper_arm")
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--f-max", type=float, default=15.0)
    p.add_argument("--nperseg", type=int, default=128)
    p.add_argument("--nfft", type=int, default=128)
    p.add_argument("--noverlap", type=int, default=96)
    p.add_argument("--normalize", default="per_recording")
    p.add_argument("--fusion-modes", nargs="+", default=["attention", "mean"],
                   choices=["attention", "mean"])
    p.add_argument("--encoder", default="cnn", choices=["cnn", "resnet18"])
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--loss", choices=("ce", "weighted_ce", "focal"), default="focal")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loso-balance", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("artifacts/fusion"))
    args = p.parse_args()

    _seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    recs = load_quaternion_recordings_multi(
        args.data_root, actions=actions, fs=args.fs,
        mode="angular_velocity", feature=args.feature,
    )
    sensors = [s.strip() for s in args.quaternion_sensors.split(",") if s.strip()]
    if sensors and len(sensors) < 3:
        for r in recs:
            r.x = select_sensor_channels(r.x, sensors, mode="angular_velocity")
    print(f"[fusion] {len(recs)} recordings over actions={actions}, "
          f"classes={dict(Counter(r.y for r in recs))}")

    # One label per patient.
    plabel = {r.subject: r.y for r in recs}
    patients = sorted(plabel)
    plabels = [plabel[p] for p in patients]
    print(f"[fusion] {len(patients)} patients, "
          f"per-class={dict(Counter(plabels))}")

    # Global target length so every spectrogram shares (F, T) and stacks.
    target_length = int(min(r.x.shape[1] for r in recs))
    common = dict(
        target_length=target_length, fs=args.fs, nperseg=args.nperseg,
        nfft=args.nfft, noverlap=args.noverlap, f_max=args.f_max,
        normalize=args.normalize, tfd_method="stft", oversample_to=None,
        length_mode="truncate",
    )
    base_train = TremorDataset(recs, augment=True, rng_seed=args.seed, **common)
    base_eval = TremorDataset(recs, augment=False, rng_seed=args.seed + 1, **common)

    target_label = CLASS_NAMES.index("ET")
    folds = _patient_loso_folds(patients, plabels, target_label,
                                args.loso_balance, args.seed)
    print(f"[fusion] ET-LOSO: {len(folds)} folds "
          f"(target={target_length} frames source length)")

    rng = np.random.default_rng(args.seed)
    summary = {}
    for fusion in args.fusion_modes:
        print(f"\n{'='*70}\nFUSION MODE = {fusion}\n{'='*70}")
        pool_logits, pool_y, pool_subj = [], [], []
        for k, (train_p, test_p) in enumerate(folds, 1):
            # carve val patients out of train
            tr = np.array(train_p)
            n_val = max(1, int(0.2 * len(tr)))
            val_p = rng.choice(tr, size=n_val, replace=False).tolist()
            train_only = [p for p in train_p if p not in set(val_p)]
            lg, y, sb = _train_fold(
                fusion, base_train, base_eval, train_only, val_p, test_p,
                plabel, args, device,
            )
            pool_logits.append(lg); pool_y.append(y); pool_subj.append(sb)
            print(f"  fold {k:>2}/{len(folds)}  test_patients={len(test_p)}")

        logits = np.concatenate(pool_logits)
        y = np.concatenate(pool_y)
        subj = np.concatenate(pool_subj)
        rep = classification_report(logits, y, CLASS_NAMES)
        y_pred = softmax(logits).argmax(-1)
        ci = bootstrap_subject_ci(y, y_pred, subj, CLASS_NAMES,
                                  n_boot=args.n_boot, seed=args.seed)
        perm = permutation_test(y, y_pred, subj, CLASS_NAMES,
                                n_perm=args.n_perm, seed=args.seed)
        pc = rep["per_class"]
        print(f"\n[{fusion}] pooled per-patient:")
        print(f"  macroF1={rep['macro_f1']:.3f}  acc={rep['accuracy']:.3f}  "
              f"N_F1={pc['N']['f1']:.2f} PD_F1={pc['PD']['f1']:.2f} "
              f"ET_F1={pc['ET']['f1']:.2f}  ET_AUC={pc['ET']['auc']:.2f}")
        for key in ("macro_f1", "N", "PD", "ET"):
            print(f"    95% CI {key:8s} {ci[key]!r}")
        print(f"    permutation p(macroF1) = {perm['p_value']:.4f}")

        rep["bootstrap_ci"] = {k: {"point": e.point, "lo": e.lo, "hi": e.hi}
                               for k, e in ci.items()}
        rep["permutation_test"] = perm
        (args.output / f"{fusion}_report.json").write_text(json.dumps(rep, indent=2))
        summary[fusion] = {"macro_f1": rep["macro_f1"],
                           "ET_f1": pc["ET"]["f1"], "ET_auc": pc["ET"]["auc"],
                           "ci": rep["bootstrap_ci"]}

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
