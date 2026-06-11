"""Subject-level 5-fold CV benchmark for TFD method comparison.

The script trains a BiLSTM for each (TFD method × fold) combination on the
SAME subject-level partitioning, so the only thing that varies between
methods is the time-frequency representation. Designed to produce the
headline table for a comparison paper.

Output:
    <output>/results.json   per-(method, fold) metrics
    <output>/summary.csv    mean ± SD per method, per class
    <output>/<method>_fold<k>/model.pt    best checkpoint each fold

Usage:
    python -m tremor.cv_benchmark \
        --data-root /path/to/Drive --action OUT \
        --data-mode quaternion --model bilstm \
        --tfd-methods stft cwt hht wavelet_packet \
        --epochs 60 --patience 15 --n-folds 5 \
        --output cv_results/OUT/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from tremor.data import CLASS_NAMES
from tremor.evaluate import classification_report
from tremor.losses import build_loss_fn, compute_class_weights
from tremor.models import build_model
from tremor.quaternion import select_sensor_channels
from tremor.quaternion_data import load_quaternion_recordings
from tremor.stft_data import load_stft_recordings
from tremor.train import TremorDataset, STFTDataset, _seed_everything


def _load_recs(args):
    if args.data_mode == "quaternion":
        feature = "raw_quaternion" if args.feature == "stft" else args.feature
        recs = load_quaternion_recordings(
            args.data_root, action=args.action, fs=args.fs,
            mode="angular_velocity", feature=feature,
        )
        sensors = [s.strip() for s in (args.quaternion_sensors or "").split(",") if s.strip()]
        if sensors and len(sensors) < 3:
            for r in recs:
                r.x = select_sensor_channels(r.x, sensors, mode="angular_velocity")
            print(f"[cv] kept sensors={sensors}  channels={recs[0].x.shape[0]}")
        return recs
    if args.data_mode == "stft":
        return load_stft_recordings(
            args.data_root, action=args.action, feature=args.feature,
        )
    raise ValueError(f"--data-mode {args.data_mode} not supported here")


def _build_target_class_loso_splits(
    subjects: np.ndarray, labels: np.ndarray, target_label: int,
    balance_subjects_per_class: int, rng_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fold per unique subject of ``target_label``.

    Each fold's test set holds out THAT one target subject's recordings
    plus ``balance_subjects_per_class`` randomly-sampled subjects from
    every other class — so per-class F1 is well-defined per fold. The
    remaining recordings form the trainval pool. Pooling the per-fold
    predictions covers every target-class subject exactly once.
    """
    target_subjects = np.unique(subjects[labels == target_label])
    if len(target_subjects) == 0:
        raise SystemExit(f"No subjects found with target_label={target_label}")
    other_classes = sorted(set(np.unique(labels).tolist()) - {target_label})
    rng = np.random.default_rng(rng_seed)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for tsubj in target_subjects:
        test_subj_set = {tsubj}
        for c in other_classes:
            pool = np.unique(subjects[labels == c])
            n_pick = min(balance_subjects_per_class, len(pool))
            picked = rng.choice(pool, size=n_pick, replace=False)
            test_subj_set.update(picked.tolist())
        test_mask = np.isin(subjects, list(test_subj_set))
        splits.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return splits


def _make_dataset(recs, target_T, args, tfd_method, *, augment, rng_seed,
                  oversample_to):
    if args.data_mode == "quaternion":
        return TremorDataset(
            recs, target_length=target_T, fs=args.fs,
            nperseg=args.nperseg, nfft=args.nfft, noverlap=args.noverlap,
            f_max=args.f_max, rng_seed=rng_seed, augment=augment,
            tfd_method=tfd_method, cwt_w0=args.cwt_w0, cwt_decim=args.cwt_decim,
            cwt_freq_step=args.cwt_freq_step, hht_max_imfs=args.hht_max_imfs,
            wp_level=args.wp_level, wp_wavelet=args.wp_wavelet,
            log_compress_on=not args.no_log_compress, normalize=args.normalize,
            length_mode=args.length_mode, pad_mode=args.pad_mode,
            spec_augment_on=args.spec_augment, oversample_to=oversample_to,
        )
    # stft mode ignores tfd_method (data is already precomputed STFT)
    return STFTDataset(
        recs, target_T=target_T, rng_seed=rng_seed, n_sensors=args.stft_n_sensors,
        n_freq_bins=args.stft_n_freq_bins,
        keep_bins=args.stft_n_freq_bins,  # no extra crop here
        log_compress_on=not args.no_log_compress, normalize=args.normalize,
        spec_augment_on=args.spec_augment, length_mode=args.length_mode,
        pad_mode=args.pad_mode, oversample_to=oversample_to, augment=augment,
    )


def _train_one(method, fold_idx, train_recs, val_recs, args, device):
    """Train a single (method, fold) and return its test_recs accuracy + state_dict.

    For CV we use train+val from this fold; test_recs is set externally."""
    lengths = [r.x.shape[1] for r in train_recs + val_recs]
    target_T = int(max(lengths) if args.length_mode == "pad" else min(lengths))
    if target_T < 4:
        raise SystemExit(f"target length {target_T} too small")

    train_ds = _make_dataset(
        train_recs, target_T, args, tfd_method=method,
        augment=True, rng_seed=args.seed,
        oversample_to=args.oversample_to,
    )
    val_ds = _make_dataset(
        val_recs, target_T, args, tfd_method=method,
        augment=False, rng_seed=args.seed + 1, oversample_to=None,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    input_size = sample_x.shape[0]
    # post-TFD time-frame count, not the raw sample length (AST needs this)
    input_tdim = sample_x.shape[1]
    model = build_model(
        name=args.model, input_size=input_size,
        num_classes=len(CLASS_NAMES), target_T=input_tdim,
        hidden=args.hidden, dropout=args.dropout,
        pretrained=getattr(args, "ast_pretrained", True),
        freeze_backbone=getattr(args, "freeze_backbone", True),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
    loss_fn = build_loss_fn(
        loss_type=args.loss,
        train_labels=[r.y for r in train_recs],
        num_classes=len(CLASS_NAMES),
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        device=device,
    )

    best_val, best_state, patience = math.inf, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_loss, n = 0.0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_loss += float(loss_fn(model(x), y)) * x.size(0)
                n += x.size(0)
            v_loss /= max(n, 1)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{method} fold {fold_idx}: training never improved")
    return model, best_state, target_T, input_size


@torch.no_grad()
def _eval_on_recs(model, recs, target_T, args, method, device):
    test_ds = _make_dataset(
        recs, target_T, args, tfd_method=method,
        augment=False, rng_seed=args.seed + 2, oversample_to=None,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    model.eval()
    logits_list, y_list = [], []
    for x, y in loader:
        logits_list.append(model(x.to(device)).cpu().numpy())
        y_list.append(y.numpy())
    return np.concatenate(logits_list), np.concatenate(y_list)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--action", required=True)
    p.add_argument("--data-mode", choices=("quaternion", "stft"), default="quaternion")
    p.add_argument("--feature", default="stft")
    p.add_argument("--tfd-methods", nargs="+",
                   default=["stft", "cwt", "hht", "wavelet_packet"],
                   choices=["stft", "cwt", "hht", "wavelet_packet",
                            "multitaper", "sst"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default="bilstm")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=("ce", "weighted_ce", "focal"), default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--oversample-to", type=int, default=None,
                   help="Per-class oversampling target. Pass a positive int "
                        "(e.g. 3x the largest class) to fight ET-class collapse.")
    # Pipeline knobs (mirroring train.py)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--nperseg", type=int, default=128)
    p.add_argument("--nfft", type=int, default=128)
    p.add_argument("--noverlap", type=int, default=96)
    p.add_argument("--f-max", type=float, default=30.0)
    p.add_argument("--cwt-w0", type=float, default=6.0)
    p.add_argument("--cwt-decim", type=int, default=8)
    p.add_argument("--cwt-freq-step", type=float, default=0.5)
    p.add_argument("--hht-max-imfs", type=int, default=8)
    p.add_argument("--wp-level", type=int, default=5)
    p.add_argument("--wp-wavelet", default="db4")
    p.add_argument("--stft-n-sensors", type=int, default=3)
    p.add_argument("--stft-n-freq-bins", type=int, default=65)
    p.add_argument("--no-log-compress", action="store_true")
    p.add_argument("--normalize",
                   choices=("none", "per_freq", "per_recording"),
                   default="per_recording")
    p.add_argument("--spec-augment", action="store_true")
    p.add_argument("--length-mode", choices=("truncate", "pad"), default="truncate")
    p.add_argument("--pad-mode", choices=("random", "center"), default="random")
    p.add_argument("--quaternion-sensors", default="hand",
                   help="Comma-separated subset of {hand, lower_arm, upper_arm}.")
    p.add_argument("--loso-target-class", default="",
                   help="If set (e.g. 'ET'), do target-class LOSO instead of "
                        "GroupKFold: one fold per unique subject of that class, "
                        "each tested exactly once. Predictions are pooled into "
                        "one honest per-class number — the right protocol when "
                        "one class is small.")
    p.add_argument("--loso-balance-subjects", type=int, default=2,
                   help="(loso mode) Non-target subjects per other class added "
                        "to each fold's test set for class balance.")
    p.add_argument("--ast-pretrained", dest="ast_pretrained",
                   action="store_true", default=True,
                   help="(ast/resnet18) Use pretrained backbone weights (default).")
    p.add_argument("--no-ast-pretrained", dest="ast_pretrained",
                   action="store_false",
                   help="(ast/resnet18) Train backbone from scratch.")
    p.add_argument("--freeze-backbone", dest="freeze_backbone",
                   action="store_true", default=True,
                   help="(ast/resnet18) Freeze the pretrained backbone (default).")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                   action="store_false",
                   help="(ast/resnet18) Fine-tune the whole backbone.")
    p.add_argument("--output", type=Path, default=Path("cv_results"))
    args = p.parse_args()

    _seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[cv] data-root={args.data_root}  action={args.action}  "
          f"data-mode={args.data_mode}")
    recs = _load_recs(args)
    print(f"[cv] {len(recs)} recordings, "
          f"classes={dict(Counter(r.y for r in recs))}, "
          f"{len(set(r.subject for r in recs))} subjects")

    if not recs:
        raise SystemExit("no recordings loaded")

    subjects = np.array([r.subject for r in recs])
    labels   = np.array([r.y for r in recs])

    if args.oversample_to is None:
        # Default: oversample ET to match the largest class count
        counts = Counter(labels.tolist())
        args.oversample_to = max(counts.values())
        print(f"[cv] --oversample-to auto-set to {args.oversample_to}")
    elif args.oversample_to <= 0:
        args.oversample_to = None
        print("[cv] oversampling disabled (rely on focal loss only)")

    # Choose the cross-validation scheme.
    if args.loso_target_class:
        if args.loso_target_class not in CLASS_NAMES:
            raise SystemExit(
                f"--loso-target-class={args.loso_target_class!r} not in {CLASS_NAMES}"
            )
        target_label = CLASS_NAMES.index(args.loso_target_class)
        split_iter = _build_target_class_loso_splits(
            subjects, labels, target_label,
            balance_subjects_per_class=args.loso_balance_subjects,
            rng_seed=args.seed,
        )
        n_folds = len(split_iter)
        print(f"[cv] target-class LOSO on '{args.loso_target_class}' "
              f"({n_folds} folds)")
    else:
        gkf = GroupKFold(n_splits=args.n_folds)
        split_iter = list(gkf.split(np.zeros(len(recs)), labels, groups=subjects))
        n_folds = args.n_folds

    all_results: dict[str, list[dict]] = defaultdict(list)
    pooled_logits: dict[str, list[np.ndarray]] = defaultdict(list)
    pooled_y: dict[str, list[np.ndarray]] = defaultdict(list)

    for fold_idx, (trainval_idx, test_idx) in enumerate(split_iter, start=1):
        # Carve val out of trainval
        rng = np.random.default_rng(args.seed + fold_idx)
        trainval_subj = np.unique(subjects[trainval_idx])
        n_val = max(1, int(0.2 * len(trainval_subj)))
        val_subj = rng.choice(trainval_subj, size=n_val, replace=False)
        val_mask = np.isin(subjects, val_subj)
        train_idx = trainval_idx[~np.isin(subjects[trainval_idx], val_subj)]
        val_idx_arr = trainval_idx[np.isin(subjects[trainval_idx], val_subj)]

        train_recs = [recs[i] for i in train_idx]
        val_recs   = [recs[i] for i in val_idx_arr]
        test_recs  = [recs[i] for i in test_idx]

        train_dist = dict(Counter(r.y for r in train_recs))
        test_dist  = dict(Counter(r.y for r in test_recs))
        print(f"\n=== fold {fold_idx}/{n_folds} ===")
        print(f"  train: {len(train_recs)} ({train_dist})")
        print(f"  val:   {len(val_recs)}")
        print(f"  test:  {len(test_recs)} ({test_dist})")

        for method in args.tfd_methods:
            print(f"\n  [fold {fold_idx}] training method={method}")
            try:
                model, best_state, target_T, _ = _train_one(
                    method, fold_idx, train_recs, val_recs, args, device,
                )
                model.load_state_dict(best_state)
                logits, y_true = _eval_on_recs(
                    model, test_recs, target_T, args, method, device,
                )
                report = classification_report(logits, y_true, CLASS_NAMES)

                fold_dir = args.output / method / f"fold{fold_idx}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, fold_dir / "model.pt")
                (fold_dir / "report.json").write_text(json.dumps(report, indent=2))

                all_results[method].append({
                    "fold": fold_idx,
                    "accuracy": report["accuracy"],
                    "per_class": report["per_class"],
                    "confusion_matrix": report["confusion_matrix"],
                    "n_train": len(train_recs), "n_test": len(test_recs),
                })
                pooled_logits[method].append(logits)
                pooled_y[method].append(y_true)
                print(f"    -> acc={report['accuracy']:.3f}  "
                      f"N_f1={report['per_class']['N']['f1']:.2f}  "
                      f"PD_f1={report['per_class']['PD']['f1']:.2f}  "
                      f"ET_f1={report['per_class']['ET']['f1']:.2f}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                all_results[method].append({"fold": fold_idx, "error": str(e)})

    # Save raw results
    (args.output / "results.json").write_text(
        json.dumps({k: v for k, v in all_results.items()}, indent=2)
    )

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY  (mean ± SD across folds)")
    print("=" * 72)
    print(f"{'method':>15s}  {'accuracy':>14s}  {'N_F1':>13s}  "
          f"{'PD_F1':>13s}  {'ET_F1':>13s}")
    summary_lines = ["method,n_folds,accuracy_mean,accuracy_sd,"
                      "N_f1_mean,N_f1_sd,PD_f1_mean,PD_f1_sd,ET_f1_mean,ET_f1_sd"]
    for method in args.tfd_methods:
        ok = [r for r in all_results[method] if "accuracy" in r]
        if not ok:
            print(f"{method:>15s}  (no successful folds)")
            continue
        acc = np.array([r["accuracy"] for r in ok])
        f1 = {c: np.array([r["per_class"][c]["f1"] for r in ok]) for c in CLASS_NAMES}
        print(f"{method:>15s}  {acc.mean():.3f} ± {acc.std():.3f}  "
              f"{f1['N'].mean():.2f} ± {f1['N'].std():.2f}   "
              f"{f1['PD'].mean():.2f} ± {f1['PD'].std():.2f}   "
              f"{f1['ET'].mean():.2f} ± {f1['ET'].std():.2f}")
        summary_lines.append(
            f"{method},{len(ok)},{acc.mean():.4f},{acc.std():.4f},"
            f"{f1['N'].mean():.4f},{f1['N'].std():.4f},"
            f"{f1['PD'].mean():.4f},{f1['PD'].std():.4f},"
            f"{f1['ET'].mean():.4f},{f1['ET'].std():.4f}"
        )
    (args.output / "summary.csv").write_text("\n".join(summary_lines) + "\n")

    # Pooled report: concatenate every fold's test predictions and score once.
    # This is the headline number for a small minority class (ET), where
    # per-fold F1 swings wildly on 1-2 test samples but the pooled AUC is stable.
    if any(pooled_logits[m] for m in args.tfd_methods):
        print("\n" + "=" * 72)
        print("POOLED  (concatenate per-fold test predictions, then score once)")
        print("=" * 72)
        print(f"{'method':>15s}  {'acc':>6s}  {'N_F1':>6s}  {'PD_F1':>6s}  "
              f"{'ET_F1':>6s}  {'N_AUC':>6s}  {'PD_AUC':>6s}  {'ET_AUC':>6s}")
        for method in args.tfd_methods:
            if not pooled_logits[method]:
                continue
            logits_cat = np.concatenate(pooled_logits[method], axis=0)
            y_cat = np.concatenate(pooled_y[method], axis=0)
            rep = classification_report(logits_cat, y_cat, CLASS_NAMES)
            pc = rep["per_class"]
            print(f"{method:>15s}  {rep['accuracy']:>6.3f}  "
                  f"{pc['N']['f1']:>6.2f}  {pc['PD']['f1']:>6.2f}  "
                  f"{pc['ET']['f1']:>6.2f}  {pc['N']['auc']:>6.2f}  "
                  f"{pc['PD']['auc']:>6.2f}  {pc['ET']['auc']:>6.2f}")
            (args.output / method / "pooled_report.json").write_text(
                json.dumps(rep, indent=2)
            )
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
