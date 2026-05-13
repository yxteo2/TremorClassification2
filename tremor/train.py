"""End-to-end training with subject-level splits.

Key correctness guarantees (the bugs in the MATLAB version are fixed):
    1. The train / val / test split is at the SUBJECT level.
    2. Oversampling and random-pad augmentation are applied to the
       TRAINING fold only — never to validation or test.
    3. The "best" model is selected by VALIDATION loss only.
    4. The test set is evaluated exactly once at the end of training.

Run:
    python -m tremor.train --data-root /path/to/repo --action DRINK
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tremor.data import (
    CLASS_NAMES,
    Recording,
    filter_by_length,
    load_recordings,
)
from tremor.evaluate import classification_report
from tremor.models import MODELS, build_model
from tremor.preprocessing import apply_stft, bandpass, center_pad, random_pad
from tremor.spectral import (
    crop_freq_bins,
    freq_bins_for_fmax,
    log_compress,
    per_freq_zscore,
    per_recording_zscore,
    spec_augment,
    time_pad,
)
from tremor.splits import subject_level_split
from tremor.stft_data import STFTRecording, load_stft_recordings


ACTION_LENGTH_LIMITS: dict[str, tuple[int, int]] = {
    "DRINK": (1000, 5000),
    "EAT": (1, 8000),
    "FNF": (1, 5000),
    "OUT": (500, 1500),
    "REST": (500, 1500),
}


class STFTDataset(Dataset):
    """Dataset over precomputed STFT magnitude matrices.

    Per-item pipeline:
        1. (optional) crop the lowest freq bins per sensor (``--f-max``).
        2. (optional) ``log1p(S/eps)`` to compress dynamic range.
        3. (optional) per-frequency z-score across time within recording.
        4. random/centred zero-pad in TIME to ``target_T`` frames.
        5. (training only, optional) SpecAugment.
    """

    def __init__(
        self,
        recs: list[STFTRecording],
        target_T: int,
        rng_seed: int,
        n_sensors: int,
        n_freq_bins: int,
        keep_bins: int,
        log_compress_on: bool,
        normalize: str,  # 'none' | 'per_freq' | 'per_recording'
        spec_augment_on: bool,
        oversample_to: int | None = None,
        augment: bool = False,
    ) -> None:
        self.target_T = target_T
        self.n_sensors = n_sensors
        self.n_freq_bins = n_freq_bins
        self.keep_bins = keep_bins
        self.log_compress_on = log_compress_on
        self.normalize = normalize
        self.spec_augment_on = spec_augment_on
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

        if oversample_to is None:
            self.recs = list(recs)
        else:
            by_class: dict[int, list[STFTRecording]] = {}
            for r in recs:
                by_class.setdefault(r.y, []).append(r)
            balanced: list[STFTRecording] = []
            for items in by_class.values():
                idx = self.rng.integers(0, len(items), size=oversample_to)
                balanced.extend(items[i] for i in idx)
            self.recs = balanced

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.recs[i]
        x = r.x
        if self.keep_bins < self.n_freq_bins:
            x = crop_freq_bins(x, self.n_sensors, self.n_freq_bins, self.keep_bins)
        if self.log_compress_on:
            x = log_compress(x)
        if self.normalize == "per_freq":
            x = per_freq_zscore(x)
        elif self.normalize == "per_recording":
            x = per_recording_zscore(x)
        x = time_pad(x, self.target_T, rng=self.rng if self.augment else None)
        if self.augment and self.spec_augment_on:
            x = spec_augment(x, self.rng)
        return torch.from_numpy(x), r.y


class TremorDataset(Dataset):
    def __init__(
        self,
        recs: list[Recording],
        target_length: int,
        fs: float,
        nperseg: int,
        nfft: int,
        noverlap: int,
        rng_seed: int,
        f_max: float | None = None,
        oversample_to: int | None = None,
        augment: bool = False,
    ) -> None:
        self.target_length = target_length
        self.fs = fs
        self.nperseg = nperseg
        self.nfft = nfft
        self.noverlap = noverlap
        self.f_max = f_max
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

        if oversample_to is None:
            self.recs = list(recs)
        else:
            by_class: dict[int, list[Recording]] = {}
            for r in recs:
                by_class.setdefault(r.y, []).append(r)
            balanced: list[Recording] = []
            for items in by_class.values():
                idx = self.rng.integers(0, len(items), size=oversample_to)
                balanced.extend(items[i] for i in idx)
            self.recs = balanced

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.recs[i]
        if self.augment:
            x = random_pad(r.x, self.target_length, self.rng)
        else:
            x = center_pad(r.x, self.target_length)
        x = apply_stft(
            x, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft,
            noverlap=self.noverlap, f_max=self.f_max,
        )
        return torch.from_numpy(x), r.y


def _train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss) * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(loss_fn(logits, y)) * x.size(0)
        correct += int((logits.argmax(-1) == y).sum())
        n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    logits_list, y_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits_list.append(model(x).cpu().numpy())
        y_list.append(y.numpy())
    return np.concatenate(logits_list), np.concatenate(y_list)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--action", default="DRINK")
    p.add_argument("--data-mode", choices=("raw", "stft"), default="stft",
                   help="'raw': load time-domain amplitudes and apply STFT here. "
                        "'stft' (default): load precomputed STFT magnitude CSVs "
                        "from ProcessedData/<feature>/<ACTION>/<CLASS>/.")
    p.add_argument("--feature", default="stft",
                   help="ProcessedData subfolder. For raw mode use e.g. "
                        "'filtered_amplitudes'; for stft mode use 'stft'.")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--model", choices=sorted(MODELS), default="tremor_bilstm",
                   help="Network architecture. Options: " + ", ".join(sorted(MODELS)))
    p.add_argument("--hidden", type=int, default=300,
                   help="LSTM/GRU hidden size (or scaled base_filters for restcn).")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--oversample-to",
        type=int,
        default=None,
        help="Per-class oversampling target for the TRAINING fold only.",
    )
    p.add_argument("--fs", type=float, default=100.0,
                   help="Sampling rate in Hz of the input recordings (default 100, no downsampling).")
    p.add_argument("--nperseg", type=int, default=128,
                   help="STFT window length in samples (MATLAB default 128).")
    p.add_argument("--nfft", type=int, default=256,
                   help="STFT FFT length in samples (MATLAB default 256).")
    p.add_argument("--noverlap", type=int, default=96,
                   help="STFT overlap in samples (MATLAB default 96, i.e. 75%).")
    p.add_argument("--f-max", type=float, default=None,
                   help="Drop STFT bins above this frequency (Hz). "
                        "Typical tremor cutoff: 15 Hz.")
    p.add_argument("--apply-bandpass", action="store_true",
                   help="Apply a 3-30 Hz zero-phase bandpass to each recording "
                        "before STFT (raw mode only).")
    p.add_argument("--stft-fs", type=float, default=100.0,
                   help="(stft mode) Original sampling rate the STFTs were "
                        "computed at — used to map --f-max to bin count.")
    p.add_argument("--stft-n-sensors", type=int, default=3,
                   help="(stft mode) Number of sensors stacked column-wise.")
    p.add_argument("--stft-n-freq-bins", type=int, default=65,
                   help="(stft mode) Frequency bins per sensor (nfft/2+1).")
    p.add_argument("--no-log-compress", action="store_true",
                   help="(stft mode) Disable log1p magnitude compression.")
    p.add_argument("--normalize",
                   choices=("none", "per_freq", "per_recording"),
                   default="per_freq",
                   help="(stft mode) Per-recording normalization. "
                        "'per_freq' (default) z-scores each frequency row "
                        "across time — the most reliable choice.")
    p.add_argument("--spec-augment", action="store_true",
                   help="(stft mode) SpecAugment freq/time masking on TRAIN only.")
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.data_mode == "raw":
        min_len, max_len = ACTION_LENGTH_LIMITS.get(args.action, (1, 10**9))
        recs = load_recordings(
            args.data_root, feature=args.feature, action=args.action
        )
        recs = filter_by_length(recs, min_len=min_len, max_len=max_len)
        if not recs:
            raise SystemExit("No recordings loaded.")
        if args.apply_bandpass:
            for r in recs:
                r.x = bandpass(r.x, fs=args.fs, band=(3.0, 30.0))
        target_length = int(max(r.x.shape[1] for r in recs) * 1.1)
    else:
        recs = load_stft_recordings(
            args.data_root, action=args.action, feature=args.feature
        )
        if not recs:
            raise SystemExit(f"No STFT files under {args.feature}/{args.action}.")
        # target_length is in TIME FRAMES here, not samples.
        target_length = int(max(r.x.shape[1] for r in recs) * 1.1)

    subjects = [r.subject for r in recs]
    labels = [r.y for r in recs]
    split = subject_level_split(
        subjects, labels, test_size=0.15, val_size=0.15, seed=args.seed
    )
    train_recs = [recs[i] for i in split.train_idx]
    val_recs = [recs[i] for i in split.val_idx]
    test_recs = [recs[i] for i in split.test_idx]

    print(
        f"[{args.data_mode}] Loaded {len(recs)} recordings, "
        f"{len(set(subjects))} subjects. "
        f"Train={len(train_recs)} / Val={len(val_recs)} / Test={len(test_recs)}."
    )
    print(f"Class distribution (train): {Counter(r.y for r in train_recs)}")
    print(f"Class distribution (val):   {Counter(r.y for r in val_recs)}")
    print(f"Class distribution (test):  {Counter(r.y for r in test_recs)}")

    if args.oversample_to is None:
        per_class = Counter(r.y for r in train_recs)
        args.oversample_to = max(per_class.values()) * 3

    if args.data_mode == "raw":
        stft_kwargs = dict(
            fs=args.fs,
            nperseg=args.nperseg,
            nfft=args.nfft,
            noverlap=args.noverlap,
            f_max=args.f_max,
        )
        train_ds = TremorDataset(
            train_recs, target_length=target_length, rng_seed=args.seed,
            oversample_to=args.oversample_to, augment=True, **stft_kwargs,
        )
        val_ds = TremorDataset(
            val_recs, target_length=target_length, rng_seed=args.seed + 1,
            augment=False, **stft_kwargs,
        )
        test_ds = TremorDataset(
            test_recs, target_length=target_length, rng_seed=args.seed + 2,
            augment=False, **stft_kwargs,
        )
    else:
        keep_bins = freq_bins_for_fmax(
            args.stft_fs, args.stft_n_freq_bins, args.f_max
        )
        print(
            f"[stft] sensors={args.stft_n_sensors} bins/sensor={args.stft_n_freq_bins} "
            f"keep_bins={keep_bins} (f_max={args.f_max} @ fs={args.stft_fs}) "
            f"target_T={target_length}"
        )
        stft_ds_kwargs = dict(
            target_T=target_length,
            n_sensors=args.stft_n_sensors,
            n_freq_bins=args.stft_n_freq_bins,
            keep_bins=keep_bins,
            log_compress_on=not args.no_log_compress,
            normalize=args.normalize,
            spec_augment_on=args.spec_augment,
        )
        train_ds = STFTDataset(
            train_recs, rng_seed=args.seed,
            oversample_to=args.oversample_to, augment=True, **stft_ds_kwargs,
        )
        val_ds = STFTDataset(
            val_recs, rng_seed=args.seed + 1, augment=False, **stft_ds_kwargs,
        )
        test_ds = STFTDataset(
            test_recs, rng_seed=args.seed + 2, augment=False, **stft_ds_kwargs,
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    input_size = sample_x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        name=args.model,
        input_size=input_size,
        num_classes=len(CLASS_NAMES),
        target_T=target_length,
        hidden=args.hidden,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model} | trainable params: {n_params:,}")
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.8)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict | None = None
    epochs_since_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"{'  *' if improved else ''}"
        )
        if epochs_since_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, args.output / "model.pt")

    test_logits, test_y = _collect(model, test_loader, device)
    report = classification_report(test_logits, test_y, CLASS_NAMES)
    print(json.dumps(report, indent=2))
    with open(args.output / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
