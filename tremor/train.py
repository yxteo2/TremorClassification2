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
    load_recordings,
)
from tremor.evaluate import classification_report
from tremor.models import MODELS, build_model
from tremor.preprocessing import apply_stft, bandpass, center_pad, random_pad
from tremor.quaternion_data import load_quaternion_recordings
from tremor.spectral import (
    crop_freq_bins,
    freq_bins_for_fmax,
    log_compress,
    per_freq_zscore,
    per_recording_zscore,
    spec_augment,
    time_pad,
)
from tremor.losses import build_loss_fn
from tremor.tfd import apply_cwt, apply_hht
from tremor.splits import subject_level_split
from tremor.stft_data import STFTRecording, load_stft_recordings


def _oversample_per_class(recs, oversample_to, rng):
    """Oversample each class to ``oversample_to`` items.

    Returns the original list unchanged when ``oversample_to`` is None.
    Empty classes are skipped (they cannot be sampled from) instead of
    raising — set up that case with a clear message at the call site.
    """
    if oversample_to is None or oversample_to <= 0:
        return list(recs)
    by_class: dict[int, list] = {}
    for r in recs:
        by_class.setdefault(r.y, []).append(r)
    balanced = []
    for cls, items in by_class.items():
        if not items:
            continue
        idx = rng.integers(0, len(items), size=oversample_to)
        balanced.extend(items[i] for i in idx)
    return balanced


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

        self.recs = _oversample_per_class(recs, oversample_to, self.rng)

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
        tfd_method: str = "stft",
        cwt_w0: float = 6.0,
        cwt_decim: int = 8,
        cwt_freq_step: float = 0.5,
        hht_max_imfs: int = 8,
        log_compress_on: bool = True,
        normalize: str = "per_recording",
        spec_augment_on: bool = False,
    ) -> None:
        self.target_length = target_length
        self.fs = fs
        self.nperseg = nperseg
        self.nfft = nfft
        self.noverlap = noverlap
        self.f_max = f_max
        self.augment = augment
        self.tfd_method = tfd_method
        self.cwt_w0 = cwt_w0
        self.cwt_decim = cwt_decim
        self.cwt_freq_step = cwt_freq_step
        self.hht_max_imfs = hht_max_imfs
        self.log_compress_on = log_compress_on
        self.normalize = normalize
        self.spec_augment_on = spec_augment_on
        self.rng = np.random.default_rng(rng_seed)

        self.recs = _oversample_per_class(recs, oversample_to, self.rng)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        r = self.recs[i]
        if self.augment:
            x = random_pad(r.x, self.target_length, self.rng)
        else:
            x = center_pad(r.x, self.target_length)
        if self.tfd_method == "cwt":
            f_high = self.f_max if self.f_max else 15.0
            freqs = np.arange(3.0, f_high + 1e-9, self.cwt_freq_step)
            x = apply_cwt(
                x, fs=self.fs, freqs=freqs, w0=self.cwt_w0, decim=self.cwt_decim,
                f_max=self.f_max,
            )
        elif self.tfd_method == "hht":
            f_high = self.f_max if self.f_max else 15.0
            freqs = np.arange(3.0, f_high + 1e-9, self.cwt_freq_step)
            x = apply_hht(
                x, fs=self.fs, freqs=freqs, max_imfs=self.hht_max_imfs,
                decim=self.cwt_decim, f_max=self.f_max,
            )
        else:
            x = apply_stft(
                x, fs=self.fs, nperseg=self.nperseg, nfft=self.nfft,
                noverlap=self.noverlap, f_max=self.f_max,
            )
        if self.log_compress_on:
            x = log_compress(x)
        if self.normalize == "per_recording":
            x = per_recording_zscore(x)
        elif self.normalize == "per_freq":
            x = per_freq_zscore(x)
        if self.augment and self.spec_augment_on:
            x = spec_augment(x, self.rng)
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
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
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


def _load_recordings_for_mode(args):
    """Dispatch to the right loader based on ``--data-mode``.

    Raises FileNotFoundError when the expected folder isn't present.
    Returns a list of Recording / STFTRecording objects.
    """
    if args.data_mode == "raw":
        recs = load_recordings(
            args.data_root, feature=args.feature, action=args.action,
        )
        if not recs:
            raise FileNotFoundError(
                f"No recordings under {args.feature}/{args.action}."
            )
        return recs

    if args.data_mode == "quaternion":
        feature = args.feature if args.feature != "stft" else "raw_quaternion"
        recs = load_quaternion_recordings(
            args.data_root, action=args.action, fs=args.fs,
            mode=args.quaternion_mode, convention=args.quaternion_convention,
            feature=feature,
        )
        if not recs:
            raise FileNotFoundError(
                f"No quaternion files under {feature}/{args.action}."
            )
        return recs

    recs = load_stft_recordings(
        args.data_root, action=args.action, feature=args.feature,
    )
    if not recs:
        raise FileNotFoundError(
            f"No STFT files under {args.feature}/{args.action}."
        )
    return recs


def _seed_everything(seed: int) -> None:
    """Seed every RNG the training run touches.

    - ``torch.manual_seed``     -> model init, dropout, BatchNorm running stats
    - ``torch.cuda.manual_seed_all`` -> all GPU RNGs (no-op on CPU)
    - ``np.random.seed``        -> legacy numpy global RNG
    - ``random.seed``           -> Python stdlib RNG (defensive; not used here)

    The per-dataset ``np.random.default_rng`` instances (oversampling,
    random padding, SpecAugment) are seeded separately with
    ``args.seed`` / ``+1`` / ``+2`` for train / val / test so that
    augmentation order is also reproducible.
    """
    import random as _stdrandom
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _stdrandom.seed(seed)
    print(f"[seed] all RNGs seeded with {seed}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--action", default="DRINK")
    p.add_argument("--data-mode",
                   choices=("raw", "stft", "quaternion"), default="stft",
                   help="'raw': time-domain amplitudes from "
                        "ProcessedData/<feature>/<ACTION>/, STFT applied here. "
                        "'stft' (default): precomputed STFT magnitude CSVs "
                        "from ProcessedData/<feature>/<ACTION>/<CLASS>/. "
                        "'quaternion': raw IMU quaternion CSVs from "
                        "ProcessedData/<feature>/<ACTION>/<CLASS>/, "
                        "converted to angular velocity then STFT.")
    p.add_argument("--quaternion-mode",
                   choices=("angular_velocity", "components"),
                   default="angular_velocity",
                   help="(quaternion mode) 'angular_velocity' (default) "
                        "computes omega = 2*(dq/dt)*q^-1 per sensor -> "
                        "9 channels in rad/s. 'components' keeps the raw 12 "
                        "quaternion columns and lets the bandpass strip the "
                        "baseline orientation.")
    p.add_argument("--quaternion-convention",
                   choices=("xyzw", "wxyz"), default="xyzw",
                   help="(quaternion mode) Component order in each 4-tuple. "
                        "Drive's raw_quaternion files use 'xyzw' (scalar last).")
    p.add_argument("--feature", default="stft",
                   help="ProcessedData subfolder. For raw mode use e.g. "
                        "'filtered_amplitudes'; for stft mode use 'stft'; "
                        "for quaternion mode use 'raw_quaternion'.")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--model", choices=sorted(MODELS), default="tremor_bilstm",
                   help="Network architecture. Options: " + ", ".join(sorted(MODELS)))
    p.add_argument("--hidden", type=int, default=300,
                   help="Hidden size: LSTM/GRU units for the recurrent models, "
                        "Conv1D base_filters for restcn. Used as-is — no floor "
                        "or scaling applied.")
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--oversample-to",
        type=int,
        default=None,
        help="Per-class oversampling target for the TRAINING fold only. "
             "Pass 0 or omit to disable. Pass a positive integer (e.g. 200) "
             "to draw that many samples per class with replacement.",
    )
    p.add_argument(
        "--auto-oversample",
        action="store_true",
        help="Override --oversample-to with 3 x the largest class count "
             "(quick way to get a roughly balanced training set).",
    )
    p.add_argument("--fs", type=float, default=100.0,
                   help="Sampling rate in Hz of the input recordings (default 100, no downsampling).")
    p.add_argument("--nperseg", type=int, default=128,
                   help="STFT window length in samples (MATLAB default 128).")
    p.add_argument("--nfft", type=int, default=256,
                   help="STFT FFT length in samples (MATLAB default 256).")
    p.add_argument("--noverlap", type=int, default=96,
                   help="STFT overlap in samples (MATLAB default 96, i.e. 75%%).")
    p.add_argument("--f-max", type=float, default=None,
                   help="Drop STFT bins above this frequency (Hz). "
                        "Typical tremor cutoff: 15 Hz.")
    p.add_argument("--apply-bandpass", action="store_true",
                   help="Apply a 3-30 Hz zero-phase bandpass to each recording "
                        "before STFT (raw mode only).")
    p.add_argument("--tfd-method", choices=("stft", "cwt", "hht"), default="stft",
                   help="Time-frequency decomposition: "
                        "'stft' (default; cheap linear-frequency spectrogram), "
                        "'cwt' (Morlet wavelet, frequency-adaptive window — "
                        "better for short non-stationary clips), "
                        "'hht' (Hilbert-Huang via EMD; best for genuinely "
                        "non-stationary tremor but slowest, needs EMD-signal). "
                        "Used in raw and quaternion modes only.")
    p.add_argument("--hht-max-imfs", type=int, default=8,
                   help="(hht) Number of EMD intrinsic mode functions to keep "
                        "per channel before computing instantaneous frequency.")
    p.add_argument("--cwt-w0", type=float, default=6.0,
                   help="(cwt) Morlet central angular frequency parameter; "
                        "higher = sharper freq resolution, longer wavelets.")
    p.add_argument("--cwt-decim", type=int, default=8,
                   help="(cwt) Temporal decimation factor applied to the CWT "
                        "output. With fs=100 and decim=8 the output frame "
                        "rate is 12.5 Hz, comparable to the STFT's hop=32.")
    p.add_argument("--cwt-freq-step", type=float, default=0.5,
                   help="(cwt) Spacing between analysis frequencies in Hz, "
                        "swept from 3 Hz to --f-max. Step 0.5 -> ~24 bins; "
                        "step 0.25 -> ~48 bins.")
    p.add_argument("--length-mode", choices=("pad", "truncate"), default="pad",
                   help="How to make all recordings the same length. "
                        "'pad' (default) sets target = max_length x 1.1; short "
                        "recordings get random/centred zero-padding, only the "
                        "longest get a small centre-crop. "
                        "'truncate' sets target = min_length so the model sees "
                        "a smaller, denser tensor; long recordings get randomly "
                        "truncated (train) or centre-cropped (val/test).")
    p.add_argument("--stft-fs", type=float, default=100.0,
                   help="(stft mode) Original sampling rate the STFTs were "
                        "computed at — used to map --f-max to bin count.")
    p.add_argument("--stft-n-sensors", type=int, default=3,
                   help="(stft mode) Number of sensors stacked column-wise.")
    p.add_argument("--stft-n-freq-bins", type=int, default=65,
                   help="(stft mode) Frequency bins per sensor (nfft/2+1).")
    p.add_argument("--no-log-compress", action="store_true",
                   help="Disable log1p magnitude compression of the spectrogram.")
    p.add_argument("--normalize",
                   choices=("none", "per_freq", "per_recording"),
                   default="per_recording",
                   help="Per-recording normalization applied after the TFD. "
                        "'per_recording' (default) z-scores over all bins of a "
                        "recording — preserves the spectral magnitude profile "
                        "that distinguishes PD (3-7 Hz) from ET (4-12 Hz). "
                        "'per_freq' z-scores each frequency row across time "
                        "and DESTROYS the spectral peak for quasi-stationary "
                        "tremors; only useful for non-stationary action data. "
                        "'none' relies entirely on the model's input BatchNorm.")
    p.add_argument("--spec-augment", action="store_true",
                   help="SpecAugment freq/time masking on TRAIN only.")
    p.add_argument("--loss", choices=("ce", "weighted_ce", "focal"),
                   default="focal",
                   help="Loss function. 'ce' is plain cross-entropy. "
                        "'weighted_ce' adds inverse-frequency class weights "
                        "(handles class imbalance). 'focal' (default) adds "
                        "the (1-p_t)^gamma down-weight on top of weighted_ce "
                        "to focus on hard examples.")
    p.add_argument("--focal-gamma", type=float, default=1.5,
                   help="Focusing parameter for focal loss. 0 = weighted CE; "
                        "1-2.5 are typical for moderately imbalanced 3-class "
                        "problems.")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing for the loss. 0.05-0.1 is a mild "
                        "regulariser; values >= 0.2 can hurt accuracy.")
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    _seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        recs = _load_recordings_for_mode(args)
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}") from None

    if args.data_mode == "raw" and args.apply_bandpass:
        for r in recs:
            r.x = bandpass(r.x, fs=args.fs, band=(3.0, 30.0))
    elif args.data_mode == "quaternion":
        print(
            f"[quaternion] mode={args.quaternion_mode} "
            f"convention={args.quaternion_convention} "
            f"channels={recs[0].x.shape[0]}"
        )
        if args.apply_bandpass:
            for r in recs:
                r.x = bandpass(r.x, fs=args.fs, band=(3.0, 30.0))

    lengths = [r.x.shape[1] for r in recs]
    if args.length_mode == "pad":
        target_length = int(max(lengths) * 1.1)
    else:
        target_length = int(min(lengths))
    if target_length < 2:
        raise SystemExit(
            f"target length is {target_length}, too small to train on. "
            f"Recording lengths range {min(lengths)}-{max(lengths)}; "
            f"check the data folder or switch --length-mode."
        )
    print(
        f"[length] mode={args.length_mode}  "
        f"min={min(lengths)} max={max(lengths)}  -> target={target_length}"
    )

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

    if args.auto_oversample:
        per_class = Counter(r.y for r in train_recs)
        missing = [c for c in range(len(CLASS_NAMES)) if per_class.get(c, 0) == 0]
        if missing:
            print(
                f"[oversample] WARNING: classes with zero training samples will "
                f"stay empty (cannot oversample from nothing): "
                f"{[CLASS_NAMES[i] for i in missing]}"
            )
        args.oversample_to = max(per_class.values()) * 3
        print(f"[oversample] auto target = {args.oversample_to} per class")
    elif args.oversample_to:
        print(f"[oversample] target = {args.oversample_to} per class")

    if args.data_mode in ("raw", "quaternion"):
        ds_kwargs = dict(
            fs=args.fs,
            nperseg=args.nperseg,
            nfft=args.nfft,
            noverlap=args.noverlap,
            f_max=args.f_max,
            tfd_method=args.tfd_method,
            cwt_w0=args.cwt_w0,
            cwt_decim=args.cwt_decim,
            cwt_freq_step=args.cwt_freq_step,
            hht_max_imfs=args.hht_max_imfs,
            log_compress_on=not args.no_log_compress,
            normalize=args.normalize,
            spec_augment_on=args.spec_augment,
        )
        print(f"[tfd] method={args.tfd_method}  "
              f"log={not args.no_log_compress}  normalize={args.normalize}")
        train_ds = TremorDataset(
            train_recs, target_length=target_length, rng_seed=args.seed,
            oversample_to=args.oversample_to, augment=True, **ds_kwargs,
        )
        val_ds = TremorDataset(
            val_recs, target_length=target_length, rng_seed=args.seed + 1,
            augment=False, **ds_kwargs,
        )
        test_ds = TremorDataset(
            test_recs, target_length=target_length, rng_seed=args.seed + 2,
            augment=False, **ds_kwargs,
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
    train_labels = [r.y for r in train_recs]
    loss_fn = build_loss_fn(
        loss_type=args.loss,
        train_labels=train_labels,
        num_classes=len(CLASS_NAMES),
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        device=device,
    )
    print(f"[loss] type={args.loss}  focal_gamma={args.focal_gamma}  "
          f"label_smoothing={args.label_smoothing}")
    if args.loss in ("weighted_ce", "focal"):
        from tremor.losses import compute_class_weights
        weights = compute_class_weights(train_labels, len(CLASS_NAMES))
        counts = Counter(train_labels)
        print("[loss] class weights (inverse-frequency, mean=1):")
        for i, name in enumerate(CLASS_NAMES):
            print(f"       {name:>3s}  count={counts.get(i, 0):4d}  "
                  f"weight={float(weights[i]):.3f}")
        if args.auto_oversample or args.oversample_to:
            print(
                "[loss] WARNING: oversampling is active AND class weights are "
                "applied. The training set sees a balanced label distribution "
                "but the loss still uses inverse-frequency weights from the "
                "original distribution — minority classes get double-corrected. "
                "Pick ONE of: '--auto-oversample / --oversample-to' or "
                "'--loss focal / weighted_ce'."
            )

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

    if best_state is None:
        raise SystemExit(
            "Training produced no validation improvement (e.g. --epochs 0). "
            "Increase --epochs and try again."
        )
    model.load_state_dict(best_state)
    torch.save(best_state, args.output / "model.pt")

    test_logits, test_y = _collect(model, test_loader, device)
    report = classification_report(test_logits, test_y, CLASS_NAMES)
    print(json.dumps(report, indent=2))
    with open(args.output / "test_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
