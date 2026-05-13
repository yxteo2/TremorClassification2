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
from tremor.model import TremorBiLSTM
from tremor.preprocessing import apply_stft, center_pad, random_pad
from tremor.splits import subject_level_split


ACTION_LENGTH_LIMITS: dict[str, tuple[int, int]] = {
    "DRINK": (1000, 5000),
    "EAT": (1, 8000),
    "FNF": (1, 5000),
    "OUT": (500, 1500),
    "REST": (500, 1500),
}


class TremorDataset(Dataset):
    def __init__(
        self,
        recs: list[Recording],
        target_length: int,
        fs: float,
        nperseg: int,
        rng_seed: int,
        oversample_to: int | None = None,
        augment: bool = False,
    ) -> None:
        self.target_length = target_length
        self.fs = fs
        self.nperseg = nperseg
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
        x = apply_stft(x, fs=self.fs, nperseg=self.nperseg)
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
    p.add_argument("--feature", default="downsize_filtered_amplitudes")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument(
        "--oversample-to",
        type=int,
        default=None,
        help="Per-class oversampling target for the TRAINING fold only.",
    )
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    min_len, max_len = ACTION_LENGTH_LIMITS.get(args.action, (1, 10**9))
    recs = load_recordings(args.data_root, feature=args.feature, action=args.action)
    recs = filter_by_length(recs, min_len=min_len, max_len=max_len)
    if not recs:
        raise SystemExit("No recordings loaded.")

    subjects = [r.subject for r in recs]
    labels = [r.y for r in recs]
    split = subject_level_split(
        subjects, labels, test_size=0.15, val_size=0.15, seed=args.seed
    )

    target_length = int(max(r.x.shape[1] for r in recs) * 1.1)

    train_recs = [recs[i] for i in split.train_idx]
    val_recs = [recs[i] for i in split.val_idx]
    test_recs = [recs[i] for i in split.test_idx]

    print(
        f"Loaded {len(recs)} recordings, "
        f"{len(set(subjects))} subjects. "
        f"Train={len(train_recs)} / Val={len(val_recs)} / Test={len(test_recs)}."
    )
    print(f"Class distribution (train): {Counter(r.y for r in train_recs)}")
    print(f"Class distribution (val):   {Counter(r.y for r in val_recs)}")
    print(f"Class distribution (test):  {Counter(r.y for r in test_recs)}")

    if args.oversample_to is None:
        per_class = Counter(r.y for r in train_recs)
        args.oversample_to = max(per_class.values()) * 3

    train_ds = TremorDataset(
        train_recs,
        target_length=target_length,
        fs=60.0,
        nperseg=64,
        rng_seed=args.seed,
        oversample_to=args.oversample_to,
        augment=True,
    )
    val_ds = TremorDataset(
        val_recs,
        target_length=target_length,
        fs=60.0,
        nperseg=64,
        rng_seed=args.seed + 1,
        augment=False,
    )
    test_ds = TremorDataset(
        test_recs,
        target_length=target_length,
        fs=60.0,
        nperseg=64,
        rng_seed=args.seed + 2,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    sample_x, _ = train_ds[0]
    input_size = sample_x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TremorBiLSTM(
        input_size=input_size,
        num_classes=len(CLASS_NAMES),
        hidden=args.hidden,
    ).to(device)
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
