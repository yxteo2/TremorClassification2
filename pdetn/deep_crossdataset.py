"""Deep BiLSTM cross-dataset models for PD/N/ET, framed on the PD-vs-ET axis.

Two model variants to compare (run on your machine — needs torch + PADS):
  * 3class : one 3-class BiLSTM; report PD-vs-ET sub-accuracy.
  * two_stage : deep N-vs-tremor BiLSTM, then a DEDICATED deep PD-vs-ET BiLSTM
                (trains only on PD+ET, so the hard axis gets full capacity),
                with an ET threshold tuned on a validation split.

Single-sensor (local lower_arm ~ PADS wrist), STFT spectrograms. Reuses the
repo's TremorDataset / build_model / focal loss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from tremor.data import CLASS_NAMES, Recording
from tremor.datasets import TremorDataset
from tremor.evaluate import classification_report, softmax
from tremor.losses import build_loss_fn
from tremor.models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def remap(recs, mapping):
    """New recs with y remapped; drop where mapping[y] is None or missing."""
    out = []
    for r in recs:
        ny = mapping.get(r.y)
        if ny is not None:
            out.append(Recording(x=r.x, y=ny, subject=r.subject, path=r.path,
                                 condition=r.condition))
    return out


def _ds(recs, target_length, augment, f_max=15.0, tfd_method="stft",
        nperseg=256, noverlap=192, oversample_to=None, spec_augment=False):
    """Dataset for the TF models.

    ``oversample_to`` balances classes at the DATA level (train split only --
    never pass it for validation or test). With 16 ET against 75 PD, focal loss
    alone leaves the minority so rare that the network collapses to predicting
    the majority: measured, 2 of 3 seeds predicted zero ET patients.
    """
    return TremorDataset(
        recs, target_length=target_length, fs=100.0, f_max=f_max,
        tfd_method=tfd_method, nperseg=nperseg, nfft=nperseg, noverlap=noverlap,
        normalize="per_recording", augment=augment, oversample_to=oversample_to,
        spec_augment_on=spec_augment, length_mode="truncate",
    )


def train_bilstm(train_recs, val_recs, num_classes, target_length,
                 epochs=60, patience=12, focal_gamma=1.5, device=DEVICE,
                 lr=1e-3, weight_decay=1e-4, hidden=128, dropout=0.4,
                 tfd_method="stft", nperseg=256, noverlap=192, seed=0,
                 init_state=None, arch="tremor_bilstm", oversample_to=None,
                 spec_augment=False, pretrained=True, freeze_backbone=True,
                 resize_to=96, batch_size=16, augment=True):
    """Train a model on TF images. THE canonical training loop for this repo.

    ``arch`` is any name in ``tremor.models.MODELS`` (tremor_bilstm, restcn,
    resnet18, ...), so this serves the architecture comparison as well as the
    two-stage work -- ``tfbench.deep`` used to carry a second, independent copy
    of this loop.

    ``init_state`` warm-starts from a state dict, to FINE-TUNE a pretrained
    stage rather than train from scratch.
    """
    torch.manual_seed(seed)
    tfd = dict(tfd_method=tfd_method, nperseg=nperseg, noverlap=noverlap)
    # oversampling and spec-augment apply to TRAIN only
    # augment=True gives a RANDOM crop window each epoch (TremorDataset
    # _fit_mode -> "random" when length_mode="truncate"). This was hardcoded
    # True, so every deep result was augmented with no ablation available.
    tr = _ds(train_recs, target_length, augment, oversample_to=oversample_to,
             spec_augment=spec_augment, **tfd)
    vl = _ds(val_recs, target_length, False, **tfd)
    # drop_last avoids a size-1 final batch, which breaks BatchNorm in train mode
    # batch_size matters more than usual here. With ~18 % ET, a batch of 16
    # holds ~3 ET and often 0-1, which makes BatchNorm statistics and the
    # gradient direction very noisy for the minority class. Larger batches give
    # more reliable minority representation per step.
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True,
                    drop_last=len(tr) > batch_size)
    vloader = DataLoader(vl, batch_size=max(batch_size, 16))
    sx, _ = tr[0]
    # freeze_backbone/pretrained only affect the transfer models (ast, resnet18).
    # With freeze_backbone=True the pretrained feature extractor is frozen and
    # only the channel adapter + classification head train -- the low-parameter
    # regime, which is the only one that has any chance at 16 ET subjects.
    model = build_model(arch, input_size=sx.shape[0], num_classes=num_classes,
                        target_T=sx.shape[1], hidden=hidden, dropout=dropout,
                        n_input_channels=sx.shape[0], pretrained=pretrained,
                        freeze_backbone=freeze_backbone,
                        resize_to=resize_to).to(device)
    n_all = sum(q.numel() for q in model.parameters())
    n_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
    if arch in ("ast", "resnet18"):
        print(f"      [{arch}] trainable {n_tr:,} / {n_all:,} params "
              f"({100*n_tr/max(n_all,1):.1f}%)", flush=True)
    if init_state is not None:
        # load everything whose shape matches (the head may differ in width)
        cur = model.state_dict()
        keep = {k: v for k, v in init_state.items()
                if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = build_loss_fn("focal", train_labels=[r.y for r in train_recs],
                            num_classes=num_classes, focal_gamma=focal_gamma,
                            device=device)
    best, best_state, pat = math.inf, None, 0
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); v = n = 0
        with torch.no_grad():
            for x, y in vloader:
                v += float(loss_fn(model(x.to(device)), y.to(device))) * len(y)
                n += len(y)
        v /= max(n, 1)
        if v < best:
            best, pat = v, 0
            best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
        else:
            pat += 1
        if pat >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_logits(model, recs, target_length, device=DEVICE,
                   tfd_method="stft", nperseg=256, noverlap=192):
    """Logits for ``recs``.

    The TFD parameters MUST match those the model was trained with. This used to
    hardcode the defaults, so a model trained on CWT was silently evaluated on
    STFT images -- wrong input, no error.
    """
    dl = DataLoader(_ds(recs, target_length, False, tfd_method=tfd_method,
                        nperseg=nperseg, noverlap=noverlap), batch_size=16)
    model.eval()
    return np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in dl])


# --------------------------------------------------------------------------- #
# Variant 5b: single 3-class deep model
# --------------------------------------------------------------------------- #
def train_3class(train_recs, val_recs, target_length, **kw):
    return train_bilstm(train_recs, val_recs, 3, target_length, **kw)


def predict_3class(model, recs, target_length):
    return softmax(predict_logits(model, recs, target_length)).argmax(1)


# --------------------------------------------------------------------------- #
# Variant 5a: two-stage deep (N-vs-tremor, then dedicated PD-vs-ET)
# --------------------------------------------------------------------------- #
class DeepTwoStage:
    def __init__(self, target_length, tune_et=False, **kw):
        # tune_et defaults OFF: on tiny val ET sets (~2-3 subjects) the tuned
        # threshold destabilises the deep PD-vs-ET model (over-calls ET). The
        # local test showed 3-class beats two-stage for the deep model anyway.
        self.tl = target_length; self.tune = tune_et; self.kw = kw; self.thr = 0.5

    def fit(self, train_recs, val_recs):
        s1_tr = remap(train_recs, {0: 0, 1: 1, 2: 1})   # N vs tremor
        s1_vl = remap(val_recs, {0: 0, 1: 1, 2: 1})
        self.s1 = train_bilstm(s1_tr, s1_vl, 2, self.tl, **self.kw)
        s2_tr = remap(train_recs, {1: 0, 2: 1})         # PD vs ET (drop N)
        s2_vl = remap(val_recs, {1: 0, 2: 1})
        self.s2 = train_bilstm(s2_tr, s2_vl, 2, self.tl, **self.kw)
        if self.tune and s2_vl:
            p = softmax(predict_logits(self.s2, s2_vl, self.tl))[:, 1]
            yv = np.array([r.y for r in s2_vl])
            from sklearn.metrics import f1_score
            best_t, best = 0.5, -1
            for t in np.linspace(0.1, 0.9, 33):
                f = f1_score(yv, (p >= t).astype(int), zero_division=0)
                if f > best:
                    best, best_t = f, float(t)
            self.thr = best_t
        return self

    def predict(self, recs):
        is_tremor = softmax(predict_logits(self.s1, recs, self.tl)).argmax(1) == 1
        out = np.zeros(len(recs), dtype=int)   # 0 = N
        if is_tremor.any():
            idx = np.flatnonzero(is_tremor)
            sub = [recs[i] for i in idx]
            p_et = softmax(predict_logits(self.s2, sub, self.tl))[:, 1]
            out[idx] = np.where(p_et >= self.thr, 2, 1)   # ET / PD
        return out


def pd_vs_et_metrics(y_true, y_pred):
    """PD-vs-ET differential summary from 3-class predictions."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = (y_true != 0) & (y_pred != 0)          # true tremor predicted tremor
    acc = float((y_true[m] == y_pred[m]).mean()) if m.any() else float("nan")
    rep = classification_report(np.log(np.eye(3)[y_pred] + 1e-6), y_true, CLASS_NAMES)
    return {"pd_vs_et_acc": acc, "macro_f1": rep["macro_f1"],
            "per_class_f1": {c: rep["per_class"][c]["f1"] for c in CLASS_NAMES},
            "confusion": rep["confusion_matrix"]}
