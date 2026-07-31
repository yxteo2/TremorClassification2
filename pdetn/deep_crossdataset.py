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


def _ds(recs, target_length, augment, f_max=15.0):
    return TremorDataset(
        recs, target_length=target_length, fs=100.0, f_max=f_max,
        tfd_method="stft", nperseg=256, nfft=256, noverlap=192,
        normalize="per_recording", augment=augment, oversample_to=None,
        length_mode="truncate",
    )


def train_bilstm(train_recs, val_recs, num_classes, target_length,
                 epochs=60, patience=12, focal_gamma=1.5, device=DEVICE):
    tr, vl = _ds(train_recs, target_length, True), _ds(val_recs, target_length, False)
    # drop_last avoids a size-1 final batch, which breaks BatchNorm in train mode
    tl = DataLoader(tr, batch_size=16, shuffle=True, drop_last=len(tr) > 16)
    vloader = DataLoader(vl, batch_size=16)
    sx, _ = tr[0]
    model = build_model("tremor_bilstm", input_size=sx.shape[0],
                        num_classes=num_classes, target_T=sx.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
def predict_logits(model, recs, target_length, device=DEVICE):
    dl = DataLoader(_ds(recs, target_length, False), batch_size=16)
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
