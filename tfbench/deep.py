"""Stage 2 — train deep models on the best methods from stage 1 and compare
architectures (ResNet / BiLSTM / TCN / ...).

Design decisions that matter for the comparison to mean anything:

* **Patient-grouped CV.** Every patient is tested exactly once by a model that
  never saw them. Recording-level splits would leak.
* **Per-recording probabilities aggregated to patient level** before scoring, so
  deep numbers are directly comparable to the stage-1 (patient-level) table.
* **Multi-seed.** One deep run is a draw, not a result -- this project once
  reported 0.903 from a run whose 4-seed mean was 0.866. `compare()` runs
  several seeds and reports the spread.
* **Paired CI across architectures**, same as stage 1.

Architectures come from the repo registry (`tremor.models.MODELS`), so
`resnet18`, `tremor_bilstm`, `restcn`, `bilstm`, `gru`, `ast` are all available.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from tremor.datasets import TremorDataset
from tremor.evaluate import softmax
from tremor.losses import build_loss_fn
from tremor.models import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#: stage-1 method name -> the TFD the deep dataset should compute.
#: Only methods representable as a (freq, time) image can feed a 2-D model.
METHOD_TO_TFD = {
    "stft256": dict(tfd_method="stft", nperseg=256, noverlap=192),
    "stft512": dict(tfd_method="stft", nperseg=512, noverlap=384),
    "cwt": dict(tfd_method="cwt"),
    "hht": dict(tfd_method="hht"),
    "multitaper": dict(tfd_method="multitaper", nperseg=256, noverlap=192),
    "sst": dict(tfd_method="sst", nperseg=256, noverlap=192),
    "wavelet_packet": dict(tfd_method="wavelet_packet"),
    "welch": dict(tfd_method="stft", nperseg=256, noverlap=192),  # closest image
    # Stage 1's winner. TremorDataset has no IMF-selection option, so the deep
    # model sees the FULL Hilbert spectrum, IMF1 included -- i.e. it does NOT
    # reproduce hht_imf2plus exactly. Since dropping IMF1 is precisely what made
    # the method work in stage 1, read this row as "HHT-family", not as a
    # like-for-like carry-over.
    "hht_imf2plus": dict(tfd_method="hht"),
}


def _ds(recs, target_length, augment, tfd, f_max=15.0):
    kw = dict(nperseg=256, nfft=256, noverlap=192)
    kw.update({k: v for k, v in tfd.items() if k != "tfd_method"})
    kw["nfft"] = kw.get("nperseg", 256)
    return TremorDataset(recs, target_length=target_length, fs=100.0, f_max=f_max,
                         tfd_method=tfd["tfd_method"], normalize="per_recording",
                         augment=augment, oversample_to=None,
                         length_mode="truncate", **kw)


def train_one(train_recs, val_recs, n_classes, target_length, tfd, arch,
              epochs=40, patience=10, lr=1e-3, focal_gamma=1.5, seed=0,
              hidden=128, dropout=0.4, device=DEVICE):
    torch.manual_seed(seed)
    tr, vl = _ds(train_recs, target_length, True, tfd), _ds(val_recs, target_length, False, tfd)
    tl = DataLoader(tr, batch_size=16, shuffle=True, drop_last=len(tr) > 16)
    vloader = DataLoader(vl, batch_size=16)
    sx, _ = tr[0]
    model = build_model(arch, input_size=sx.shape[0], num_classes=n_classes,
                        target_T=sx.shape[1], hidden=hidden, dropout=dropout,
                        n_input_channels=sx.shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = build_loss_fn("focal", train_labels=[r.y for r in train_recs],
                            num_classes=n_classes, focal_gamma=focal_gamma,
                            device=device)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            opt.zero_grad()
            loss_fn(model(x.to(device)), y.to(device)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval(); v = n = 0
        with torch.no_grad():
            for x, y in vloader:
                v += float(loss_fn(model(x.to(device)), y.to(device))) * len(y); n += len(y)
        v /= max(n, 1)
        if v < best:
            best, bad = v, 0
            best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, recs, target_length, tfd, device=DEVICE):
    model.eval()
    dl = DataLoader(_ds(recs, target_length, False, tfd), batch_size=16)
    return softmax(np.concatenate([model(x.to(device)).cpu().numpy() for x, _ in dl]))


def run_cv(recs, method, arch, axis="PD_vs_ET", n_splits=5, seed=0,
           target_length=None, verbose=True, **train_kw):
    """Patient-grouped CV for one (method, architecture). Returns per-patient
    probabilities, labels and patient ids."""
    tfd = METHOD_TO_TFD[method]
    if axis == "PD_vs_ET":
        recs = [r for r in recs if r.y != 0]
        y_rec = np.array([1 if r.y == 2 else 0 for r in recs])
    else:
        y_rec = np.array([1 if r.y != 0 else 0 for r in recs])
    recs = [type(r)(x=r.x, y=int(v), subject=r.subject, path=r.path,
                    condition=r.condition) for r, v in zip(recs, y_rec)]
    groups = np.array([r.subject for r in recs])
    if target_length is None:
        target_length = int(np.percentile([r.x.shape[1] for r in recs], 25))

    P = np.zeros((len(recs), 2))
    for k, (tr_i, te_i) in enumerate(
            GroupKFold(n_splits=n_splits).split(np.zeros(len(recs)), y_rec, groups)):
        tr = [recs[i] for i in tr_i]
        # inner val split by patient
        tr_s = np.array(sorted({r.subject for r in tr}))
        rng = np.random.default_rng(seed + k)
        vs = set(rng.choice(tr_s, max(1, len(tr_s) // 4), replace=False))
        t2 = [r for r in tr if r.subject not in vs] or tr
        v2 = [r for r in tr if r.subject in vs] or tr
        m = train_one(t2, v2, 2, target_length, tfd, arch, seed=seed + k, **train_kw)
        P[te_i] = predict(m, [recs[i] for i in te_i], target_length, tfd)
        if verbose:
            print(f"    fold {k+1}/{n_splits} done", flush=True)

    pats = np.array(sorted(set(groups)))
    Pp = np.array([P[groups == p].mean(0) for p in pats])
    yp = np.array([y_rec[groups == p][0] for p in pats])
    return Pp, yp, pats


def bal_acc(y, pred):
    return 0.5 * (recall_score(y, pred, pos_label=1, zero_division=0)
                  + recall_score(y, pred, pos_label=0, zero_division=0))


def compare(recs, methods, archs, axis="PD_vs_ET", seeds=(0, 1), n_splits=5,
            **train_kw):
    """Grid over (method x architecture x seed). Returns a results dict."""
    res = {}
    print(f"{'method':>15}{'arch':>16}{'bal-acc mean':>14}{'sd':>7}{'seeds':>7}")
    for meth in methods:
        for a in archs:
            vals, store = [], []
            for s in seeds:
                try:
                    Pp, yp, pats = run_cv(recs, meth, a, axis=axis,
                                          n_splits=n_splits, seed=s,
                                          verbose=False, **train_kw)
                except Exception as e:
                    print(f"{meth:>15}{a:>16}  FAILED {type(e).__name__}: {str(e)[:40]}")
                    break
                vals.append(bal_acc(yp, Pp.argmax(1))); store.append((Pp, yp, pats))
            if vals:
                res[(meth, a)] = {"bal_acc": vals, "runs": store}
                print(f"{meth:>15}{a:>16}{np.mean(vals):>14.3f}{np.std(vals):>7.3f}"
                      f"{len(vals):>7}", flush=True)
    return res
