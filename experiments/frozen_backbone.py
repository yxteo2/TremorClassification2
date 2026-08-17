"""Frozen pretrained ViT as a feature extractor -- transfer learning done right.

The earlier ViT result in this project (AUC 0.540, chance) used **random**
weights, because ImageNet checkpoints cannot be downloaded in this environment.
That measured a random projection, not transfer learning, and it should not have
been allowed to stand as evidence against the approach.

This is the honest version: the backbone is frozen and only a linear head
trains, so the trainable parameter count is ~1.5 k -- squarely inside the
1e3-1e4 band where every model on this cohort has peaked. The capacity finding
(1 k > 3 k > 9 k > 35 k > 11.2 M) penalises *trainable* parameters, so a frozen
85 M-parameter backbone does not contradict it.

Needs a local checkpoint, since downloads are blocked:

    experiments/frozen_backbone.py --weights vit_fp16.pt

A tremor spectrogram is a very different image statistic from the natural
photographs ImageNet features were trained on -- mostly one bright horizontal
band on a dark field, with none of the object structure those filters detect.
The result to beat is macro precision 0.660 from a 5 k-parameter model.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.signal import stft as _stft
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, tune_offsets

SPLITS = 20
NF, NT = 64, 64
F_LO, F_HI = 3.0, 15.0


def spectrogram_table(recs, ch=slice(0, 3), fs=100.0):
    """(patients, NF, NT) log spectrograms -- ViT needs a 2-D image."""
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        x = np.atleast_2d(np.asarray(x))
        n = int(min(256, x.shape[-1]))
        f, _, Z = _stft(x, fs=fs, nperseg=n, noverlap=int(n * 0.75), axis=-1)
        P = (np.abs(Z) ** 2).mean(0)
        k = (f >= F_LO) & (f <= F_HI)
        P = P[k]
        ti = np.linspace(0, P.shape[1] - 1, NT)
        P = np.array([np.interp(ti, np.arange(P.shape[1]), row) for row in P])
        fi = np.linspace(0, P.shape[0] - 1, NF)
        P = np.array([np.interp(fi, np.arange(P.shape[0]), col) for col in P.T]).T
        P = np.log(P / (P.sum() + 1e-20) + 1e-8)
        rows[r.subject].append(P)
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[p], 0) for p in pats],
                                   dtype=np.float32)),
            np.array([lab[p] for p in pats]), np.array(pats))


def load_frozen_vit(path):
    """ViT-B/16 with ImageNet weights loaded from disk, backbone frozen."""
    from torchvision.models import vit_b_16
    m = vit_b_16(weights=None)
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    sd = {k: v.float() for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if len(missing) > 20:
        raise RuntimeError(f"checkpoint does not match vit_b_16: "
                           f"{len(missing)} missing keys, e.g. {missing[:3]}")
    print(f"  loaded checkpoint: {len(sd)} tensors, "
          f"{len(missing)} missing / {len(unexpected)} unexpected keys")
    m.heads = nn.Identity()
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


@torch.no_grad()
def vit_features(model, X, batch=16):
    """(patients, 768) frozen embeddings. Spectrogram -> 224x224, 3 channels."""
    out = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for i in range(0, len(X), batch):
        z = torch.tensor(X[i:i + batch], dtype=torch.float32).unsqueeze(1)
        # per-image min-max to [0,1]: ImageNet features expect image-like range
        zmin = z.amin(dim=(2, 3), keepdim=True)
        zmax = z.amax(dim=(2, 3), keepdim=True)
        z = (z - zmin) / (zmax - zmin + 1e-8)
        z = Fn.interpolate(z, size=(224, 224), mode="bilinear",
                           align_corners=False)
        z = z.repeat(1, 3, 1, 1)
        z = (z - mean) / std
        out.append(model(z).cpu().numpy())
    return np.concatenate(out)


def head_fit(Xtr, ytr, Xva, yva, Xout, seed=0, epochs=300, lr=1e-3, wd=1e-3,
             nc=3):
    """Linear head only -- the sole trainable part."""
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.long)
    xv, yv = T(Xva), torch.tensor(yva, dtype=torch.long)
    head = nn.Linear(Xtr.shape[1], nc)
    c = np.bincount(ytr, minlength=nc).astype(float)
    w = torch.tensor(c.sum() / (nc * np.maximum(c, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = np.inf, None
    for _ in range(epochs):
        head.train(); opt.zero_grad()
        lf(head(xt), yt).backward(); opt.step(); sch.step()
        head.eval()
        with torch.no_grad():
            v = float(lf(head(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in head.state_dict().items()}
    if state:
        head.load_state_dict(state)
    head.eval()
    with torch.no_grad():
        return [torch.softmax(head(T(z)), 1).numpy() for z in Xout]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="local ViT-B/16 ImageNet checkpoint (.pt)")
    args = ap.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit(f"checkpoint not found: {args.weights}\n"
                         "Downloads are proxy-blocked here; the file must be "
                         "present locally.")
    torch.set_num_threads(1)

    from common.cohorts import load_all
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    A, B, C = (spectrogram_table(rA, slice(3, 6)),
               spectrogram_table(rB, slice(3, 6)),
               spectrogram_table(rC, slice(0, 3)))
    rng = np.random.default_rng(0)
    keep = []
    for c in (0, 1, 2):
        i = np.flatnonzero(C[1] == c)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    Xs = np.concatenate([A[0], B[0], C[0][keep]])
    y = np.concatenate([A[1], B[1], C[1][keep]])
    coh = np.concatenate([np.full(len(A[1]), "2015"),
                          np.full(len(B[1]), "NewData"),
                          np.full(len(keep), "PADS")])
    key = np.array([f"{c}_{l}" for c, l in zip(coh, y)])
    print(f"n={len(y)}  spectrograms {Xs.shape}")

    print("extracting frozen ViT features ...")
    vit = load_frozen_vit(args.weights)
    n_tr = sum(p.numel() for p in vit.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in vit.parameters())
    print(f"  backbone {n_all/1e6:.1f} M params, {n_tr} trainable")
    F = vit_features(vit, Xs)
    print(f"  features {F.shape}   head trainable params: "
          f"{F.shape[1] * 3 + 3}\n")

    out = []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(F, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(F[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        mu = F[tr].mean(0, keepdims=True)
        sd = F[tr].std(0, keepdims=True) + 1e-8
        r = [head_fit((F[tr]-mu)/sd, y[tr], (F[va]-mu)/sd, y[va],
                      [(F[va]-mu)/sd, (F[te]-mu)/sd], seed=s) for s in (0, 1, 2)]
        pv = np.mean([a[0] for a in r], 0)
        pt = np.mean([a[1] for a in r], 0)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, Fm, _ = precision_recall_fscore_support(y[te], pred,
                                                      labels=[0, 1, 2],
                                                      zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), Fm.mean()])
    a = np.array(out); m, s = a.mean(0), a.std(0)
    print(f"{'config':>36}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    print(f"{'frozen ViT-B/16 + linear head':>36}"
          + "".join(f"{m[i]:>9.3f}" for i in range(5))
          + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)))
    print(f"{'reference: 5k two-stream model':>36}"
          f"{0.639:>9.3f}{0.655:>9.3f}{0.685:>9.3f}{0.660:>9.3f}{0.593:>9.3f}")
    np.save("scratch/frozen_vit_scores.npy", a)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
