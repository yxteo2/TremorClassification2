"""Masked-spectrum self-supervised pretraining, then fine-tune on PD vs ET.

Every model here trains on 423 labelled patients. Far more **unlabelled** signal
exists and has never been used: PADS has two extracted tasks across two wrists
(~1500 recordings), 2015 has three actions, NewData seven tasks across two
limbs -- roughly 3000 recordings, about 7x the labelled set.

This pretrains an encoder to reconstruct **masked frequency bins** from the rest
of the spectrum, on all of it, with no labels, then fine-tunes on PD-vs-ET.

Why masked reconstruction rather than contrastive:

Contrastive learning builds invariance to whatever the positive pairs declare to
be nuisance, and on this data every natural choice destroys a measured
biomarker -- amplitude scaling removes tremor amplitude, time warping removes
tremor frequency, left-vs-right positives remove the asymmetry finding. Masked
reconstruction requires **no invariance choice at all**: the task is to predict
held-out bins, so nothing is deliberately discarded. It is also what the
published state-of-the-art result on PADS uses.

The honest prior: every capacity-adding method in this project has failed below
~28 minority patients, and pretraining does not add minority patients. What it
adds is a better initialisation for the encoder, which is a different lever.

Run: ``python -m experiments.masked_pretrain``
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common.cohorts import logbin
from experiments.final_model import method_table
from experiments.pd_vs_et import build as build_labelled

NBIN, REPEATS = 16, 10


# --------------------------------------------------------------------------- #
# Unlabelled corpus: every recording, every task, every cohort
# --------------------------------------------------------------------------- #
def unlabelled_spectra():
    from common.load_2025 import ALL_TASKS_2025, load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    chunks = []
    for action in ("OUT", "REST", "WING"):
        try:
            r = load_quaternion_recordings("Data", action=action,
                                           mode="angular_velocity")
            chunks.append(("2015-" + action, r, slice(3, 6)))
        except Exception:
            pass
    for task in ALL_TASKS_2025:
        try:
            r = load_2025_all(conditions=(task,))
            if r:
                chunks.append(("New-" + task, r, slice(3, 6)))
        except Exception:
            pass
    for folder in ("pads_stretchhold", "pads_relaxed"):
        if os.path.isdir(folder):
            try:
                chunks.append((folder, load_pads_extracted(folder), slice(0, 3)))
            except Exception:
                pass

    out = []
    for tag, recs, ch in chunks:
        rows = []
        for r in recs:
            x = r.x[ch] if r.x.shape[0] > 3 else r.x
            from scipy.signal import welch
            f, P = welch(np.atleast_2d(x), fs=100.0,
                         nperseg=min(512, x.shape[-1]), axis=-1)
            P = P.mean(0)
            k = (f >= 3.0) & (f <= 15.0)
            v = P[k]
            if v.sum() <= 0:
                continue
            rows.append(v / v.sum())
        if rows:
            arr = logbin(np.array(rows))
            out.append(arr)
            print(f"    {tag:>22}: {len(arr):>5} recordings")
    X = np.vstack(out)
    return np.nan_to_num(X.astype(np.float32))


# --------------------------------------------------------------------------- #
# Encoder + masked reconstruction
# --------------------------------------------------------------------------- #
class SpectrumEncoder(nn.Module):
    def __init__(self, n_bins=NBIN, d=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(32 * 4, d), nn.ReLU())
        self.d = d

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class MaskedModel(nn.Module):
    """Encoder plus a decoder that reconstructs the full spectrum."""

    def __init__(self, n_bins=NBIN, d=32):
        super().__init__()
        self.enc = SpectrumEncoder(n_bins, d)
        self.dec = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_bins))

    def forward(self, x):
        return self.dec(self.enc(x))


def pretrain(X, epochs=300, mask_frac=0.25, lr=3e-3, seed=0, batch=256):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-8
    Xt = (Xt - mu) / sd
    m = MaskedModel()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    n_mask = max(1, int(mask_frac * NBIN))
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            xb = Xt[perm[i:i + batch]]
            if len(xb) < 2:
                continue
            mask = torch.zeros_like(xb, dtype=torch.bool)
            idx = torch.rand(xb.shape, generator=g).argsort(1)[:, :n_mask]
            mask.scatter_(1, idx, True)
            xin = xb.masked_fill(mask, 0.0)
            opt.zero_grad()
            # loss on the MASKED bins only -- the reconstruction task proper
            loss = ((m(xin) - xb) ** 2 * mask).sum() / mask.sum()
            loss.backward(); opt.step()
            tot += float(loss) * len(xb)
        sch.step()
        if ep % 100 == 0:
            print(f"      pretrain epoch {ep:>3}  masked MSE {tot/len(Xt):.4f}",
                  flush=True)
    return m.enc, (mu.numpy(), sd.numpy())


def finetune(enc_state, norm, Xtr, ytr, Xte, seed=0, epochs=200, lr=3e-3,
             freeze=False):
    torch.manual_seed(seed)
    mu, sd = norm
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    enc = SpectrumEncoder()
    if enc_state is not None:
        enc.load_state_dict(enc_state)
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
    head = nn.Linear(enc.d, 2)
    params = list(head.parameters()) + ([] if freeze else list(enc.parameters()))
    c = np.bincount(ytr, minlength=2).astype(float)
    w = torch.tensor(c.sum() / (2 * np.maximum(c, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        enc.train(); head.train(); opt.zero_grad()
        lf(head(enc(xt)), yt).backward(); opt.step(); sch.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        return torch.softmax(head(enc(xe)), 1).numpy()[:, 1]


def ppv(sens, spec, p):
    return sens * p / (sens * p + (1 - spec) * (1 - p) + 1e-12)


def main():
    torch.set_num_threads(1)
    print("building unlabelled corpus ...")
    U = unlabelled_spectra()
    print(f"  unlabelled: {U.shape[0]} recordings, {U.shape[1]} bins\n")

    print("pretraining (masked bin reconstruction) ...")
    enc, norm = pretrain(U)
    enc_state = {k: v.clone() for k, v in enc.state_dict().items()}
    print()

    data = build_labelled()
    for name, tags, k in (("PADS", ["PADS"], 5),
                          ("MERGED", ["2015", "NewData", "PADS"], 5),
                          ("in-house", ["2015", "NewData"], 3)):
        spec = np.vstack([data[t][0]["spectrum"] for t in tags])
        y3 = np.concatenate([data[t][1] for t in tags])
        m = y3 != 0
        y = (y3[m] == 2).astype(int)
        X = np.nan_to_num(spec[m])
        print(f"\n{'='*72}")
        print(f"{name}  PD vs ET  n={len(y)} ET={int(y.sum())}  spectrum only")
        print(f"{'='*72}")
        print(f"{'init':>28}{'AUC':>16}{'bal-acc':>16}{'ETsens':>10}{'PDspec':>10}")
        res = {}
        for lab, st, fr in (("random init (scratch)", None, False),
                            ("SSL pretrained, fine-tuned", enc_state, False),
                            ("SSL pretrained, frozen", enc_state, True)):
            rows = []
            for rep in range(REPEATS):
                p = np.zeros(len(y))
                for tr, te in StratifiedKFold(k, shuffle=True,
                                              random_state=rep).split(X, y):
                    p[te] = np.mean([finetune(st, norm, X[tr], y[tr], X[te],
                                              seed=s, freeze=fr)
                                     for s in (0, 1)], 0)
                pr = (p >= np.quantile(p, 1 - y.mean())).astype(int)
                se = recall_score(y, pr, pos_label=1, zero_division=0)
                sp = recall_score(y, pr, pos_label=0, zero_division=0)
                rows.append([roc_auc_score(y, p), 0.5 * (se + sp), se, sp])
            a = np.array(rows); res[lab] = a
            mu_, sd_ = a.mean(0), a.std(0)
            print(f"{lab:>28}" + "".join(f"{mu_[i]:>10.3f} +/-{sd_[i]:<4.3f}"
                                         for i in range(2))
                  + f"{mu_[2]:>10.3f}{mu_[3]:>10.3f}")
        base = res["random init (scratch)"]
        print(f"\n  paired vs random init:")
        for q in ("SSL pretrained, fine-tuned", "SSL pretrained, frozen"):
            d = res[q] - base
            for i, nm in enumerate(("AUC", "bal-acc")):
                boot = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d),
                                                                replace=True))
                        for s in range(4000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                star = "*" if lo > 0 or hi < 0 else " "
                print(f"    {q:>28} {nm:>8} {d[:, i].mean():+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
