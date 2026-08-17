"""Train on WINDOWS, predict on PATIENTS -- the standard fix for small n.

Every model in this project trains on **404 rows**, one per patient, because the
spectra are averaged per patient before training. But each recording holds tens
of seconds of signal, so a patient supplies dozens of 4 s windows. Training on
windows and aggregating to patients at inference gives the network **10-40x more
training examples from the same data**, which is the normal remedy when a deep
model is data-limited -- and every diagnostic here says this one is:

  * capacity ordering is 1 k > 3 k > 9 k > 35 k > 11.2 M
  * descriptor fusion still helps (+0.021), so the network is not learning what
    hand-computed peak location and bandwidth already state

This is NOT the window-level evaluation tested earlier (`window_vs_patient_level.md`),
which scored each window as its own case and came out WORSE than patient-level.
Here windows are a **training** device only; every reported metric is
patient-level, with a patient's windows averaged in probability space.

Splits remain patient-disjoint: a patient's windows are all in one fold, so no
window of a test patient is ever seen in training.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import NBIN, TEST_FRAC, VAL_FRAC, tune_offsets
from models.architectures import Spectrum1DCNN

SPLITS = 20
WIN_S, HOP_S, FS = 4.0, 2.0, 100.0
F_LO, F_HI = 3.0, 15.0


def window_spectra(recs, ch, fs=FS, win_s=WIN_S, hop_s=HOP_S, nb=NBIN):
    """(windows, nb) log-binned spectra, with the patient each window came from."""
    n, hop = int(win_s * fs), int(hop_s * fs)
    X, y, pat = [], [], []
    for r in recs:
        sig = r.x[ch] if r.x.shape[0] > 3 else r.x
        sig = np.atleast_2d(np.asarray(sig))
        T = sig.shape[-1]
        if T < n:
            starts = [0]
        else:
            starts = list(range(0, T - n + 1, hop))
        for s0 in starts:
            seg = sig[:, s0:s0 + n]
            f, P = welch(seg, fs=fs, nperseg=min(256, seg.shape[-1]), axis=-1)
            P = P.mean(0)
            k = (f >= F_LO) & (f <= F_HI)
            v = P[k]
            if v.sum() <= 0:
                continue
            v = v / v.sum()
            m = len(v) // nb * nb
            if m == 0:
                continue
            X.append(np.log(v[:m] + 1e-8).reshape(nb, -1).mean(1))
            y.append(r.y)
            pat.append(r.subject)
    return np.array(X, np.float32), np.array(y), np.array(pat)


def patient_from_windows(prob, pat):
    """Average window probabilities within each patient."""
    order = sorted(set(pat.tolist()))
    idx = {p: i for i, p in enumerate(order)}
    acc = np.zeros((len(order), prob.shape[1]))
    cnt = np.zeros(len(order))
    for p, pr in zip(pat, prob):
        acc[idx[p]] += pr
        cnt[idx[p]] += 1
    return acc / np.maximum(cnt, 1)[:, None], np.array(order)


def train_windows(model_fn, Xtr, ytr, Xva, yva, Xout, seed=0, epochs=60,
                  lr=3e-3, wd=1e-3, nc=3, batch=256):
    """Minibatch here, unlike the patient-level loop: with thousands of windows
    a single batch no longer fits the data, and full-batch would give 60 steps."""
    torch.manual_seed(seed)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = T(Xtr), torch.tensor(ytr, dtype=torch.long)
    xv, yv = T(Xva), torch.tensor(yva, dtype=torch.long)
    m = model_fn()
    c = np.bincount(ytr, minlength=nc).astype(float)
    w = torch.tensor(c.sum() / (nc * np.maximum(c, 1)), dtype=torch.float32)
    lf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    best, state = np.inf, None
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), batch):
            ix = perm[i:i + batch]
            if len(ix) < 2:
                continue
            opt.zero_grad(); lf(m(xt[ix]), yt[ix]).backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            v = float(lf(m(xv), yv))
        if v < best:
            best = v
            state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    if state:
        m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return [torch.softmax(m(T(z)), 1).numpy() for z in Xout]


def main():
    torch.set_num_threads(1)
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    rA = load_quaternion_recordings("Data", action="OUT", mode="angular_velocity")
    rB = load_2025_all(conditions=("OUT",))
    rC = load_pads_extracted("pads_stretchhold")
    parts = [window_spectra(rA, slice(3, 6)), window_spectra(rB, slice(3, 6)),
             window_spectra(rC, slice(0, 3))]
    Xw = np.vstack([p[0] for p in parts])
    yw = np.concatenate([p[1] for p in parts])
    pw = np.concatenate([p[2] for p in parts])

    # cap PADS at 90 patients/class, as everywhere else
    pads = np.char.startswith(pw.astype(str), "PADS")
    rng = np.random.default_rng(0)
    keep_pat = set(pw[~pads].tolist())
    lab = {p: yw[pw == p][0] for p in set(pw[pads].tolist())}
    for c in (0, 1, 2):
        cand = sorted([p for p, l in lab.items() if l == c])
        keep_pat |= set(rng.choice(cand, min(90, len(cand)), replace=False).tolist())
    m = np.array([p in keep_pat for p in pw])
    Xw, yw, pw = Xw[m], yw[m], pw[m]

    pats = np.array(sorted(set(pw.tolist())))
    plab = np.array([yw[pw == p][0] for p in pats])
    per = np.array([(pw == p).sum() for p in pats])
    print(f"patients {len(pats)}   windows {len(yw)}   "
          f"windows/patient median {int(np.median(per))} "
          f"(min {per.min()}, max {per.max()})")
    print(f"training rows: {len(yw)} windows vs {len(pats)} patients "
          f"= {len(yw)/len(pats):.1f}x more\n")

    key = np.array([f"{p.split('_')[0]}_{l}" for p, l in zip(pats, plab)])
    mk = lambda: Spectrum1DCNN(NBIN, num_classes=3, ch=8)

    out = []
    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(pats, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(pats[tv],
                                                                    key[tv]))
        tr_p, va_p, te_p = set(pats[tv[t0]]), set(pats[tv[v0]]), set(pats[te])
        itr = np.array([p in tr_p for p in pw])
        iva = np.array([p in va_p for p in pw])
        ite = np.array([p in te_p for p in pw])
        mu = Xw[itr].mean(0, keepdims=True)
        sd = Xw[itr].std(0, keepdims=True) + 1e-8
        r = [train_windows(mk, (Xw[itr] - mu) / sd, yw[itr],
                           (Xw[iva] - mu) / sd, yw[iva],
                           [(Xw[iva] - mu) / sd, (Xw[ite] - mu) / sd], seed=s)
             for s in (0, 1, 2)]
        pv_w = np.mean([a[0] for a in r], 0)
        pt_w = np.mean([a[1] for a in r], 0)
        pv, va_ids = patient_from_windows(pv_w, pw[iva])
        pt, te_ids = patient_from_windows(pt_w, pw[ite])
        yv = np.array([yw[pw == p][0] for p in va_ids])
        yt = np.array([yw[pw == p][0] for p in te_ids])
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, yv)).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(yt, pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    a = np.array(out); m_, s_ = a.mean(0), a.std(0)
    print(f"{'config':>34}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
          f"{'macroF1':>9}  |{'  sd':>7}")
    print(f"{'WINDOW-trained, patient-scored':>34}"
          + "".join(f"{m_[i]:>9.3f}" for i in range(5))
          + "  |" + "".join(f"{s_[i]:>7.3f}" for i in range(5)))
    print(f"\n{'patient-trained reference (final_model)':>44}"
          f"  precET 0.685  macroP 0.660")
    np.save("scratch/window_training_scores.npy", a)
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
