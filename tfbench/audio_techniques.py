"""Audio-literature techniques applied to tremor spectra.

Research: how the audio / sound-event-detection community builds deep models on
time-frequency data, and which of it transfers here.

Four techniques, each with a specific reason to expect it to matter:

1. **Frequency-aware convolution** (arXiv:2203.15296, arXiv:2403.13252).
   Convolution enforces translation equivariance along frequency, but frequency
   is not a shift-invariant axis. The objection is sharper for tremor than for
   audio: PD rest tremor is 4-6 Hz and ET is 6-12 Hz, so peak LOCATION is the
   most diagnostic quantity, and `Spectrum1DCNN` slides identical filters over
   every bin and cannot represent it. Tested as a frequency-coordinate channel
   (`FreqCoordCNN`) and as per-band kernel mixing (`FreqDynamicCNN`).

2. **PCEN-style learnable compression** (Wang et al., IEEE SPL 2019). PCEN beats
   the fixed pointwise log of a log-mel front-end. Its gain-control term needs a
   time axis, so what carries over is the per-band trainable exponent, which can
   recover log compression as a special case.

3. **SpecAugment frequency masking** (Park et al.). Earlier augmentation here
   shifted and noised the whole spectrum and did nothing; masking bands is a
   different operation that forces the model off any single band.

4. **Multi-resolution input.** Standard in audio. Round 7 measured multitaper,
   cwt and wavelet_packet each beating welch; concatenating them is the
   multi-resolution version of that finding.

**A falsifiable prediction.** `DescriptorFusion` helped the plain CNN by +0.021,
and the descriptors state peak frequency explicitly -- exactly what a
frequency-translation-invariant conv discards. If frequency-aware convolution
captures the same information natively, fusion should stop helping on top of it.
Both arms are run to check.

Run: ``python -m tfbench.audio_techniques``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from tfbench.cohort_strategies import (NBIN, SPLITS, TEST_FRAC, VAL_FRAC,
                                       desc_table, load_all, logbin,
                                       tune_offsets)
from tfbench.small_nets import (AudioStyleNet, DescriptorFusion, FreqCoordCNN,
                                FreqDynamicCNN, ResidualTCN, Spectrum1DCNN,
                                TRUNKS)


def train(model_fn, Xtr, ytr, Xva, yva, Xout, seed=0, epochs=200, lr=3e-3,
          wd=1e-3, nc=3):
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
    best, state = np.inf, None
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        lf(m(xt), yt).backward(); opt.step(); sch.step()
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


def evaluate(name, Xs, y, key, fns, splits=SPLITS, verbose=True,
             standardize=True):
    """``standardize=False`` passes the RAW non-negative spectrum through.

    Required by :class:`LearnableCompression`, which raises its input to a
    fractional power: zero-mean standardised input makes that NaN, which
    silently produced a one-class model (sd 0.000 across every split).
    """
    out = []
    for sp in range(splits):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(Xs[0], key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(Xs[0][tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]
        pv_l, pt_l = [], []
        for X, fn in zip(Xs, fns):
            if standardize:
                mu = X[tr].mean(0, keepdims=True)
                sd = X[tr].std(0, keepdims=True) + 1e-8
            else:
                mu, sd = 0.0, 1.0
            r = [train(fn, (X[tr]-mu)/sd, y[tr], (X[va]-mu)/sd, y[va],
                       [(X[va]-mu)/sd, (X[te]-mu)/sd], seed=s) for s in (0, 1, 2)]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    a = np.array(out)
    if verbose:
        m, s = a.mean(0), a.std(0)
        print(f"{name:>38}" + "".join(f"{m[i]:>9.3f}" for i in range(5))
              + "  |" + "".join(f"{s[i]:>7.3f}" for i in range(5)), flush=True)
    return a


def paired(name, a, base, splits=SPLITS):
    d = a - base
    print(f"  {name}:")
    for i, nm in enumerate(("precN", "precPD", "precET", "macroP", "macroF1")):
        b = [np.mean(np.random.default_rng(s).choice(d[:, i], len(d), replace=True))
             for s in range(4000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"    {nm:>8} {d[:, i].mean():+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")


def main():
    torch.set_num_threads(1)
    sb, dc, y, key, coh = load_all(cap=90)
    F = sb.shape[1]
    nd = dc.shape[1]
    both = np.hstack([sb, dc])
    print(f"n={len(y)}  bins={F}  descriptors+asym={nd}\n")

    H = (f"{'config':>38}{'precN':>9}{'precPD':>9}{'precET':>9}{'macroP':>9}"
         f"{'macroF1':>9}  |{'  sd':>7}")
    RT = lambda: ResidualTCN(F, num_classes=3, ch=16)

    def FUS(bb, trunk="cnn"):
        """Fusion wrapper. Frequency-aware backbones build their own input
        channels, so they need the generic .trunk() extractor."""
        return lambda: DescriptorFusion(bb(), TRUNKS[trunk], F, nd, 8 * 2 * 4)
    print("### A. frequency-aware convolution (the main hypothesis)")
    print(H)
    res = {}
    res["base"] = evaluate("Spectrum1DCNN + ResidualTCN (base)", [both, sb], y, key,
                           [FUS(lambda: Spectrum1DCNN(F, 3, ch=8)), RT])
    res["coord"] = evaluate("FreqCoordCNN + ResidualTCN", [both, sb], y, key,
                            [FUS(lambda: FreqCoordCNN(F, 3, ch=8), "trunk"), RT])
    res["dyn"] = evaluate("FreqDynamicCNN + ResidualTCN", [both, sb], y, key,
                          [FUS(lambda: FreqDynamicCNN(F, 3, ch=8), "trunk"), RT])

    print("\n### B. does frequency-awareness SUBSUME descriptor fusion?")
    print(H)
    res["cnn_nofus"] = evaluate("Spectrum1DCNN alone (no fusion)", [sb], y, key,
                                [lambda: Spectrum1DCNN(F, 3, ch=8)])
    res["coord_nofus"] = evaluate("FreqCoordCNN alone (no fusion)", [sb], y, key,
                                  [lambda: FreqCoordCNN(F, 3, ch=8)])

    print("\n### C. PCEN-style compression and SpecAugment masking")
    print(H)
    raw = np.exp(sb)          # undo the log so learnable compression can act
    res["pcen"] = evaluate("learnable compression (PCEN-ish)", [raw], y, key,
                           [lambda: AudioStyleNet(FreqCoordCNN(F, 3, ch=8), F,
                                                  compress=True)],
                           standardize=False)
    res["logref"] = evaluate("  reference: fixed log, same backbone", [sb], y, key,
                             [lambda: FreqCoordCNN(F, 3, ch=8)])
    for mw in (2, 4):
        res[f"mask{mw}"] = evaluate(f"SpecAugment freq-mask w<={mw}", [sb], y, key,
                                    [lambda mw=mw: AudioStyleNet(
                                        FreqCoordCNN(F, 3, ch=8), F, mask_width=mw)])

    print("\npaired vs base (bootstrap 95 % CI):")
    paired("FreqCoordCNN", res["coord"], res["base"])
    paired("FreqDynamicCNN", res["dyn"], res["base"])
    print("\npaired, fusion-free arms (FreqCoord vs plain CNN):")
    paired("FreqCoordCNN alone", res["coord_nofus"], res["cnn_nofus"])
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
