"""Hand the network the demodulated signal instead of the raw waveform.

`waveform_deep.py` feeds a TCN the band-passed waveform. A network reading that
must first learn to **demodulate** — separate amplitude from phase — before it can
see anything about oscillator states, and demodulation costs capacity this cohort
cannot spare.

The analytic signal does that step in closed form. `analytic_channels` returns
two series at 40 Hz instead of one:

    channel 0   log envelope, z-scored     amplitude modulation over time
    channel 1   instantaneous frequency,   deviation from the patient's own
                centred on its own median  median — a *stability* channel

That second channel is the Häring mechanism written directly as a time series:
*"several discrete but stable signal states in PD indicate several central
oscillators, while ET points towards a singular pacemaker"*. Switching between
oscillator states appears as level changes; a single pacemaker as a flat line.

Verified on synthetics before use: a stable 6 Hz oscillation gives IF sd
**0.381 Hz**, while a signal alternating between 5 and 7 Hz states gives
**1.082 Hz** — 2.8x higher, with envelopes indistinguishable (0.955 vs 0.973).
The channel measures state switching and not amplitude.

Centring the IF on the patient's own median matters: it removes absolute tremor
frequency, which the spectrum stream already carries, so this stream adds
stability information rather than duplicating frequency information. It also
makes the channel comparable across cohorts.

Arms on the merged 3-class protocol, 20 splits, trained in one loop so all are
paired:

  A. reported model            spectrum + descriptors + IF trajectory
  B. analytic TCN alone        envelope + IF stability over time
  C. soft vote of A and B

The reported model already has an IF-trajectory stream, but it is summarised to
64 points; this one is full-resolution at 384 and carries the envelope alongside.

Run: ``python -m experiments.analytic_deep``
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, SPLITS, TL, build
from frequency.tables import spectrum_table
from models.architectures import (ResidualTCN, Spectrum1DCNN, TRUNKS,
                                  TwoStreamNet)
from signal_processing.waveform import LENGTH, patient_analytic

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SEEDS = (0, 1, 2)
N_REC, N_CH = 2, 2


class AnalyticTCN(nn.Module):
    """Dilated residual TCN over time on (envelope, IF-stability) channels."""

    def __init__(self, length=LENGTH, n_rec=N_REC, n_ch=N_CH, ch=12,
                 num_classes=3, dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.length, self.n_rec, self.n_ch = length, n_rec, n_ch
        blocks, c_in = [], n_ch
        for d in dilations:
            blocks.append(nn.ModuleDict({
                "body": nn.Sequential(
                    nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)),
                "skip": nn.Identity() if c_in == ch else nn.Conv1d(c_in, ch, 1),
            }))
            c_in = ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch, num_classes))

    def forward(self, x):
        B = x.shape[0]
        k = self.n_rec * self.n_ch * self.length
        w = x[:, :k].reshape(B * self.n_rec, self.n_ch, self.length)
        m = x[:, k:].reshape(B, self.n_rec, 1)
        z = w
        for b in self.blocks:
            z = torch.relu(b["body"](z) + b["skip"](z))
        z = z.mean(-1).reshape(B, self.n_rec, -1)
        z = (z * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(z)


def build_analytic(order):
    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings

    src = [(load_quaternion_recordings("Data", action="OUT",
                                       mode="angular_velocity"), slice(3, 6)),
           (load_2025_all(conditions=("OUT",)), slice(3, 6))]
    if os.path.isdir("pads_stretchhold"):
        src.append((load_pads_extracted("pads_stretchhold"), slice(0, 3)))
    packs, pats = [], []
    for recs, ch in src:
        X, M, _, p = patient_analytic(recs, ch=ch, n_rec=N_REC)
        packs.append(np.hstack([X.reshape(len(X), -1), M]))
        pats.append(p)
    allX, allp = np.vstack(packs), np.concatenate(pats)
    idx = {p: i for i, p in enumerate(allp)}
    miss = [p for p in order if p not in idx]
    if miss:
        print(f"  WARNING {len(miss)} patients have no analytic signal")
    D = allX.shape[1]
    return np.array([allX[idx[p]] if p in idx else np.zeros(D) for p in order],
                    dtype=np.float32)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]
    nd = D.shape[1]
    packed = np.hstack([spec, D, traj])

    from common.load_2025 import load_2025_all
    from common.loaders import load_pads_extracted
    from common.quaternion_data import load_quaternion_recordings
    A = spectrum_table(load_quaternion_recordings("Data", action="OUT",
                                                  mode="angular_velocity"),
                       ch=slice(3, 6))
    B_ = spectrum_table(load_2025_all(conditions=("OUT",)), ch=slice(3, 6))
    C = spectrum_table(load_pads_extracted("pads_stretchhold"), ch=slice(0, 3))
    rng = np.random.default_rng(0)
    keep = []
    for cl in (0, 1, 2):
        i = np.flatnonzero(C[1] == cl)
        keep.extend(rng.choice(i, min(90, len(i)), replace=False))
    keep = np.array(sorted(keep))
    order = np.concatenate([A[2], B_[2], C[2][keep]])
    assert np.array_equal(np.concatenate([A[1], B_[1], C[1][keep]]), y), \
        "patient order does not match build()"

    print("building analytic tensors ...", flush=True)
    W = build_analytic(order)
    npar = sum(p.numel() for p in AnalyticTCN().parameters()
               if p.requires_grad)
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"analytic pack {W.shape}, AnalyticTCN {npar} params\n", flush=True)

    # what the IF-stability channel says about the classes, before any model
    k = N_REC * N_CH * LENGTH
    ifc = W[:, :k].reshape(len(W), N_REC, N_CH, LENGTH)[:, 0, 1, :]
    print("IF stability (sd of instantaneous frequency, Hz) by class:")
    for cl, nm in ((0, "N"), (1, "PD"), (2, "ET")):
        v = ifc[y == cl].std(1)
        print(f"  {nm:>3}  mean {v.mean():.3f}  median {np.median(v):.3f}")
    print(flush=True)

    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    mkA = lambda: AnalyticTCN()

    ARMS = ("A reported model", "B analytic TCN alone", "C soft vote A + B")
    res = {a: [] for a in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        pv_l, pt_l = [], []
        for X, mk in ((packed, mk1), (spec, mk2)):
            mu = X[tr].mean(0, keepdims=True)
            sd = X[tr].std(0, keepdims=True) + 1e-8
            r = [train(mk, (X[tr] - mu) / sd, y[tr], (X[va] - mu) / sd, y[va],
                       [(X[va] - mu) / sd, (X[te] - mu) / sd], seed=s)
                 for s in SEEDS]
            pv_l.append(np.mean([a[0] for a in r], 0))
            pt_l.append(np.mean([a[1] for a in r], 0))
        pvA, ptA = np.mean(pv_l, 0), np.mean(pt_l, 0)

        # already normalised per recording; do not standardise per time index
        r = [train(mkA, W[tr], y[tr], W[va], y[va], [W[va], W[te]], seed=s)
             for s in SEEDS]
        pvB = np.mean([a[0] for a in r], 0)
        ptB = np.mean([a[1] for a in r], 0)

        def score(pv, pt):
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            return [P[0], P[1], P[2], P.mean(), F.mean()]

        res["A reported model"].append(score(pvA, ptA))
        res["B analytic TCN alone"].append(score(pvB, ptB))
        res["C soft vote A + B"].append(score(0.5 * (pvA + pvB),
                                              0.5 * (ptA + ptB)))
        print(f"  split {sp+1}/{SPLITS} done", flush=True)

    for a in ARMS:
        res[a] = np.array(res[a])

    print(f"\n{'arm':>24}" + "".join(f"{c:>9}" for c in NM) + "   sd(macroP)")
    for a in ARMS:
        print(f"{a:>24}" + "".join(f"{v:>9.3f}" for v in res[a].mean(0))
              + f"{res[a][:, 3].std():>12.3f}")

    print("\npaired vs the reported model, same splits:")
    for a in ARMS[1:]:
        print(f"  {a}:")
        for (dd, lo, hi), c in zip(paired(res[a], res["A reported model"]), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
