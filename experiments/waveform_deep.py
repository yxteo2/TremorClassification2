"""A TCN over TIME on the raw waveform — the one input no model here has seen.

Every network in this project convolves along **frequency**. `ResidualTCN` is
"over the frequency axis", `SpectrumBiLSTM` reads the spectrum "as a sequence
over FREQUENCY", `Spectrum1DCNN` is "a 1-D CNN over the frequency axis". The only
time-domain stream, `TrajectoryEncoder`, sees an instantaneous-frequency
trajectory already reduced to 64 points.

`catch22_waveform_features.md` makes that gap worth closing: six fixed temporal
statistics matched ten tuned spectral descriptors on PADS PD-vs-ET (AUC 0.798 vs
0.794) using 60 % of the dimensions and **half the fold variance**. Temporal
structure carries comparable information. A convolution over time can learn it
adaptively where catch22 applies 22 fixed formulas.

Input from `signal_processing/waveform.py`: band-pass 3-15 Hz, project onto the
principal axis (rotation-invariant; the magnitude would double the frequency),
decimate 100 -> 40 Hz, z-score, centre-crop to 384 samples. The crop length is
chosen so **nothing is ever padded** — padding amount would be a cohort
signature, since PADS recordings are always 1024 samples and NewData always 1000.

Arms on the merged 3-class protocol, 20 splits, everything else identical to the
reported model:

  A. reported model                 spectrum + descriptors + IF trajectory
  B. waveform TCN alone             the new stream on its own
  C. soft vote of A and B           does the waveform add anything?

Both arms are trained inside the same split loop and their probabilities are
averaged for C, so all three are exactly paired.

**The honest prior.** A 384-point sequence at 404 patients is the regime where
this project's capacity findings say things fail. Arm B alone is expected to be
weak; the question is arm C. If the waveform carried information the spectrum
lacks, the vote should beat the reported model even when B is worse than A.

Run: ``python -m experiments.waveform_deep``
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
from signal_processing.waveform import LENGTH, patient_tensor

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SEEDS = (0, 1, 2)
N_REC = 2


class WaveformTCN(nn.Module):
    """Dilated residual TCN over TIME, pooled over a patient's recordings.

    Receptive field with kernel 5 and dilations (1, 2, 4, 8) is 121 samples =
    3.0 s at 40 Hz, about 18 cycles of a 6 Hz tremor — enough to see
    cycle-to-cycle structure without the capacity of a full-length model.

    Input is packed flat as ``[waveforms (n_rec * L) | mask (n_rec)]`` to fit the
    existing training harness.
    """

    def __init__(self, length=LENGTH, n_rec=N_REC, ch=12, num_classes=3,
                 dilations=(1, 2, 4, 8), dropout=0.2):
        super().__init__()
        self.length, self.n_rec = length, n_rec
        blocks, c_in = [], 1
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
        w = x[:, :self.n_rec * self.length].reshape(B * self.n_rec, 1,
                                                    self.length)
        m = x[:, self.n_rec * self.length:].reshape(B, self.n_rec, 1)
        z = w
        for b in self.blocks:
            z = torch.relu(b["body"](z) + b["skip"](z))
        z = z.mean(-1).reshape(B, self.n_rec, -1)          # pool over time
        z = (z * m).sum(1) / m.sum(1).clamp(min=1)         # pool over recordings
        return self.head(z)


def build_waveforms(order):
    """(patients, n_rec * L + n_rec) aligned to the reported patient order."""
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
        X, M, _, p = patient_tensor(recs, ch=ch, n_rec=N_REC)
        packs.append(np.hstack([X.reshape(len(X), -1), M]))
        pats.append(p)
    allX = np.vstack(packs)
    allp = np.concatenate(pats)
    idx = {p: i for i, p in enumerate(allp)}
    miss = [p for p in order if p not in idx]
    if miss:
        print(f"  WARNING {len(miss)} patients have no waveform")
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

    # reference patient order, matching build()
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

    print("building waveform tensors ...", flush=True)
    W = build_waveforms(order)
    nz = (np.abs(W[:, :N_REC * LENGTH]).sum(1) > 0).mean()
    npar = sum(p.numel() for p in WaveformTCN().parameters()
               if p.requires_grad)
    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"waveform pack {W.shape}, {nz:.1%} of patients have a waveform, "
          f"WaveformTCN {npar} params\n", flush=True)

    mk1 = lambda: TwoStreamNet(Spectrum1DCNN(NBIN, 3, ch=8), TRUNKS["cnn"],
                               8 * 2 * 4, NBIN, nd, TL)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    mkW = lambda: WaveformTCN()

    ARMS = ("A reported model", "B waveform TCN alone", "C soft vote A + B")
    res = {a: [] for a in ARMS}

    for sp in range(SPLITS):
        tv, te = next(StratifiedShuffleSplit(1, test_size=TEST_FRAC,
                                             random_state=sp).split(packed, key))
        t0, v0 = next(StratifiedShuffleSplit(1, test_size=VAL_FRAC,
                                             random_state=sp).split(packed[tv],
                                                                    key[tv]))
        tr, va = tv[t0], tv[v0]

        # ---- A: the reported model, standardised as usual ------------------ #
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

        # ---- B: the waveform TCN. NOT standardised: the waveform is already
        # z-scored per recording, and standardising per time index across
        # patients is meaningless and would corrupt the mask columns.
        r = [train(mkW, W[tr], y[tr], W[va], y[va], [W[va], W[te]], seed=s)
             for s in SEEDS]
        pvB = np.mean([a[0] for a in r], 0)
        ptB = np.mean([a[1] for a in r], 0)

        def score(pv, pt):
            pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
            P, _, F, _ = precision_recall_fscore_support(
                y[te], pred, labels=[0, 1, 2], zero_division=0)
            return [P[0], P[1], P[2], P.mean(), F.mean()]

        res["A reported model"].append(score(pvA, ptA))
        res["B waveform TCN alone"].append(score(pvB, ptB))
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
