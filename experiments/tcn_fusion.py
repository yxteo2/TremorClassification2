"""Where the descriptors meet the TCN, rather than whether they are present.

`TwoStreamNet` fuses by **late concatenation**: the spectrum goes through a conv
trunk, the descriptors through a small MLP, the trajectory through a dilated TCN,
and the three feature vectors are concatenated immediately before the classifier.

    parts = [self.spec_feat_fn(self.spec, s), self.traj(t)]
    if d is not None: parts.append(self.desc(d))
    return self.head(torch.cat(parts, dim=1))

That is one choice out of several, and it is the one where the descriptors can
influence the spectrum representation the *least* -- they never touch it. The
convolutional trunk extracts the same features from a 4 Hz peak whether the
patient's bandwidth is 1 Hz or 4 Hz, because bandwidth arrives only after the
trunk has finished.

Four integration points, same inputs and same trunk depth throughout, so only
the meeting place changes:

  late concat (current)   descriptors join at the classifier
  early channels          each descriptor is broadcast along frequency and fed
                          as an extra INPUT channel, so the first convolution
                          already sees them alongside the spectrum
  FiLM                    descriptors generate a per-channel scale and shift for
                          every residual block (Perez et al.). The canonical way
                          to condition a conv net on a feature vector, and
                          parameter-light: 2 numbers per channel per block.
  gate                    descriptors generate one multiplicative gate per trunk
                          channel, applied once after pooling. The cheapest form
                          of conditioning, as a control on FiLM's extra depth.

The trajectory stream is untouched and concatenated at the head in every arm, so
this isolates the descriptor-spectrum interface alone.

**The honest prior.** Thirteen feature unions and every attention mechanism have
failed here because they add parameters, and 49 ET patients cannot pay for them.
FiLM and gate both add parameters. `early channels` roughly does not -- it moves
the descriptors from an MLP branch into the input, so the branch disappears as
the input widens. If anything wins, the prediction is that it is that one, and
parameter counts are printed so the reading is not guesswork.

Run: ``python -m experiments.tcn_fusion``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit

from common.protocol import TEST_FRAC, VAL_FRAC, train, tune_offsets
from experiments.alltasks_final import paired
from experiments.final_model import NBIN, SPLITS, TL, build
from models.architectures import ResidualTCN, TrajectoryEncoder

NM = ("precN", "precPD", "precET", "macroP", "macroF1")
SEEDS = (0, 1, 2)


class FusionTCN(nn.Module):
    """Residual dilated TCN over frequency, with a choice of where descriptors enter.

    Input is packed ``[spectrum | descriptors | trajectory.flatten()]`` to fit the
    existing flat-matrix harness, exactly as ``TwoStreamNet`` expects.
    """

    def __init__(self, n_spec, n_desc, traj_len, mode="late", ch=16,
                 dilations=(1, 2, 4), dropout=0.2, traj_dim=16,
                 desc_hidden=16, num_classes=3, n_traj_ch=2):
        super().__init__()
        self.n_spec, self.n_desc, self.traj_len = n_spec, n_desc, traj_len
        self.n_traj_ch, self.mode, self.ch = n_traj_ch, mode, ch

        c_in = 1 + (n_desc if mode == "early" else 0)
        blocks = []
        for d in dilations:
            blocks.append(nn.ModuleDict({
                "body": nn.Sequential(
                    nn.Conv1d(c_in, ch, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(ch, ch, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)),
                "skip": nn.Identity() if c_in == ch else nn.Conv1d(c_in, ch, 1),
            }))
            c_in = ch
        self.blocks = nn.ModuleList(blocks)

        if mode == "film":
            self.film = nn.ModuleList(
                [nn.Linear(n_desc, 2 * ch) for _ in dilations])
        if mode == "gate":
            self.gate = nn.Linear(n_desc, ch)
        self.desc = (nn.Sequential(nn.Linear(n_desc, desc_hidden), nn.ReLU(),
                                   nn.Dropout(dropout))
                     if mode == "late" and n_desc else None)

        self.traj = TrajectoryEncoder(n_traj_ch, traj_dim)
        d_head = ch + traj_dim + (desc_hidden if self.desc is not None else 0)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_head, num_classes))

    def forward(self, x):
        i = self.n_spec
        s = x[:, :i]
        dsc = x[:, i:i + self.n_desc]
        t = x[:, i + self.n_desc:].reshape(x.shape[0], self.n_traj_ch,
                                           self.traj_len)
        z = s.unsqueeze(1)
        if self.mode == "early":
            # each descriptor held constant across the frequency axis
            z = torch.cat([z, dsc.unsqueeze(-1).expand(-1, -1, i)], dim=1)
        for k, b in enumerate(self.blocks):
            h = b["body"](z)
            if self.mode == "film":
                g, be = self.film[k](dsc).chunk(2, dim=1)
                h = h * (1.0 + g.unsqueeze(-1)) + be.unsqueeze(-1)
            z = torch.relu(h + b["skip"](z))
        z = z.mean(-1)
        if self.mode == "gate":
            z = z * torch.sigmoid(self.gate(dsc))
        parts = [z, self.traj(t)]
        if self.desc is not None:
            parts.append(self.desc(dsc))
        return self.head(torch.cat(parts, dim=1))


def evaluate(mode, spec, desc, traj, y, key, splits=SPLITS):
    """FusionTCN soft-voted with the plain ResidualTCN, as the reported model does."""
    nd = desc.shape[1]
    packed = np.hstack([spec, desc, traj])
    mk1 = lambda: FusionTCN(NBIN, nd, TL, mode=mode)
    mk2 = lambda: ResidualTCN(NBIN, num_classes=3, ch=16)
    out = []
    for sp in range(splits):
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
        pv, pt = np.mean(pv_l, 0), np.mean(pt_l, 0)
        pred = (np.log(pt + 1e-12) + tune_offsets(pv, y[va])).argmax(1)
        P, _, F, _ = precision_recall_fscore_support(y[te], pred, labels=[0, 1, 2],
                                                     zero_division=0)
        out.append([P[0], P[1], P[2], P.mean(), F.mean()])
    return np.array(out)


def main():
    torch.set_num_threads(1)
    d = build()
    y, key = d["y"], d["key"]
    D = np.hstack([d["DESC"], d["ASYM"], d["HAVE"]])
    traj, spec = d["TRAJ"], d["SPEC"]["multitaper"]
    nd = D.shape[1]

    print(f"n={len(y)}  N={int((y==0).sum())} PD={int((y==1).sum())} "
          f"ET={int((y==2).sum())}   {SPLITS} splits")
    print(f"spectrum {NBIN} bins, {nd} descriptors, trajectory {TL}\n")

    MODES = (("late concat (current)", "late"),
             ("early input channels", "early"),
             ("FiLM conditioning", "film"),
             ("channel gate", "gate"))

    print(f"{'arm':>26}{'params':>9}" + "".join(f"{c:>9}" for c in NM)
          + "   sd(macroP)")
    res = {}
    for lab, mode in MODES:
        npar = sum(p.numel() for p in FusionTCN(NBIN, nd, TL, mode=mode)
                   .parameters() if p.requires_grad)
        res[lab] = evaluate(mode, spec, D, traj, y, key)
        m = res[lab].mean(0)
        print(f"{lab:>26}{npar:>9}" + "".join(f"{v:>9.3f}" for v in m)
              + f"{res[lab][:, 3].std():>12.3f}", flush=True)

    base = res["late concat (current)"]
    print("\npaired vs late concat, same splits:")
    for lab, _ in MODES[1:]:
        print(f"  {lab}:")
        for (dd, lo, hi), c in zip(paired(res[lab], base), NM):
            star = "*" if lo > 0 or hi < 0 else " "
            print(f"    {c:>8} {dd:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")
    print("\nMARKER_DONE", flush=True)


if __name__ == "__main__":
    main()
