"""Small networks matched to the data, not to ImageNet.

Every deep model tried here so far was fed a raw spectrogram and asked to learn
its own features: BiLSTM, ResNet18, WideResNet, ViT. All sat at chance, while
logistic regression on 10 hand-computed descriptors reached AUC 0.729 (2015
REST) and 0.812 (NewData DRINK). The obvious reading is that feature learning
is what fails at this n -- not the classifier.

So these two operate where the signal demonstrably is:

``MLPHead``      a 2-layer MLP on the SAME 10 descriptors the linear model uses.
                 Tests whether a non-linear boundary beats a linear one, with
                 ~200 parameters instead of 1e5-1e8.
``Spectrum1DCNN`` a small 1D CNN over the power SPECTRUM (frequency axis only),
                 not the 2-D spectrogram. Tremor structure is spectral, so a 1-D
                 convolution over frequency is the matched inductive bias; a 2-D
                 image model spends its capacity on a time axis that the
                 descriptors already showed adds nothing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """2-layer MLP on precomputed descriptors."""

    def __init__(self, n_features, num_classes=2, hidden=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))

    def forward(self, x):
        return self.net(x)


class Spectrum1DCNN(nn.Module):
    """1-D CNN over the frequency axis of a power spectrum."""

    def __init__(self, n_bins, num_classes=2, ch=8, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(ch * 2 * 4, num_classes))

    def forward(self, x):
        return self.head(self.conv(x.unsqueeze(1)))


def fit_predict(model_fn, Xtr, ytr, Xte, epochs=300, lr=3e-3, wd=1e-3,
                seed=0, class_weight=True, device="cpu"):
    """Train on a fold and return test probabilities. Small enough for full-batch."""
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    m = model_fn().to(device)
    w = None
    if class_weight:
        cnt = np.bincount(ytr, minlength=2).astype(float)
        w = torch.tensor(cnt.sum() / (2 * np.maximum(cnt, 1)),
                         dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(m(Xtr_t), ytr_t).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(Xte_t), 1).cpu().numpy()[:, 1]


class SpectrumBiLSTM(nn.Module):
    """BiLSTM reading the power spectrum as a sequence over FREQUENCY.

    The repo's `tremor_bilstm` runs over the TIME axis of a full spectrogram and
    sits at chance on PD-vs-ET. This runs over the **frequency** axis of a 1-D
    spectrum instead: the sequence is "power at 3 Hz, 3.2 Hz, ..., 15 Hz", so
    the recurrence models how spectral shape unfolds across frequency -- the
    structure the descriptors summarise by hand.

    Sized to match `MLPHead` (hundreds of parameters), not the ~1e5 of the
    spectrogram BiLSTM.
    """

    def __init__(self, n_bins, num_classes=2, hidden=8, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden * 2, num_classes))

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))     # (B, n_bins, 2*hidden)
        return self.head(out.mean(1))          # average over frequency


class TCNBiLSTM(nn.Module):
    """TCN over FREQUENCY per time frame, then BiLSTM over TIME.

    Motivated by two measured results on this data:

    * a BiLSTM over the **time** axis of a raw spectrogram sits at chance
      (bal-acc 0.513) -- tremor is quasi-stationary, so there is little for a
      sequence model to find along time in the raw bins;
    * a BiLSTM over the **frequency** axis of a time-averaged spectrum reaches
      0.851 -- the spectral shape is where the signal lives.

    So: extract spectral shape first, then aggregate over time. A dilated 1-D
    conv stack runs across frequency **independently at each frame**, producing
    one embedding per time step; the BiLSTM then summarises how that embedding
    evolves. Time-averaging (what `SpectrumBiLSTM` does implicitly) is thrown
    away only if the temporal evolution carries something -- this architecture
    can learn to ignore it.

    Input ``(B, F, T)``. Kept small: a few thousand parameters.
    """

    def __init__(self, n_freq, num_classes=2, tcn_ch=8, rnn_hidden=8,
                 dropout=0.3, dilations=(1, 2, 4)):
        super().__init__()
        layers, c_in = [], 1
        for d in dilations:
            layers += [nn.Conv1d(c_in, tcn_ch, 3, padding=d, dilation=d),
                       nn.BatchNorm1d(tcn_ch), nn.ReLU()]
            c_in = tcn_ch
        self.tcn = nn.Sequential(*layers)          # over the FREQUENCY axis
        self.pool = nn.AdaptiveAvgPool1d(1)        # one embedding per frame
        self.rnn = nn.LSTM(tcn_ch, rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(rnn_hidden * 2, num_classes))

    def forward(self, x):
        b, f, t = x.shape
        # every (sample, frame) pair becomes one sequence over frequency
        z = x.permute(0, 2, 1).reshape(b * t, 1, f)
        z = self.pool(self.tcn(z)).reshape(b, t, -1)   # (B, T, tcn_ch)
        out, _ = self.rnn(z)
        return self.head(out.mean(1))


class AxisFusionNet(nn.Module):
    """(3, F, T) per-axis spectrograms -> TCN fuses x/y/z -> BiLSTM over frequency.

    Every model tried before this one collapsed the three angular-velocity axes
    into a single spectrum by averaging, discarding per-axis structure. That is
    a real loss: `pdetn/quaternion_tf.py` showed cross-axis phase carries orbit
    geometry (circularity, handedness) that no per-axis power average can see.

    Pipeline:
      1. stack x, y, z spectrograms -> (B, 3, F, T)
      2. a small dilated conv **across the axis dimension** at each (f, t) cell
         fuses the three components into `fuse_ch` channels -- this is the
         "TCN sums the xyz components" step
      3. average over time (tremor is quasi-stationary; a time-axis BiLSTM was
         measured at chance while a frequency-axis one reached 0.913)
      4. BiLSTM over FREQUENCY on the fused channels, then classify

    Kept in the 9-35 k parameter band, where the frequency BiLSTM peaked.
    """

    def __init__(self, n_freq, n_axes=3, num_classes=2, fuse_ch=8,
                 rnn_hidden=32, dropout=0.3):
        super().__init__()
        # fuse across axes: treat the 3 axes as the conv "length" dimension
        self.fuse = nn.Sequential(
            nn.Conv1d(1, fuse_ch, 3, padding=1), nn.BatchNorm1d(fuse_ch), nn.ReLU(),
            nn.Conv1d(fuse_ch, fuse_ch, 3, padding=2, dilation=2),
            nn.BatchNorm1d(fuse_ch), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.rnn = nn.LSTM(fuse_ch, rnn_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(rnn_hidden * 2, num_classes))

    def forward(self, x):
        b, a, f, t = x.shape
        z = x.mean(-1)                              # (B, A, F) -- average time
        z = z.permute(0, 2, 1).reshape(b * f, 1, a)  # each (sample, freq): seq over axes
        z = self.fuse(z).reshape(b, f, -1)          # (B, F, fuse_ch)
        out, _ = self.rnn(z)                        # over FREQUENCY
        return self.head(out.mean(1))
