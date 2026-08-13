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


# --------------------------------------------------------------------------- #
# Runner -- so results are reproducible without a scratch script
# --------------------------------------------------------------------------- #
def spectrum_table(recs, ch=slice(3, 6), fs=100.0, f_lo=3.0, f_hi=15.0,
                   nperseg=512):
    """(patients, F) normalised power spectrum, axes averaged.

    Averaging the axes is a rotation-invariant reduction, not a free loss --
    keeping them separate (``AxisFusionNet``) measured worse at this n.
    """
    from collections import defaultdict
    from scipy.signal import welch
    rows, lab = defaultdict(list), {}
    for r in recs:
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
        P = P.mean(0)
        k = (f >= f_lo) & (f <= f_hi)
        s = P[k]
        rows[r.subject].append(s / (s.sum() + 1e-20))
        lab[r.subject] = r.y
    pats = sorted(rows)
    return (np.nan_to_num(np.array([np.mean(rows[p], 0) for p in pats])),
            np.array([lab[p] for p in pats]), np.array(pats))


def loso_nn(X, y, groups, model_fn, seeds=(0, 1, 2), epochs=150,
            class_weight=False, **kw):
    """Patient-level LOSO with a small net; probability averaged over seeds.

    The scaler is fit on the TRAIN split of each fold only. ``class_weight``
    defaults False: on the frequency BiLSTM it costs precision (0.667 -> 0.600)
    and AUC (0.942 -> 0.870) while buying recall -- sweep both and report both,
    do not assume.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    prob = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        prob[te] = np.mean([fit_predict(model_fn, (X[tr] - mu) / sd, y[tr],
                                        (X[te] - mu) / sd, seed=s, epochs=epochs,
                                        class_weight=class_weight, **kw)
                            for s in seeds], 0)
    return prob


def best_model(n_bins, num_classes=2):
    """The configuration that measured best: frequency BiLSTM, hidden 32.

    bal-acc 0.913 / recall 1.000 with class weighting, or bal-acc 0.790 /
    AUC 0.942 / precision 0.667 without. 9,090 parameters -- the capacity
    optimum; h=128 and the 11-86 M pretrained backbones are both far past it.
    """
    return SpectrumBiLSTM(n_bins, num_classes, hidden=32)


def evaluate(recs, axis="PD_vs_ET", class_weight=False, model_fn=None,
             seeds=(0, 1, 2), epochs=150, ch=slice(3, 6)):
    """End-to-end: recordings -> spectra -> LOSO -> metrics dict."""
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)
    X, y3, g = spectrum_table(recs, ch=ch)
    if axis == "PD_vs_ET":
        m = y3 != 0
        X, y, g = X[m], (y3[m] == 2).astype(int), g[m]
    else:
        y = (y3 != 0).astype(int)
    fn = model_fn or (lambda: best_model(X.shape[1]))
    prob = loso_nn(X, y, g, fn, seeds=seeds, epochs=epochs,
                   class_weight=class_weight)
    pred = (prob >= 0.5).astype(int)
    bal = 0.5 * (recall_score(y, pred, pos_label=1, zero_division=0)
                 + recall_score(y, pred, pos_label=0, zero_division=0))
    return {"bal_acc": bal, "auc": roc_auc_score(y, prob),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "n": len(y), "n_pos": int(y.sum()), "prob": prob, "y": y,
            "patients": g}


class SpectrumTCN(nn.Module):
    """Dilated TCN over the FREQUENCY axis of a 1-D spectrum.

    The convolutional counterpart to :class:`SpectrumBiLSTM`. Dilation lets a
    small stack see the whole 3-15 Hz band without pooling it away: with
    kernel 3 and dilations 1/2/4/8 the receptive field is 31 bins, roughly the
    full spectrum here.

    Included because the recurrent and convolutional families can disagree --
    on PD-vs-ET DRINK the frequency BiLSTM beat the linear model while a
    TCN+BiLSTM hybrid over time did not.
    """

    def __init__(self, n_bins, num_classes=2, ch=16, dropout=0.3,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, c_in = [], 1
        for d in dilations:
            layers += [nn.Conv1d(c_in, ch, 3, padding=d, dilation=d),
                       nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)]
            c_in = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(ch, num_classes))

    def forward(self, x):
        return self.head(self.tcn(x.unsqueeze(1)))


class BilateralAttention(nn.Module):
    """Interleaved self-attention over LEFT and RIGHT limb spectra.

    Follows the interleaved-encoder idea (Vaswani-style blocks over a
    concatenated two-limb sequence with learned modality embeddings), adapted
    in one respect: the paper interleaves along **time**, this interleaves
    along **frequency**.

    That change is forced by what was measured on this data. Tremor is
    quasi-stationary over a 10 s window: a BiLSTM over the time axis of a
    spectrogram sits at chance (bal-acc 0.513) while the same family over the
    frequency axis reaches 0.913. Attention over time would be attending to an
    axis with little structure; attention over frequency attends to spectral
    shape, which is where the discriminative information demonstrably is.

    Rationale for going bilateral at all: PD signs typically begin unilaterally
    and stay more severe on that side, so the left-right *relationship* carries
    information a single-limb model discards. NewData records both limbs per
    subject (action codes 01-07 right, 08-14 left).

    Sequence layout, with F frequency bins per limb::

        H0 = [proj(X_L) + pos + m_L]  ||  [proj(X_R) + pos + m_R]    (2F, d)

    Self-attention over the 2F sequence gives a 2F x 2F map whose diagonal
    blocks are within-limb and whose off-diagonal blocks are left-right
    interactions -- the same interactions an explicit dual-stream
    cross-attention would compute, with one tied set of projection weights
    instead of four.
    """

    def __init__(self, n_bins, num_classes=2, d=32, n_heads=4, n_blocks=2,
                 ff=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(self._sinusoidal(n_bins, d), requires_grad=False)
        self.mod = nn.Parameter(torch.randn(2, d) * 0.02)   # learned L / R tags
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=False)
        self.enc = nn.TransformerEncoder(block, num_layers=n_blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    @staticmethod
    def _sinusoidal(n, d):
        pos = torch.arange(n).float().unsqueeze(1)
        i = torch.arange(0, d, 2).float()
        ang = pos / torch.pow(10000, i / d)
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang[:, :pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x):
        """x: (B, 2*F) -- left spectrum concatenated with right."""
        b, twoF = x.shape
        f = twoF // 2
        xl, xr = x[:, :f].unsqueeze(-1), x[:, f:].unsqueeze(-1)
        hl = self.proj(xl) + self.pos[:f] + self.mod[0]
        hr = self.proj(xr) + self.pos[:f] + self.mod[1]
        h = torch.cat([hl, hr], dim=1)          # (B, 2F, d)
        return self.head(self.enc(h).mean(1))   # pool over all 2F positions
