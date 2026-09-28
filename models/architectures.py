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


#: Names of the six left-right asymmetry descriptors, in column order.
ASYM_NAMES = ("corr", "cos", "peak_df", "log_peak_ratio", "log_power_ratio", "l1")


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


class ResidualTCN(nn.Module):
    """A TCN with actual residual blocks, over the frequency axis.

    :class:`SpectrumTCN` is a plain dilated conv stack with no residual
    connections, so it is a deep feedforward net rather than a TCN in the
    Bai-Kolter-Koltun sense. Residual connections are the part of that design
    that makes depth trainable, and their absence is a plausible reason the TCN
    trailed the 1-D CNN despite having a larger receptive field.

    Each block is (dilated conv -> BN -> ReLU -> dropout) twice, plus a 1x1
    projection shortcut when the channel count changes.
    """

    def __init__(self, n_bins, num_classes=3, ch=16, dropout=0.2,
                 dilations=(1, 2, 4), pool="avg"):
        super().__init__()
        self.pool_kind = pool
        blocks, c_in = [], 1
        for d in dilations:
            blocks.append(nn.ModuleDict({
                "body": nn.Sequential(
                    nn.Conv1d(c_in, ch, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(ch, ch, 3, padding=d, dilation=d),
                    nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)),
                "skip": (nn.Identity() if c_in == ch else nn.Conv1d(c_in, ch, 1)),
            }))
            c_in = ch
        self.blocks = nn.ModuleList(blocks)
        # attention pooling over FREQUENCY, as in AttnPoolBiLSTM: average
        # pooling weights the 3 Hz bin as heavily as the tremor peak, and a
        # learned weighting was worth +0.012 to the BiLSTM.
        self.attn = nn.Conv1d(ch, 1, 1) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(ch, num_classes)

    def _pool(self, z):
        if self.attn is None:
            return z.mean(-1)
        w = torch.softmax(self.attn(z), dim=-1)      # (B, 1, F)
        return (z * w).sum(-1)

    def trunk(self, x):
        z = x.unsqueeze(1)
        for b in self.blocks:
            z = torch.relu(b["body"](z) + b["skip"](z))
        return z

    def forward(self, x):
        return self.fc(self.drop(self._pool(self.trunk(x))))


class AttnPoolBiLSTM(nn.Module):
    """BiLSTM over frequency with ATTENTION pooling instead of a mean.

    :class:`SpectrumBiLSTM` averages hidden states over all frequency bins,
    weighting the 3 Hz bin exactly as much as the bin at the tremor peak. Tremor
    information is concentrated in a 1-2 Hz neighbourhood of that peak, so a
    learned weighting is the matched read-out: the network chooses which bins to
    listen to rather than being forced to average them.

    Costs one extra (2H -> 1) linear layer over `SpectrumBiLSTM`.
    """

    def __init__(self, n_bins, num_classes=3, hidden=32, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden * 2, num_classes))

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))             # (B, F, 2H)
        w = torch.softmax(self.attn(out), dim=1)       # (B, F, 1) over frequency
        return self.head((out * w).sum(1))


class DescriptorFusion(nn.Module):
    """A spectrum backbone with hand-computed descriptors joined at the head.

    The repo's two model families have never been combined. Logistic regression
    on 10 descriptors and a small net on the raw spectrum score comparably,
    which leaves open that each holds something the other does not: the
    descriptors state peak location, bandwidth and harmonic structure
    explicitly, while the network sees the whole spectral shape.

    Input is ``[spectrum | descriptors]`` concatenated on the feature axis. The
    backbone reads the spectrum slice through ``feat_fn``; the descriptor slice
    goes through a small MLP and is concatenated to the pooled representation
    before the classifier.
    """

    def __init__(self, backbone, feat_fn, n_spec, n_desc, feat_dim,
                 num_classes=3, hidden=16, dropout=0.3):
        super().__init__()
        self.n_spec, self.backbone, self.feat_fn = n_spec, backbone, feat_fn
        self.desc = nn.Sequential(nn.Linear(n_desc, hidden), nn.ReLU(),
                                  nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(feat_dim + hidden, num_classes))

    def forward(self, x):
        s, d = x[:, :self.n_spec], x[:, self.n_spec:]
        return self.head(torch.cat([self.feat_fn(self.backbone, s),
                                    self.desc(d)], dim=1))


#: Feature extractors that strip the classifier off each backbone family, for
#: use as ``DescriptorFusion(feat_fn=...)``. Each returns (B, feat_dim).
TRUNKS = {
    # backbones that build their own input (e.g. a frequency-coordinate
    # channel) must expose .trunk() -- feeding them a bare 1-channel spectrum
    # raises a channel-count error.
    "trunk":  (lambda m, s: m.trunk(s)),
    "cnn":    (lambda m, s: m.conv(s.unsqueeze(1)).flatten(1)),
    "tcn":    (lambda m, s: m.trunk(s).mean(-1)),
    "bilstm": (lambda m, s: m.rnn(s.unsqueeze(-1))[0].mean(1)),
}


# --------------------------------------------------------------------------- #
# Techniques imported from the audio / sound-event-detection literature
# --------------------------------------------------------------------------- #


class TrajectoryEncoder(nn.Module):
    """TCN over the instantaneous-frequency / envelope TRAJECTORY.

    The tremor literature's PD-vs-ET discriminator is the stability of the
    instantaneous frequency over time (Di Biase et al., Brain 2017), not the
    shape of the averaged spectrum. `signal_processing.stability.if_trajectory` produces
    that trajectory as a (2, T) sequence -- centred instantaneous frequency in
    Hz and relative envelope.

    This is a genuine use of a sequence model over TIME, and is not the test
    that previously came back at chance: that one ran a BiLSTM over raw
    61-dimensional spectrogram frames, asking whether spectral SHAPE evolves.
    Here the input is a 2-channel physically meaningful trajectory.
    """

    def __init__(self, n_ch=2, out_dim=16, ch=16, dropout=0.2,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, c_in = [], n_ch
        for d in dilations:
            layers += [nn.Conv1d(c_in, ch, 3, padding=d, dilation=d),
                       nn.BatchNorm1d(ch), nn.ReLU(), nn.Dropout(dropout)]
            c_in = ch
        self.tcn = nn.Sequential(*layers)
        self.out_dim = out_dim
        self.proj = nn.Linear(ch * 2, out_dim)      # mean AND std pooling

    def forward(self, x):                            # (B, 2, T)
        z = self.tcn(x)
        # std pooling matters here: the discriminative quantity is how much the
        # trajectory VARIES, which mean pooling would average away
        h = torch.cat([z.mean(-1), z.std(-1)], dim=1)
        return torch.relu(self.proj(h))


class TwoStreamNet(nn.Module):
    """Spectrum stream + trajectory stream, joined at the head.

    Stream 1 sees the log-binned spectrum (what every model here has used).
    Stream 2 sees the instantaneous-frequency trajectory (what the tremor
    literature says actually separates PD from ET). Optional descriptor vector
    is concatenated alongside.

    Input is packed as ``[spectrum | descriptors | trajectory.flatten()]`` so it
    fits the existing flat-matrix training harness.
    """

    def __init__(self, spec_backbone, spec_feat_fn, spec_dim, n_spec, n_desc,
                 traj_len, num_classes=3, traj_dim=16, desc_hidden=16,
                 dropout=0.3, n_traj_ch=2):
        super().__init__()
        self.n_spec, self.n_desc, self.traj_len = n_spec, n_desc, traj_len
        self.n_traj_ch = n_traj_ch
        self.spec, self.spec_feat_fn = spec_backbone, spec_feat_fn
        self.traj = TrajectoryEncoder(n_traj_ch, traj_dim)
        self.desc = (nn.Sequential(nn.Linear(n_desc, desc_hidden), nn.ReLU(),
                                   nn.Dropout(dropout)) if n_desc else None)
        d = spec_dim + traj_dim + (desc_hidden if n_desc else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        i = self.n_spec
        s = x[:, :i]
        d = x[:, i:i + self.n_desc] if self.n_desc else None
        t = x[:, i + self.n_desc:].reshape(x.shape[0], self.n_traj_ch,
                                           self.traj_len)
        parts = [self.spec_feat_fn(self.spec, s), self.traj(t)]
        if d is not None:
            parts.append(self.desc(d))
        return self.head(torch.cat(parts, dim=1))


