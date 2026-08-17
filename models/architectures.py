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


class SpectrumResNet1D(nn.Module):
    """Residual 1-D CNN over the FREQUENCY axis -- the combination of what won.

    Three things have independently helped on this data, and this is the model
    that has all three at once:

    * a **1-D** view over frequency beat every 2-D spectrogram model
      (`Spectrum1DCNN` LOCO 0.506 against `Small2DCNN` 0.432);
    * **residual connections** turned the plain dilated stack into the best
      single model (`SpectrumTCN` 0.500 -> `ResidualTCN` 0.510, ET precision
      0.269 -> 0.352);
    * **ResNet18's** depth/BatchNorm/minibatch recipe gave the best ET precision
      of any 2-D model (0.287) despite the 2-D input being the wrong view.

    Structure: stem conv, then `n_blocks` basic residual blocks with stride-2
    downsampling on the frequency axis, then global pooling. Kept in the
    1e4-parameter band where every model on this cohort has peaked, rather than
    ResNet18's 11 M.
    """

    def __init__(self, n_bins, num_classes=3, ch=16, n_blocks=3, dropout=0.2,
                 pool="avg"):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, ch, 5, padding=2),
                                  nn.BatchNorm1d(ch), nn.ReLU())
        blocks = []
        c = ch
        for i in range(n_blocks):
            co = c * 2 if i else c
            stride = 2 if i else 1
            blocks.append(nn.ModuleDict({
                "body": nn.Sequential(
                    nn.Conv1d(c, co, 3, stride=stride, padding=1),
                    nn.BatchNorm1d(co), nn.ReLU(), nn.Dropout(dropout),
                    nn.Conv1d(co, co, 3, padding=1), nn.BatchNorm1d(co)),
                "skip": (nn.Identity() if (stride == 1 and c == co)
                         else nn.Sequential(nn.Conv1d(c, co, 1, stride=stride),
                                            nn.BatchNorm1d(co))),
            }))
            c = co
        self.blocks = nn.ModuleList(blocks)
        self.attn = nn.Conv1d(c, 1, 1) if pool == "attn" else None
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        z = self.stem(x.unsqueeze(1))
        for b in self.blocks:
            z = torch.relu(b["body"](z) + b["skip"](z))
        if self.attn is None:
            z = z.mean(-1)
        else:
            z = (z * torch.softmax(self.attn(z), dim=-1)).sum(-1)
        return self.fc(self.drop(z))


class MultiTaskSpectrum(nn.Module):
    """Backbone with an auxiliary REGRESSION head on the tremor peak frequency.

    The classification signal is 404 labels; the spectrum additionally contains
    a continuous, physically meaningful target -- where the tremor peak sits --
    that needs no extra annotation. Predicting it as an auxiliary task gives the
    shared trunk a denser gradient than three class labels alone, which is the
    standard argument for multi-task learning in the small-label regime.

    ``forward`` returns (logits, peak_prediction); the trainer weights the
    auxiliary MSE by ``aux_weight``.
    """

    def __init__(self, backbone, feat_fn, feat_dim, num_classes=3, dropout=0.2):
        super().__init__()
        self.backbone, self.feat_fn = backbone, feat_fn
        self.cls = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(feat_dim, num_classes))
        self.aux = nn.Linear(feat_dim, 1)

    def forward(self, x):
        h = self.feat_fn(self.backbone, x)
        return self.cls(h), self.aux(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Techniques imported from the audio / sound-event-detection literature
# --------------------------------------------------------------------------- #
class FreqCoordCNN(nn.Module):
    """1-D CNN with an explicit FREQUENCY COORDINATE channel (CoordConv-style).

    Sound-event-detection work (Nam et al., "Frequency Dynamic Convolution",
    arXiv:2203.15296) points out that convolution enforces translation
    equivariance along frequency, but frequency is **not** a shift-invariant
    axis -- the same pattern at a different frequency means something different.

    That objection is sharper for tremor than for audio. PD rest tremor sits at
    4-6 Hz and ET at 6-12 Hz, so peak *location* is the single most diagnostic
    quantity, and :class:`Spectrum1DCNN` slides identical filters over every bin
    and therefore cannot represent it.

    The cheapest fix is to hand the network the coordinate: concatenate a
    normalised frequency ramp as a second input channel, so each filter can
    condition on where in the band it is looking.

    This also predicts something already measured -- `DescriptorFusion` helped
    the plain CNN by +0.021, and the descriptors state peak frequency
    explicitly. If the coordinate channel captures the same information, fusion
    should stop helping on top of it.
    """

    def __init__(self, n_bins, num_classes=3, ch=8, dropout=0.3):
        super().__init__()
        self.register_buffer("coord",
                             torch.linspace(-1, 1, n_bins).view(1, 1, n_bins))
        self.conv = nn.Sequential(
            nn.Conv1d(2, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(ch * 2 * 4, num_classes))

    def trunk(self, x):
        """Pooled features. Builds the 2-channel input, so DescriptorFusion
        cannot feed this backbone a bare 1-channel spectrum."""
        z = x.unsqueeze(1)
        z = torch.cat([z, self.coord.expand(z.shape[0], -1, -1)], dim=1)
        return self.conv(z).flatten(1)

    def forward(self, x):
        return self.head(self.trunk(x))


class FreqDynamicCNN(nn.Module):
    """Frequency-dynamic convolution: per-frequency mixing of basis kernels.

    A 1-D adaptation of Frequency Dynamic Convolution (arXiv:2203.15296). Rather
    than one kernel shared across the whole band, ``n_basis`` kernels are run in
    parallel and combined with weights that depend on the frequency bin, so
    different regions of the 3-15 Hz band get different effective filters.

    The mixing weights come from the frequency coordinate alone (not the input),
    which is the deterministic variant -- with 404 patients, input-conditioned
    attention over basis kernels is more capacity than the data supports.
    """

    def __init__(self, n_bins, num_classes=3, ch=8, n_basis=4, dropout=0.3):
        super().__init__()
        self.n_basis = n_basis
        self.basis = nn.Conv1d(1, ch * n_basis, 5, padding=2)
        # per-bin softmax over the basis kernels, learned from position only
        self.mix = nn.Parameter(torch.zeros(n_basis, n_bins))
        self.bn = nn.BatchNorm1d(ch)
        self.post = nn.Sequential(
            nn.Conv1d(ch, ch * 2, 3, padding=1), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(ch * 2 * 4, num_classes))

    def trunk(self, x):
        b, f = x.shape
        z = self.basis(x.unsqueeze(1)).view(b, self.n_basis, -1, f)
        w = torch.softmax(self.mix, dim=0).view(1, self.n_basis, 1, f)
        z = torch.relu(self.bn((z * w).sum(1)))
        return self.post(z).flatten(1)

    def forward(self, x):
        return self.head(self.trunk(x))


class LearnableCompression(nn.Module):
    """Per-band learnable compression, in the spirit of PCEN.

    PCEN (Wang et al., IEEE SPL 2019) replaces the fixed pointwise logarithm of
    a log-mel front-end with a trainable, per-channel compression, and
    outperforms log compression on keyword spotting and bioacoustics. Its
    adaptive-gain term needs a time axis, which patient-averaged spectra do not
    have, so what carries over is the **per-band trainable exponent**:

        y_f = ((x_f + eps) ** alpha_f - eps ** alpha_f) / scale

    with ``alpha_f`` learned per frequency bin. At alpha -> 0 this approaches
    log compression, which is the current fixed choice, so the module can
    recover the existing behaviour and move away from it if the data prefers.

    Expects a RAW (non-log) normalised spectrum.
    """

    def __init__(self, n_bins, init_alpha=0.25):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((n_bins,), float(np.log(init_alpha))))
        self.eps = 1e-8

    def forward(self, x):
        a = torch.exp(self.log_alpha).clamp(1e-3, 2.0)
        return ((x + self.eps) ** a - self.eps ** a) * 10.0


class SpecAugment1D(nn.Module):
    """SpecAugment frequency masking for a 1-D spectrum (training only).

    Park et al.'s SpecAugment masks contiguous bands of the spectrogram. Only
    the frequency-masking half applies to a time-averaged spectrum. Earlier
    augmentation attempts here shifted and added noise to the whole spectrum and
    did nothing; masking is a different operation -- it forces the model not to
    depend on any single band.
    """

    def __init__(self, max_width=3, n_masks=1):
        super().__init__()
        self.max_width, self.n_masks = max_width, n_masks

    def forward(self, x):
        if not self.training or self.max_width < 1:
            return x
        x = x.clone()
        b, f = x.shape
        for _ in range(self.n_masks):
            w = torch.randint(1, self.max_width + 1, (1,)).item()
            s = torch.randint(0, max(f - w, 1), (b,), device=x.device)
            idx = torch.arange(f, device=x.device).view(1, -1)
            m = (idx >= s.view(-1, 1)) & (idx < (s + w).view(-1, 1))
            x = x.masked_fill(m, 0.0)
        return x


class AudioStyleNet(nn.Module):
    """Compose the audio-literature front-end pieces onto a spectrum backbone.

    ``compress`` applies a PCEN-style learnable per-band exponent (expects a raw
    spectrum), ``spec_augment`` masks frequency bands during training, and the
    backbone is any spectrum model -- typically :class:`FreqCoordCNN` or
    :class:`FreqDynamicCNN`, which remove the frequency-translation invariance.
    """

    def __init__(self, backbone, n_bins, compress=False, mask_width=0):
        super().__init__()
        self.compress = LearnableCompression(n_bins) if compress else None
        self.aug = SpecAugment1D(mask_width) if mask_width else None
        self.backbone = backbone

    def forward(self, x):
        if self.compress is not None:
            x = self.compress(x)
        if self.aug is not None:
            x = self.aug(x)
        return self.backbone(x)


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


class SpectrumTransformer(nn.Module):
    """Small transformer over FREQUENCY bins, sized for 404 patients.

    Attention was tested here before only as `BilateralAttention` on 25-patient
    NewData with 61-bin raw spectra, and as `vit_b_16` (85.8 M parameters, AUC
    0.540 -- chance). Neither is a fair test of attention on the CURRENT input:
    16 log-scaled bins, 404 patients.

    This is the fair version. Each frequency bin is a token, with a fixed
    sinusoidal position encoding so the model knows *where* in the 3-15 Hz band
    it is looking -- the property a plain convolution lacks and which
    `FreqCoordCNN` was built to supply.

    Kept at ~10-30 k parameters, the band every model on this cohort has peaked
    in, rather than the 1e6+ of a vision transformer.
    """

    def __init__(self, n_bins, num_classes=3, d=32, n_heads=4, n_layers=2,
                 ff=64, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.register_buffer("pos", self._sinusoidal(n_bins, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, ff, dropout,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    @staticmethod
    def _sinusoidal(n, d):
        pos = torch.arange(n).float().unsqueeze(1)
        i = torch.arange(0, d, 2).float()
        ang = pos / torch.pow(10000.0, i / d)
        pe = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang[:, :pe[:, 1::2].shape[1]])
        return pe

    def trunk(self, x):
        h = self.proj(x.unsqueeze(-1)) + self.pos
        return self.enc(h).mean(1)

    def forward(self, x):
        return self.head(self.trunk(x))


class CrossStreamAttention(nn.Module):
    """Spectrum tokens attend to the instantaneous-frequency trajectory.

    The principled use of attention on this data. The two streams carry
    different things -- the spectrum says WHERE the tremor sits in frequency,
    the trajectory says HOW STEADY it is over time -- and `TwoStreamNet` joins
    them only at the classifier head, so neither can condition on the other.

    Cross-attention lets each frequency bin query the trajectory, which is the
    mechanism the published bilateral-wrist result uses across limbs rather than
    across streams.

    Input packed as ``[spectrum | descriptors | trajectory.flatten()]``.
    """

    def __init__(self, n_spec, n_desc, traj_len, n_traj_ch=2, num_classes=3,
                 d=32, n_heads=4, desc_hidden=16, dropout=0.2):
        super().__init__()
        self.n_spec, self.n_desc = n_spec, n_desc
        self.traj_len, self.n_traj_ch = traj_len, n_traj_ch
        self.sp_proj = nn.Linear(1, d)
        self.tr_proj = nn.Linear(n_traj_ch, d)
        self.register_buffer("sp_pos", SpectrumTransformer._sinusoidal(n_spec, d))
        self.register_buffer("tr_pos", SpectrumTransformer._sinusoidal(traj_len, d))
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.desc = (nn.Sequential(nn.Linear(n_desc, desc_hidden), nn.ReLU(),
                                   nn.Dropout(dropout)) if n_desc else None)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d + (desc_hidden if n_desc else 0), num_classes))

    def forward(self, x):
        i = self.n_spec
        s = x[:, :i]
        dsc = x[:, i:i + self.n_desc] if self.n_desc else None
        t = x[:, i + self.n_desc:].reshape(x.shape[0], self.n_traj_ch,
                                           self.traj_len).transpose(1, 2)
        q = self.sp_proj(s.unsqueeze(-1)) + self.sp_pos      # (B, F, d)
        kv = self.tr_proj(t) + self.tr_pos                   # (B, T, d)
        a, _ = self.attn(q, kv, kv)
        h = self.norm(q + a).mean(1)
        if dsc is not None:
            h = torch.cat([h, self.desc(dsc)], dim=1)
        return self.head(h)
