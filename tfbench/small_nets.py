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


def fit_predict_proba(model_fn, Xtr, ytr, Xte, num_classes=3, epochs=300,
                      lr=3e-3, wd=1e-3, seed=0, class_weight=True, device="cpu"):
    """Multiclass counterpart of :func:`fit_predict`; returns full probabilities.

    ``fit_predict`` slices ``[:, 1]`` and so only works on a binary axis. The
    three-cohort N/PD/ET problem needs the whole simplex to compute per-class
    precision, which is the metric that actually matters here -- macro F1 hides
    an ET column that is often 0.1-0.2.
    """
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    m = model_fn().to(device)
    w = None
    if class_weight:
        cnt = np.bincount(ytr, minlength=num_classes).astype(float)
        w = torch.tensor(cnt.sum() / (num_classes * np.maximum(cnt, 1)),
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
        return torch.softmax(m(Xte_t), 1).cpu().numpy()


def kfold_proba(X, y, groups, model_fn, num_classes=3, k=5, seeds=(0, 1, 2),
                epochs=200, class_weight=True, **kw):
    """Patient-disjoint stratified k-fold; probabilities averaged over seeds.

    The scaler is fit on each fold's TRAIN split only. k-fold rather than LOSO
    because the deep models are refit per seed per fold and LOSO at ~250
    patients is 750 fits per architecture.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    prob = np.zeros((len(y), num_classes))
    cv = StratifiedGroupKFold(k, shuffle=True, random_state=0)
    for tr, te in cv.split(X, y, groups=groups):
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-8
        prob[te] = np.mean([fit_predict_proba(
            model_fn, (X[tr] - mu) / sd, y[tr], (X[te] - mu) / sd,
            num_classes=num_classes, seed=s, epochs=epochs,
            class_weight=class_weight, **kw) for s in seeds], 0)
    return prob


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


def bilateral_table(recs, side_of, ch=slice(3, 6), fs=100.0, f_lo=3.0,
                    f_hi=15.0, nperseg=512):
    """(patients, 2*F) left|right spectra, for :class:`BilateralAttention`.

    ``side_of(rec) -> "left" | "right" | None``. A patient missing one limb is
    dropped rather than zero-filled: a zero spectrum is not "no tremor", it is
    an out-of-distribution input, and with 6 ET subjects one such row moves the
    metric.
    """
    from collections import defaultdict
    from scipy.signal import welch
    rows, lab = defaultdict(lambda: {"left": [], "right": []}), {}
    for r in recs:
        s = side_of(r)
        if s is None:
            continue
        x = r.x[ch] if r.x.shape[0] > 3 else r.x
        f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
        P = P.mean(0)
        k = (f >= f_lo) & (f <= f_hi)
        v = P[k]
        rows[r.subject][s].append(v / (v.sum() + 1e-20))
        lab[r.subject] = r.y
    pats = [p for p in sorted(rows) if rows[p]["left"] and rows[p]["right"]]
    X = np.array([np.concatenate([np.mean(rows[p]["left"], 0),
                                  np.mean(rows[p]["right"], 0)]) for p in pats])
    return (np.nan_to_num(X), np.array([lab[p] for p in pats]), np.array(pats))


#: Names of the six left-right asymmetry descriptors, in column order.
ASYM_NAMES = ("corr", "cos", "peak_df", "log_peak_ratio", "log_power_ratio", "l1")


def asym_feats(Xb):
    """Explicit left-right interaction from a ``bilateral_table`` matrix.

    This is the hand-coded counterpart to what :class:`BilateralAttention`
    would have to learn: six numbers describing how the two limbs' spectra
    differ, rather than a 2F-token attention map over them.

    The clinical premise is real -- PD signs begin unilaterally and stay more
    severe on that side, while ET is typically more symmetric -- and it is
    cheap to state directly:

    ``corr``            correlation of the two mean-centred spectral shapes
    ``cos``             cosine similarity of the raw shapes
    ``peak_df``         |peak-bin difference| between limbs
    ``log_peak_ratio``  log ratio of peak heights
    ``log_power_ratio`` log ratio of total in-band power
    ``l1``              L1 distance between the two shapes

    Takes ``(patients, 2F)`` as produced by :func:`bilateral_table`; returns
    ``(patients, 6)``.
    """
    f = Xb.shape[1] // 2
    L, R = Xb[:, :f], Xb[:, f:]
    eps = 1e-12
    Lc, Rc = L - L.mean(1, keepdims=True), R - R.mean(1, keepdims=True)
    corr = ((Lc * Rc).sum(1)
            / (np.linalg.norm(Lc, axis=1) * np.linalg.norm(Rc, axis=1) + eps))
    cos = ((L * R).sum(1)
           / (np.linalg.norm(L, axis=1) * np.linalg.norm(R, axis=1) + eps))
    pk = np.abs(L.argmax(1) - R.argmax(1)).astype(float)
    hi = np.log((L.max(1) + eps) / (R.max(1) + eps))
    pw = np.log((L.sum(1) + eps) / (R.sum(1) + eps))
    l1 = np.abs(L - R).sum(1)
    return np.column_stack([corr, cos, pk, hi, pw, l1])


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
    shape of the averaged spectrum. `tfbench.stability.if_trajectory` produces
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
                 dropout=0.3):
        super().__init__()
        self.n_spec, self.n_desc, self.traj_len = n_spec, n_desc, traj_len
        self.spec, self.spec_feat_fn = spec_backbone, spec_feat_fn
        self.traj = TrajectoryEncoder(2, traj_dim)
        self.desc = (nn.Sequential(nn.Linear(n_desc, desc_hidden), nn.ReLU(),
                                   nn.Dropout(dropout)) if n_desc else None)
        d = spec_dim + traj_dim + (desc_hidden if n_desc else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        i = self.n_spec
        s = x[:, :i]
        d = x[:, i:i + self.n_desc] if self.n_desc else None
        t = x[:, i + self.n_desc:].reshape(x.shape[0], 2, self.traj_len)
        parts = [self.spec_feat_fn(self.spec, s), self.traj(t)]
        if d is not None:
            parts.append(self.desc(d))
        return self.head(torch.cat(parts, dim=1))
