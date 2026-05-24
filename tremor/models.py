"""Network architectures for tremor classification.

All models accept input shape ``(B, F, T)`` (batch, features, time) and
output ``(B, num_classes)`` logits. ``F`` is the number of stacked
sensor*frequency channels coming out of the STFT/spectral pipeline.

Use :func:`build_model` (or the ``MODELS`` registry) to instantiate a
model by string name from the ``--model`` CLI flag.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


# ---------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------

class _BiLSTMBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float,
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            bidirectional=bidirectional, batch_first=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.dropout(self.relu(x))


class _ResidualBiLSTMBlock(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float,
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden,
            bidirectional=bidirectional, batch_first=True,
        )
        out_dim = hidden * 2 if bidirectional else hidden
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x):
        residual = x
        out, _ = self.lstm(x)
        out = self.dropout(self.norm(out))
        if self.proj is not None:
            residual = self.proj(residual)
        return out + residual


class _GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.02):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


class _SimpleResidual1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dropout_p: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                               padding=padding, bias=False)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.dropout(self.norm2(self.conv2(out)))
        return self.relu(out + identity)


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class TremorBiLSTM(nn.Module):
    """Original 4-stack BiLSTM port from MATLAB ``RNNv4.m`` (default model)."""

    def __init__(self, input_size: int, num_classes: int = 3,
                 hidden: int = 300, num_layers: int = 4, dropout: float = 0.4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.blocks = nn.ModuleList()
        in_dim = input_size
        for _ in range(num_layers):
            self.blocks.append(
                nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
            )
            in_dim = hidden * 2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_bn = nn.BatchNorm1d(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)
        for i, lstm in enumerate(self.blocks):
            x, _ = lstm(x)
            if i < len(self.blocks) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.out_bn(x)
        x = self.dropout(x)
        return self.fc(x)


class BiLSTM(nn.Module):
    """Stacked BiLSTMBlocks + final BiLSTM head + FC stack (Drive port)."""

    def __init__(self, input_size: int, num_classes: int = 3,
                 hidden: int = 128, hidden2: int = 64, num_layers: int = 2,
                 dropout: float = 0.4, n_fc_layers: int = 1,
                 fc_hidden: int = 128):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size, eps=1e-5)
        blocks = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden * 2
            blocks.append(_BiLSTMBlock(in_dim, hidden, dropout))
        self.bi_stack = nn.Sequential(*blocks)
        self.lstm_final = nn.LSTM(
            input_size=hidden * 2, hidden_size=hidden2,
            bidirectional=True, batch_first=True,
        )
        self.out_bn = nn.BatchNorm1d(hidden2 * 2)
        self.dropout = nn.Dropout(dropout)
        fc_layers = []
        for i in range(n_fc_layers):
            in_d = hidden2 * 2 if i == 0 else fc_hidden
            fc_layers.append(nn.Linear(in_d, fc_hidden))
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)
        x = self.bi_stack(x)
        x, _ = self.lstm_final(x)
        x = x.mean(dim=1)
        x = self.out_bn(x)
        x = self.dropout(x)
        x = self.fc_stack(x)
        return self.fc(x)


class LSTM(nn.Module):
    """Single-layer unidirectional LSTM + FC head (Drive port)."""

    def __init__(self, input_size: int, num_classes: int = 3,
                 hidden: int = 128, dropout: float = 0.4,
                 n_fc_layers: int = 1, fc_hidden: int = 128):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size, eps=1e-5)
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden,
            bidirectional=False, batch_first=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        fc_layers = []
        for i in range(n_fc_layers):
            in_d = hidden if i == 0 else fc_hidden
            fc_layers.append(nn.Linear(in_d, fc_hidden))
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.dropout(x.mean(dim=1))
        x = self.fc_stack(x)
        return self.fc(x)


class GRU(nn.Module):
    """Single-layer unidirectional GRU + FC head (Drive port)."""

    def __init__(self, input_size: int, num_classes: int = 3,
                 hidden: int = 128, dropout: float = 0.4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size, eps=1e-5)
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden,
            bidirectional=False, batch_first=True,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.relu(x)
        x = self.dropout(x.mean(dim=1))
        return self.fc(x)


class ResNetBiLSTM(nn.Module):
    """Stacked residual BiLSTM blocks + final BiLSTM head (Drive port)."""

    def __init__(self, input_size: int, num_classes: int = 3,
                 hidden: int = 128, hidden2: int = 64, num_layers: int = 3,
                 dropout: float = 0.3, n_fc_layers: int = 1,
                 fc_hidden: int = 128):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size, eps=1e-5)
        blocks = []
        for i in range(num_layers):
            in_d = input_size if i == 0 else hidden * 2
            blocks.append(_ResidualBiLSTMBlock(in_d, hidden, dropout))
        self.lstm_stack = nn.Sequential(*blocks)
        self.lstm_final = nn.LSTM(
            input_size=hidden * 2, hidden_size=hidden2,
            bidirectional=True, batch_first=True,
        )
        self.final_norm = nn.LayerNorm(hidden2 * 2)
        self.dropout = nn.Dropout(dropout)
        fc_layers = []
        for i in range(n_fc_layers):
            in_d = hidden2 * 2 if i == 0 else fc_hidden
            fc_layers.append(nn.Linear(in_d, fc_hidden))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)
        x = self.lstm_stack(x)
        x, _ = self.lstm_final(x)
        x = self.final_norm(x)
        x = self.dropout(x.mean(dim=1))
        x = self.fc_stack(x)
        return self.fc(x)


class ResTCN(nn.Module):
    """1D residual TCN with Gaussian noise injection (Drive port).

    Treats features as Conv1D channels and time as the spatial axis, so
    the input shape ``(B, F, T)`` is consumed directly.
    """

    def __init__(self, input_size: int, num_classes: int = 3,
                 base_filters: int = 32, dropout: float = 0.3,
                 noise_sigma: float = 0.02):
        super().__init__()
        self.input_size = input_size
        self.noise = _GaussianNoise(noise_sigma)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, base_filters, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.layer2 = _SimpleResidual1D(
            base_filters, base_filters * 2, 5, stride=2, padding=2,
            dropout_p=dropout,
        )
        self.layer3 = _SimpleResidual1D(
            base_filters * 2, base_filters * 4, 3, 1, 1, dropout_p=dropout,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 4, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.noise(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class CustomAST(nn.Module):
    """Audio Spectrogram Transformer initialised from a pretrained DeiT.

    Treats ``(B, F, T)`` as a single-channel 2-D image and patches it
    with a strided Conv2d, then runs the standard ViT blocks. Requires
    the ``timm`` package and downloads the pretrained checkpoint on
    first run.
    """

    def __init__(self, input_size: int, num_classes: int = 3,
                 input_tdim: int = 512, fstride: int = 10, tstride: int = 10,
                 model_name: str = "deit_base_distilled_patch16_384",
                 pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "CustomAST requires the 'timm' package. Install with: pip install timm"
            ) from e

        self.input_fdim = input_size
        self.input_tdim = input_tdim

        self.v = timm.create_model(model_name, pretrained=pretrained)

        embed_dim = self.v.pos_embed.shape[2]
        original_num_patches = self.v.patch_embed.num_patches
        original_hw = int(original_num_patches ** 0.5)

        new_proj = nn.Conv2d(1, embed_dim, kernel_size=(16, 16),
                             stride=(fstride, tstride))
        original_weights = self.v.patch_embed.proj.weight
        new_proj.weight = nn.Parameter(
            torch.sum(original_weights, dim=1, keepdim=True)
        )
        new_proj.bias = self.v.patch_embed.proj.bias

        self.v.patch_embed = _ASTPatchEmbed(new_proj)
        f_dim, t_dim = self._get_grid_shape(
            fstride, tstride, input_size, input_tdim, embed_dim
        )
        self.v.patch_embed.num_patches = f_dim * t_dim

        pos_embed = self.v.pos_embed[:, 2:, :]
        pos_embed = pos_embed.reshape(
            1, original_hw, original_hw, embed_dim
        ).permute(0, 3, 1, 2)

        if t_dim <= original_hw:
            start = original_hw // 2 - t_dim // 2
            pos_embed = pos_embed[:, :, :, start:start + t_dim]
        else:
            pos_embed = nn.functional.interpolate(
                pos_embed, size=(original_hw, t_dim),
                mode="bilinear", align_corners=False,
            )
        if f_dim <= original_hw:
            start = original_hw // 2 - f_dim // 2
            pos_embed = pos_embed[:, :, start:start + f_dim, :]
        else:
            pos_embed = nn.functional.interpolate(
                pos_embed, size=(f_dim, t_dim),
                mode="bilinear", align_corners=False,
            )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        cls_dist = self.v.pos_embed[:, :2, :]
        self.v.pos_embed = nn.Parameter(torch.cat([cls_dist, pos_embed], dim=1))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @staticmethod
    def _get_grid_shape(fstride, tstride, fdim, tdim, embed_dim):
        test = torch.zeros(1, 1, fdim, tdim)
        proj = nn.Conv2d(1, embed_dim, kernel_size=(16, 16),
                         stride=(fstride, tstride))
        out = proj(test)
        return out.shape[2], out.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.v.patch_embed(x)

        cls = self.v.cls_token.expand(x.shape[0], -1, -1)
        dist = self.v.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, dist, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return self.mlp_head(x)


class _ASTPatchEmbed(nn.Module):
    def __init__(self, proj: nn.Conv2d):
        super().__init__()
        self.proj = proj
        self.num_patches = 0  # set externally

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------

MODELS: dict[str, Callable[..., nn.Module]] = {
    "tremor_bilstm": TremorBiLSTM,
    "bilstm": BiLSTM,
    "lstm": LSTM,
    "gru": GRU,
    "resbilstm": ResNetBiLSTM,
    "restcn": ResTCN,
    "ast": CustomAST,
}


def build_model(
    name: str,
    input_size: int,
    num_classes: int,
    target_T: int | None = None,
    hidden: int = 128,
    dropout: float = 0.4,
    ast_pretrained: bool = True,
) -> nn.Module:
    """Instantiate a model by registry name with sensible defaults."""
    if name not in MODELS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODELS)}"
        )
    cls = MODELS[name]
    if name in ("tremor_bilstm", "bilstm", "lstm", "gru", "resbilstm"):
        return cls(input_size=input_size, num_classes=num_classes,
                   hidden=hidden, dropout=dropout)
    if name == "restcn":
        return cls(input_size=input_size, num_classes=num_classes,
                   base_filters=hidden, dropout=dropout)
    if name == "ast":
        if target_T is None:
            raise ValueError("AST needs target_T (the time-frame count after padding).")
        return cls(input_size=input_size, num_classes=num_classes,
                   input_tdim=target_T, pretrained=ast_pretrained)
    raise AssertionError(f"unhandled model name: {name}")
