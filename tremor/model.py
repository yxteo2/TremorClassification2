"""BiLSTM tremor classifier (PyTorch).

Matches the architecture from ``trainingFunctions/RNNv4.m``:
four stacked BiLSTM blocks, ReLU + dropout between blocks, batch-norm on
the input and on the last block's output, FC head.
"""

from __future__ import annotations

import torch
from torch import nn


class TremorBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        hidden: int = 300,
        num_layers: int = 4,
        dropout: float = 0.4,
    ) -> None:
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
        # x: (B, C, T) → BN over channels → (B, T, C) for LSTM
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)

        for i, lstm in enumerate(self.blocks):
            x, _ = lstm(x)
            if i < len(self.blocks) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        x = x[:, -1, :]
        x = self.out_bn(x)
        x = self.dropout(x)
        return self.fc(x)
