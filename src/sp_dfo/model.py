from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branch_1 = self.conv1(inputs)
        branch_2 = self.conv2(inputs)
        branch_3 = self.conv3(inputs)
        outputs = torch.cat([branch_1, branch_2, branch_3], dim=1)
        return self.relu(outputs)


class Chrononet(nn.Module):
    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels)
        self.conv_block2 = ConvBlock(96)
        self.conv_block3 = ConvBlock(96)
        self.gru1 = nn.GRU(96, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.linear = nn.Linear(1875 * 2, 1)
        self.gru4 = nn.GRU(96, 32, batch_first=True)
        self.last_lin = nn.Linear(32, 2)

    def forward(
        self,
        inputs: torch.Tensor,
        return_emb: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        outputs = self.conv_block1(inputs)
        outputs = self.conv_block2(outputs)
        outputs = self.conv_block3(outputs)

        outputs = outputs.permute(0, 2, 1)
        gru_1, _ = self.gru1(outputs)
        gru_2, _ = self.gru2(gru_1)
        outputs = torch.cat([gru_1, gru_2], dim=2)
        gru_3, _ = self.gru3(outputs)
        outputs = torch.cat([gru_1, gru_2, gru_3], dim=2)

        outputs = outputs.permute(0, 2, 1)
        outputs = self.linear(outputs)
        outputs = F.relu(outputs)
        outputs = outputs.permute(0, 2, 1)

        embeddings, _ = self.gru4(outputs)
        embeddings = embeddings.flatten(1, 2)
        logits = self.last_lin(embeddings)

        if return_emb:
            return logits, embeddings

        return logits


__all__ = ["Chrononet"]
