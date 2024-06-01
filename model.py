import torch
from torch import nn
from torch.nn import functional as F

class Conv_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.relu(x)
        return x

class Chrononet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv_block1 = Conv_block(6)
        self.conv_block2 = Conv_block(96)
        self.conv_block3 = Conv_block(96)
        self.gru1 = nn.GRU(96, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.linear = nn.Linear(1875*2, 1)
        self.gru4 = nn.GRU(96, 32, batch_first=True)
        self.last_lin = nn.Linear(32, 2)

    def forward(self, x, return_emb=False):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1)
        x1, _ = self.gru1(x)
        x2, _ = self.gru2(x1)
        x = torch.cat([x1, x2], dim=2)
        x3, _ = self.gru3(x)
        x = torch.cat([x1, x2, x3], dim=2)
        x4 = x.permute(0, 2, 1)
        x = self.linear(x4)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x4, _ = self.gru4(x)
        x4 = x4.flatten(1, 2)
        out = self.last_lin(x4)
        if return_emb:
            return out, x4
        else:
            return out