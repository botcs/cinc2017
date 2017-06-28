import torch as th

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim
import torchvision


class BasicBlock(nn.Module):
    # Stg. like:
    # [(3, 3), 16*k]
    def __init__(self, in_channels=8, out_channels=8, kernel_size=7):
        super(BasicBlock, self).__init__()

        self.identity = lambda x: x
        if in_channels != out_channels:
            self.identity = nn.Conv1d(in_channels, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size//2, bias=False)
        self.dropout = nn.Dropout(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size//2, bias=False)

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        self.dropout(h)
        h = self.conv2(F.relu(self.bn2(h)))

        out = self.identity(x) + h
        return out


class ConvModule(nn.Module):
    # Stg. like: N = 3
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    def __init__(self, in_channels=8, channels=8, N=1):
        super(ConvModule, self).__init__()
        residuals = [BasicBlock(in_channels, channels)]
        for n in range(1, N):
            residuals.append(BasicBlock(channels, channels))

        self.residuals = nn.Sequential(*residuals)

    def forward(self, x):
        return self.residuals(x)
