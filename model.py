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


class MultiKernelBlock(nn.Module):
    # Mentioned here: https://arxiv.org/pdf/1611.06455.pdf

    def __init__(self, in_channels, out_channels, kernel_sizes=[8, 5, 3]):
        super(MultiKernelBlock, self).__init__()

        self.kernel_list = []
        for k in kernel_sizes:
            self.kernel_list.append(nn.Conv1d(
                in_channels, out_channels, k,
                padding=kernel_size//2, bias=False))

    def forward(self, x):
        y = self.kernel_list[0](x)
        for conv in self.kernel_list[1:]:
            y += conv(x)
        return y


class ConvModule(nn.Module):
    # Stg. like: N = 3
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    def __init__(self, block=BasicBlock, in_channels=8, channels=8, N=1,
                 num_classes=3):
        super(ConvModule, self).__init__()
        residuals = [BasicBlock(in_channels, channels)]
        for n in range(1, N):
            residuals.append(BasicBlock(channels, channels))

        self.residuals = nn.Sequential(*residuals)

    def forward(self, x):
        return self.residuals(x)


class BaseLineFCN(nn.Module):
    def __init__(self, in_channels, channels=[128, 256, 128], num_classes=3):
        super(BaseLineFCN, self).__init__()

        self.conv1 = MultiKernelBlock(in_channels, channels[0])
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = MultiKernelBlock(channels[0], channels[1])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv3 = MultiKernelBlock(channels[1], channels[2])
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.logit = nn.Conv1d(channels[2], num_classes, 1)

    def forward(self, x, lens):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        out = F.relu(self.bn3(self.conv3(x)))
        # Avg POOLing
        num_features = out.size()[1]
        lens = lens[:, None].expand(len(x), num_features)
        out = torch.sum(out, dim=-1).squeeze() / lens
        return self.logit(out[:, :, None]).squeeze()

    def forward_conv(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))

        return F.softmax(self.logit(out))


class ResNet(nn.Module):
    def __init__(self, in_channels=32, channels=8, kernel_size=7,
                 num_classes=3):
        super(ResNet, self).__init__()
        self.conv_init = nn.Conv1d(
            in_channels, channels, kernel_size, padding=kernel_size//2)
        self.bn_init = nn.BatchNorm1d(in_channels)
        self.res1 = BasicBlock(channels, channels)
        self.res2 = BasicBlock(channels, channels)
        self.res3 = BasicBlock(channels, channels)
        self.logit = nn.Conv1d(channels, num_classes, 1, bias=True)

    def forward(self, x, lens):
        out = F.relu(self.conv_init(self.bn_init(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        num_features = out.size()[1]
        lens = lens[:, None].expand(len(x), num_features)
        features = torch.sum(out, dim=-1).squeeze() / lens
        out = self.logit(features[:, :, None]).squeeze()

        return out

    def forward_conv(self, x):
        out = F.relu(self.conv_init(self.bn_init(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = F.softmax(self.bn_out(self.logit(out)))

        return out


class WideResNet(nn.Module):
    def __init__(self, in_channels=32, k=2, N=1, num_classes=3):
        super(WideResNet, self).__init__()
        init_depth = 8
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, init_depth*k, 7, padding=3),
            nn.BatchNorm1d(init_depth*k),
            F.relu(inplace=True)
        )

        self.conv2 = ConvModule(init_depth*k, 2*init_depth*k, N)
        self.conv3 = ConvModule(2*init_depth*k, 4*init_depth*k, N)
        self.conv4 = ConvModule(4*init_depth*k, 8*init_depth*k, N)

        self.logit = nn.Conv1d(8*init_depth*k, num_classes, 1, bias=False)

    def forward(self, x, lens):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        num_features = out.size()[1]
        lens = lens[:, None].expand(len(x), num_features)
        out = torch.sum(out, dim=-1).squeeze() / lens

        return self.logit(out[:, :, None]).squeeze()

    def forward_conv(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return self.logit(out)
