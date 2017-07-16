import torch as th

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim
import torchvision




class MultiKernelBlock(nn.Module):
    # Mentioned here: https://arxiv.org/pdf/1611.06455.pdf
    '''def cuda(self, *args, **kwargs):
    super(MultiKernelBlock, self).cuda(*args, **kwargs)
    for conv in self.kernel_list:
        conv.cuda(*args, **kwargs)
        return self'''

    def __init__(self, in_channels, out_channels,
                 kernel_sizes=[9, 5, 3], dilations=[9, 5, 3]):
        super(MultiKernelBlock, self).__init__()

        self.kernel_list = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            self.kernel_list.append(nn.Conv1d(
                in_channels, out_channels, k,
                padding=k//2*d, bias=False, dilation=d)) # k-dilation is just a wild guess

    def forward(self, x):
        act = [conv(x) for conv in self.kernel_list]
        out = th.sum(th.stack(act), dim=0).squeeze(0)
        return out



class DilatedFCN(nn.Module):
    def __init__(self, in_channels, channels, dilations,
                 num_classes=3):
        super(DilatedFCN, self).__init__()

        self.conv1 = MultiKernelBlock(in_channels, channels[0],
                                      dilations=dilations[0])
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = MultiKernelBlock(channels[0], channels[1],
                                      dilations=dilations[1])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv3 = MultiKernelBlock(channels[1], channels[2],
                                      dilations=dilations[2])
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.logit = nn.Conv1d(channels[2], num_classes, 1)

    def forward(self, x, lens):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        # Avg POOLing
        num_features = out.size()[1]
        lens = lens[:, None].expand(len(x), num_features)
        out = th.sum(out, dim=-1).squeeze() / lens
        return self.logit(out[:, :, None]).squeeze()

    def forward_conv(self, x, softmax=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        if softmax:
            return F.softmax(self.logit(out))
        return self.logit(out)

class FCN(nn.Module):
    def __init__(self, in_channels, channels, dilations, num_ext_features=222,
                 num_classes=3):
        super(FCN, self).__init__()

        self.pool = nn.MaxPool1d(2)
        
        self.conv1 = nn.Conv1d(in_channels, channels[0], 17, padding=8,
                                      dilation=dilations[0])
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], 9, padding=4,
                                      dilation=dilations[1])
        self.bn2 = nn.BatchNorm1d(channels[1])
        
        
        self.conv3 = nn.Conv1d(channels[1], channels[2], 9, padding=4,
                                      dilation=dilations[2])
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.conv4 = nn.Conv1d(channels[2], channels[3], 9, padding=4,
                                      dilation=dilations[3])
        self.bn4 = nn.BatchNorm1d(channels[3])
        
        
        self.conv5 = nn.Conv1d(channels[3], channels[4], 9, padding=4,
                                      dilation=dilations[4])
        self.bn5 = nn.BatchNorm1d(channels[4])
        self.conv6 = nn.Conv1d(channels[4], channels[5], 9, padding=4,
                                      dilation=dilations[5])
        self.bn6 = nn.BatchNorm1d(channels[5])
        
        self.logit = nn.Conv1d(channels[5] + num_ext_features, num_classes, 1)

    def forward(self, x, lens, mat_features):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.pool(out)
        # Avg POOLing
        num_features = out.size()[1]
        lens = lens[:, None].expand(len(x), num_features)
    
        net_features = th.sum(out, dim=-1).squeeze() / lens
        features = th.cat([mat_features, net_features], dim=1)
        out = self.logit(features[:, :, None]).squeeze()

        return out

    
    
    
class DilatedBlock(nn.Module):
    # Stg. like:
    # [(3, 3), 16*k]
    def __init__(self, in_channels=8, out_channels=8, kernel_size=7, dilation=2):
        super(DilatedBlock, self).__init__()

        self.identity = lambda x: x
        if in_channels != out_channels:
            self.identity = nn.Conv1d(in_channels, out_channels, 1)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size//2*dilation, bias=False,
            dilation=dilation)

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(self.dropout(F.relu(self.bn2(h))))
        out = self.identity(x) + h
        return out

class ConvModule(nn.Module):
    # Stg. like: N = 3
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    def __init__(self, in_channel, channel=8, block=DilatedBlock, N=32):
        super(ConvModule, self).__init__()
        residuals = []
        residuals.append(DilatedBlock(in_channel, channel, 17))
        for n in range(1, N):
            residuals.append(DilatedBlock(channel, channel, 17))

        self.residuals = nn.Sequential(*residuals)

    def forward(self, x):
        return self.residuals(x)
                       
class ResNet(nn.Module):
    def __init__(self, N_blocks, 
                 channel, in_channel=1, kernel_size=32, 
                 init_dilation=2, num_classes=3, num_ext_features=222,
                 pool_after_M_blocks=1):
        super(ResNet, self).__init__()
        self.N_blocks = N_blocks
        self.channel = channel
        self.pool_after_M_blocks = pool_after_M_blocks
        self.init_dilation = init_dilation
        self.kernels_size = kernel_size
        self.conv_init = nn.Conv1d(
            in_channel, channel, kernel_size,
            padding=kernel_size//2*init_dilation, 
            dilation=init_dilation, bias=False)
        self.bn_init = nn.BatchNorm1d(channel)
        blocks = [ConvModule(channel, channel)]
        for N in range(1, N_blocks):
            print('generating block:', N)
            blocks.append(ConvModule(channel*N, channel*(N+1), N=4))
            #print(blocks[-1])
            if N % pool_after_M_blocks == 0:           
                blocks.append(nn.MaxPool1d(2))
        self.net = nn.Sequential(*blocks)
        self.bn_end = nn.BatchNorm1d(channel * N_blocks)
        self.logit = nn.Conv1d(channel * N_blocks + num_ext_features, 
                               num_classes, 1, bias=True)
        

    def forward(self, x, lens, mat_features):
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.net(out)
        num_features = self.N_blocks * self.channel
        lens = lens[:, None].expand(len(x), num_features)
        net_features = th.sum(out, dim=-1).squeeze() / lens
        features = th.cat([mat_features, net_features], dim=1)
        out = self.logit(features[:, :, None]).squeeze()

        return out
    
    def forward_features(self, x):
        out = F.relu(self.bn_init(self.conv_init(x)))
        out = self.net(out)

        return out


class WideResNet(nn.Module):
    def __init__(self, in_channels=32, k=2, N=3, num_classes=3):
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
        out = th.sum(out, dim=-1).squeeze() / lens

        return self.logit(out[:, :, None]).squeeze()

    def forward_conv(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return self.logit(out)
