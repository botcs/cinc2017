import torch as th

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim
import torchvision



class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    
    def forward(self, x):
        return self.scale * F.elu(x, self.alpha)
    
    

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
    def __init__(self, in_channels, channels, dilations, num_ext_features=0,
                 num_classes=3):
        super(FCN, self).__init__()
        self.num_ext_features = num_ext_features
        self.num_classes = num_classes
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

    def forward(self, x, lens=None, mat_features=None):
        if lens is None:
            lens = x.size()[1]
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.pool(out)
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.pool(out)
        # Avg POOLing
        if self.num_ext_features > 0:
            num_features = out.size()[1]
            lens = lens[:, None].expand(len(x), num_features)
            net_features = th.sum(out, dim=-1).squeeze() / lens
            features = th.cat([mat_features[:, :self.num_ext_features], net_features], dim=1)
            out = self.logit(features[:, :, None]).squeeze()
        else:
            lens = lens[:, None].expand(len(x), self.num_classes)
            out = self.logit(out)
            out = th.sum(out, dim=-1).squeeze() / lens
            
        return out


class VGG16(nn.Module):
    def __init__(self, in_channels, channels, dilations, num_classes=3):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
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
        
        self.conv7 = nn.Conv1d(channels[5], channels[6], 9, padding=4,
                               dilation=dilations[6])
        self.bn7 = nn.BatchNorm1d(channels[6])
        
        self.conv8 = nn.Conv1d(channels[6], channels[7], 9, padding=4,
                               dilation=dilations[7])
        self.bn8 = nn.BatchNorm1d(channels[7])
        
        self.conv9 = nn.Conv1d(channels[7], channels[8], 9, padding=4,
                               dilation=dilations[8])
        self.bn9 = nn.BatchNorm1d(channels[8])
        
        self.conv10 = nn.Conv1d(channels[8], channels[9], 9, padding=4,
                               dilation=dilations[9])
        self.bn10 = nn.BatchNorm1d(channels[9])
        
        self.conv11 = nn.Conv1d(channels[9], channels[10], 9, padding=4,
                               dilation=dilations[10])
        self.bn11 = nn.BatchNorm1d(channels[10])
        
        self.conv12 = nn.Conv1d(channels[10], channels[11], 9, padding=4,
                               dilation=dilations[11])
        self.bn12 = nn.BatchNorm1d(channels[11])
        
        self.conv13 = nn.Conv1d(channels[11], channels[12], 9, padding=4,
                               dilation=dilations[12])
        self.bn13 = nn.BatchNorm1d(channels[12])
        
        self.dense1 = nn.Conv1d(channels[12], channels[13], 1)
        self.bn14 = nn.BatchNorm1d(channels[13])
        self.drop1 = nn.Dropout(inplace=True)
        self.dense2 = nn.Conv1d(channels[13], channels[13], 1)
        self.bn15 = nn.BatchNorm1d(channels[13])
        self.drop2 = nn.Dropout(inplace=True)
        self.logit = nn.Conv1d(channels[13], num_classes, 1)

    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = self.pool(out)
        out = F.relu(self.bn8(self.conv8(out)))
        out = F.relu(self.bn9(self.conv9(out)))
        out = F.relu(self.bn10(self.conv10(out)))
        out = self.pool(out)
        out = F.relu(self.bn11(self.conv11(out)))
        out = F.relu(self.bn12(self.conv12(out)))
        out = F.relu(self.bn13(self.conv13(out)))
        out = self.pool(out)
        out = F.relu(self.bn14(self.dense1(out)))
        out = self.drop1(out)
        out = F.relu(self.bn14(self.dense2(out)))
        out = self.drop2(out)
        
        
        # Avg POOLing
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out

class VGG16NoDense(nn.Module):
    def __init__(self, in_channels, channels, use_selu, num_classes=3,
                 dilations=[1, 2,  1, 2,  1, 2, 4,  1, 2, 4,  1, 2, 4]):
        super(VGG16NoDense, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool1d(2)
        self.activation = SELU() if use_selu else nn.ReLU()
        
        self.conv1 = nn.Conv1d(in_channels, channels[0], 17, padding=8,
                               dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        
        self.conv2 = nn.Conv1d(channels[0], channels[1], 9, padding=4,
                               dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(channels[1])
        
        self.conv3 = nn.Conv1d(channels[1], channels[2], 9, padding=4,
                               dilation=dilations[2], bias=False)
        self.bn3 = nn.BatchNorm1d(channels[2])
        
        self.conv4 = nn.Conv1d(channels[2], channels[3], 9, padding=4,
                               dilation=dilations[3], bias=False)
        self.bn4 = nn.BatchNorm1d(channels[3])
        
        self.conv5 = nn.Conv1d(channels[3], channels[4], 9, padding=4,
                               dilation=dilations[4], bias=False)
        self.bn5 = nn.BatchNorm1d(channels[4])
        
        self.conv6 = nn.Conv1d(channels[4], channels[5], 9, padding=4,
                               dilation=dilations[5], bias=False)
        self.bn6 = nn.BatchNorm1d(channels[5])
        
        self.conv7 = nn.Conv1d(channels[5], channels[6], 9, padding=4,
                               dilation=dilations[6], bias=False)
        self.bn7 = nn.BatchNorm1d(channels[6])
        
        self.conv8 = nn.Conv1d(channels[6], channels[7], 9, padding=4,
                               dilation=dilations[7], bias=False)
        self.bn8 = nn.BatchNorm1d(channels[7])
        
        self.conv9 = nn.Conv1d(channels[7], channels[8], 9, padding=4,
                               dilation=dilations[8], bias=False)
        self.bn9 = nn.BatchNorm1d(channels[8])
        
        self.conv10 = nn.Conv1d(channels[8], channels[9], 9, padding=4,
                               dilation=dilations[9], bias=False)
        self.bn10 = nn.BatchNorm1d(channels[9])
        
        self.conv11 = nn.Conv1d(channels[9], channels[10], 9, padding=4,
                               dilation=dilations[10], bias=False)
        self.bn11 = nn.BatchNorm1d(channels[10])
        
        self.conv12 = nn.Conv1d(channels[10], channels[11], 9, padding=4,
                               dilation=dilations[11], bias=False)
        self.bn12 = nn.BatchNorm1d(channels[11])
        
        self.conv13 = nn.Conv1d(channels[11], channels[12], 9, padding=4,
                               dilation=dilations[12], bias=False)
        self.bn13 = nn.BatchNorm1d(channels[12])
        
        self.drop1 = nn.Dropout(inplace=True)
        
        self.logit = nn.Conv1d(channels[12], num_classes, 1)
        
    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(out)
        out = self.activation(self.bn3(self.conv3(out)))
        out = self.activation(self.bn4(self.conv4(out)))
        out = self.pool(out)
        out = self.activation(self.bn5(self.conv5(out)))
        out = self.activation(self.bn6(self.conv6(out)))
        out = self.activation(self.bn7(self.conv7(out)))
        out = self.pool(out)
        out = self.activation(self.bn8(self.conv8(out)))
        out = self.activation(self.bn9(self.conv9(out)))
        out = self.activation(self.bn10(self.conv10(out)))
        out = self.pool(out)
        out = self.activation(self.bn11(self.conv11(out)))
        out = self.activation(self.bn12(self.conv12(out)))
        out = self.activation(self.bn13(self.conv13(out)))
        out = self.drop1(out)
        
        # Avg POOLing
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out   


    
class SkipFCN(nn.Module):
    def __init__(self, in_channels, channels, use_selu=True, num_classes=3,
                 dilations=[1, 2,  1, 2,  1, 2, 4,  1, 2, 4,  1, 2, 4]):
        super(SkipFCN, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool1d(2)
        self.activation = SELU() if use_selu else nn.ReLU()
        
        self.conv1 = nn.Conv1d(in_channels, channels[0], 17, padding=8,
                               dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        
        self.conv2 = nn.Conv1d(channels[0], channels[1], 9, padding=4,
                               dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(channels[1])
        ########## POOL ##########
        self.conv3 = nn.Conv1d(channels[1]*2, channels[2], 9, padding=4,
                               dilation=dilations[2], bias=False)
        self.bn3 = nn.BatchNorm1d(channels[2])
        
        self.conv4 = nn.Conv1d(channels[2], channels[3], 9, padding=4,
                               dilation=dilations[3], bias=False)
        self.bn4 = nn.BatchNorm1d(channels[3])
        ########## POOL ##########
        self.conv5 = nn.Conv1d(channels[3]*2, channels[4], 9, padding=4,
                               dilation=dilations[4], bias=False)
        self.bn5 = nn.BatchNorm1d(channels[4])
        
        self.conv6 = nn.Conv1d(channels[4], channels[5], 9, padding=4,
                               dilation=dilations[5], bias=False)
        self.bn6 = nn.BatchNorm1d(channels[5])
        
        self.conv7 = nn.Conv1d(channels[5], channels[6], 9, padding=4,
                               dilation=dilations[6], bias=False)
        self.bn7 = nn.BatchNorm1d(channels[6])
        ########## POOL ##########
        self.conv8 = nn.Conv1d(channels[6]*2, channels[7], 9, padding=4,
                               dilation=dilations[7], bias=False)
        self.bn8 = nn.BatchNorm1d(channels[7])
        
        self.conv9 = nn.Conv1d(channels[7], channels[8], 9, padding=4,
                               dilation=dilations[8], bias=False)
        self.bn9 = nn.BatchNorm1d(channels[8])
        
        self.conv10 = nn.Conv1d(channels[8], channels[9], 9, padding=4,
                               dilation=dilations[9], bias=False)
        self.bn10 = nn.BatchNorm1d(channels[9])
        ########## POOL ##########
        self.conv11 = nn.Conv1d(channels[9]*2, channels[10], 9, padding=4,
                               dilation=dilations[10], bias=False)
        self.bn11 = nn.BatchNorm1d(channels[10])
        
        self.conv12 = nn.Conv1d(channels[10], channels[11], 9, padding=4,
                               dilation=dilations[11], bias=False)
        self.bn12 = nn.BatchNorm1d(channels[11])
        
        self.conv13 = nn.Conv1d(channels[11], channels[12], 9, padding=4,
                               dilation=dilations[12], bias=False)
        self.bn13 = nn.BatchNorm1d(channels[12])
        
        self.drop1 = nn.Dropout(inplace=True)
        
        self.logit = nn.Conv1d(channels[12], num_classes, 1)
        
    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(out)
        x = self.pool(x)
        out = th.cat([x, out], dim=1)
        out = self.activation(self.bn3(self.conv3(out)))
        out = self.activation(self.bn4(self.conv4(out)))
        out = self.pool(out)
        x = self.pool(x)
        out = th.cat([x, out], dim=1)
        out = self.activation(self.bn5(self.conv5(out)))
        out = self.activation(self.bn6(self.conv6(out)))
        out = self.activation(self.bn7(self.conv7(out)))
        out = self.pool(out)
        x = self.pool(x)
        out = th.cat([x, out], dim=1)
        out = self.activation(self.bn8(self.conv8(out)))
        out = self.activation(self.bn9(self.conv9(out)))
        out = self.activation(self.bn10(self.conv10(out)))
        out = self.pool(out)
        x = self.pool(x)
        out = th.cat([x, out], dim=1)
        out = self.activation(self.bn11(self.conv11(out)))
        out = self.activation(self.bn12(self.conv12(out)))
        out = self.activation(self.bn13(self.conv13(out)))
        out = self.drop1(out)
        
        # Avg POOLing
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out       
    
        
class VGG19NoDense(nn.Module):
    def __init__(self, in_channels, channels, use_selu, num_classes=3,
                 dilations=[1, 2,  1, 2,  1, 2, 4, 4,  1, 2, 4, 4,  1, 2, 4, 4]):
        super(VGG19NoDense, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool1d(2)
        
        self.conv1 = nn.Conv1d(in_channels, channels[0], 17, padding=8,
                               dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.use_selu = use_selu
        # this part is repeating heavily
        layer_list = []
        for i in range(15):
            if i in set([0, 2, 6, 10, 14]):
                layer_list.append(nn.Sequential(*[
                    nn.Conv1d(channels[i], channels[i+1], 9, padding=4,
                          dilation=dilations[i+1], bias=False),
                    nn.BatchNorm1d(channels[i+1]),
                    SELU() if use_selu else nn.ReLU(),
                    nn.MaxPool1d(2)])
                )
            else:
                layer_list.append(nn.Sequential(*[
                    nn.Conv1d(channels[i], channels[i+1], 9, padding=4,
                          dilation=dilations[i+1], bias=False),
                    nn.BatchNorm1d(channels[i+1])],
                    SELU() if use_selu else nn.ReLU())
                )
                
        self.hidden = nn.Sequential(*layer_list)
        
        self.drop1 = nn.Dropout(inplace=True)
        
        self.logit = nn.Conv1d(channels[15], num_classes, 1)
        
    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
            
            
        if self.use_selu:
            out = SELU()(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.hidden(out)
        
        out = self.drop1(out)
        
        # Avg POOLing
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out    

    

    
    
class DilatedBlock(nn.Module):
    # Stg. like:
    # [(3, 3), 16*k]
    def __init__(self, in_channels=8, out_channels=8, kernel_size=9, dilation=2, nonlin=nn.ReLU):
        super(DilatedBlock, self).__init__()
        self.nonlin = nonlin
        self.identity = lambda x: x
        if in_channels != out_channels:
            self.identity = nn.Conv1d(in_channels, out_channels, 1)
        
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size//2, bias=False),
            self.nonlin,
            nn.BatchNorm1d(out_channels),
            nn.Dropout(inplace=True),
            nn.Conv1d(
                out_channels, out_channels, kernel_size,
                padding=kernel_size//2*dilation, bias=False,
                dilation=dilation),
            self.nonlin
        )
        
    def forward(self, x):
        out = self.block(x)
        out = self.identity(x) + out
        return out

class ConvModule(nn.Module):
    # Stg. like: N = 3
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    # [(3, 3), 16*k]
    def __init__(self, in_channel, channel=8, block=DilatedBlock, N=3, nonlin=nn.ReLU, kernel_size=17):
        super(ConvModule, self).__init__()
        residuals = []
        residuals.append(DilatedBlock(in_channel, channel, kernel_size, nonlin=nonlin))
        for n in range(1, N):
            residuals.append(DilatedBlock(channel, channel, kernel_size, nonlin=nonlin))

        self.residuals = nn.Sequential(*residuals)

    def forward(self, x):
        return self.residuals(x)
                       
class ResNet(nn.Module):
    def __init__(self, K_blocks, N_res_in_block, 
                 channel, use_selu, 
                 in_channel=1, init_kernel_size=17, 
                 num_classes=3,
                 pool_after_M_blocks=1):
        super(ResNet, self).__init__()
        self.K_blocks = K_blocks
        self.channel = channel
        self.num_classes = num_classes
        self.pool_after_M_blocks = pool_after_M_blocks
        self.conv_init = nn.Conv1d(
            in_channel, channel, init_kernel_size,
            padding=init_kernel_size//2, bias=False)
        self.bn_init = nn.BatchNorm1d(channel)
        
        if use_selu:
            self.nonlin = SELU()
        else:
            self.nonlin = nn.ReLU()
            
        blocks = [ConvModule(channel, channel, nonlin=self.nonlin, N=N_res_in_block)]
        for K in range(K_blocks):
            blocks.append(ConvModule(
                channel*2**K, channel*2**(K+1), 
                N=N_res_in_block, nonlin=self.nonlin))
            
            if K % pool_after_M_blocks == 0:           
                blocks.append(nn.MaxPool1d(2))
        blocks += [
            nn.Conv1d(channel * 2 ** K_blocks, channel * 2 ** K_blocks, 9, dilation=2),
            nn.BatchNorm1d(channel * 2 ** K_blocks),
            self.nonlin
        ]
        self.net = nn.Sequential(*blocks)
        self.bn_end = nn.BatchNorm1d(channel * 2 ** K_blocks)
        self.logit = nn.Conv1d(channel * 2 ** K_blocks, num_classes, 1, bias=True)
        print(self)

    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.forward_features(x)
        
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out 
    
    def forward_features(self, x):
        out = self.nonlin(self.bn_init(self.conv_init(x)))
        out = self.net(out)

        return out

class EncodeWideResNet(nn.Module):
    def __init__(self, in_channel, init_channel, num_enc_layer, N_res_in_block, use_selu=True, num_classes=3):
        
        super(EncodeWideResNet, self).__init__()
        init_depth = init_channel
        
        if use_selu:
            self.nonlin = SELU()
        else:
            self.nonlin = nn.ReLU()
            
        encoder = [
            nn.Conv1d(in_channel, init_depth, 7, padding=3),
            nn.BatchNorm1d(init_depth),
            self.nonlin,
            nn.MaxPool1d(2)
        ]
        for l in range(0, num_enc_layer-1):
            encoder += [
                nn.Conv1d(init_depth*2**l, init_depth*2**(l + 1), 7, padding=3),
                nn.BatchNorm1d(init_depth*2**(l + 1)),
                self.nonlin,
                nn.MaxPool1d(2)
            ]
        self.encoder = nn.Sequential(*encoder)
        res_init_depth = init_depth*2**(l + 1)
        N = N_res_in_block
        self.resnet = nn.Sequential(
            ConvModule(res_init_depth, res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
            ConvModule(res_init_depth, res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
            ConvModule(res_init_depth, res_init_depth, N, nonlin=self.nonlin, kernel_size=9)
        )

        self.logit = nn.Conv1d(init_depth, num_classes, 1, bias=False)
        self.num_classes = 3
        
        print(self)

    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.forward_features(x)
        
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out 
    
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_resnet(self, x):
        return self.resnet(x)
    
    def forward_features(self, x):
        out = self.encoder(x)
        out = self.resnet(out)

        return out

class SkipResNet(nn.Module):
    def __init__(self, in_channel, init_channel, num_enc_layer, N_res_in_block, use_selu=True, num_classes=3):
        
        super(EncodeWideResNet, self).__init__()
        init_depth = init_channel
        
        if use_selu:
            self.nonlin = SELU()
        else:
            self.nonlin = nn.ReLU()
            
        encoder = [
            nn.Conv1d(in_channel, init_depth, 7, padding=3),
            nn.BatchNorm1d(init_depth),
            self.nonlin,
            nn.MaxPool1d(2)
        ]
        for l in range(0, num_enc_layer-1):
            encoder += [
                nn.Conv1d(init_depth*2**l, init_depth*2**(l + 1), 7, padding=3),
                nn.BatchNorm1d(init_depth*2**(l + 1)),
                self.nonlin,
                nn.MaxPool1d(2)
            ]
        self.encoder = nn.Sequential(*encoder)
        res_init_depth = init_depth*2**(l + 1)
        N = N_res_in_block
        
        self.res1 = ConvModule(res_init_depth, res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
        self.res2 = ConvModule(res_init_depth*2, res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
        self.res3 = ConvModule(res_init_depth*2, res_init_depth, N, nonlin=self.nonlin, kernel_size=9)
        

        self.logit = nn.Conv1d(init_depth, num_classes, 1, bias=False)
        self.num_classes = 3
        
        print(self)

    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.forward_features(x)
        
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out 
    
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_resnet(self, x):
        out = self.res1(x)
        out = self.res2(th.cat([x, out], dim=1))
        out = self.res3(th.cat([x, out], dim=1))
        return out
    
    def forward_features(self, x):
        out = self.encoder(x)
        out = self.resnet(out)

        return out
    
    
class WideResNet(nn.Module):
    def __init__(self, in_channel, init_channel, channel_exponential, 
                 num_enc_layer, N_res_in_block, use_selu=True, num_classes=3):
        
        super(EncodeWideResNet, self).__init__()
        init_depth = init_channel
        
        if use_selu:
            self.nonlin = SELU()
        else:
            self.nonlin = nn.ReLU()
            
        encoder = [
            nn.Conv1d(in_channel, init_depth, 7, padding=3),
            nn.BatchNorm1d(init_depth),
            self.nonlin,
            nn.MaxPool1d(2)
        ]
        for l in range(0, num_enc_layer-1):
            encoder += [
                nn.Conv1d(init_depth*2**l, init_depth*2**(l + 1), 7, padding=3),
                nn.BatchNorm1d(init_depth*2**(l + 1)),
                self.nonlin,
                nn.MaxPool1d(2)
            ]
        self.encoder = nn.Sequential(*encoder)
        res_init_depth = init_depth*2**(l + 1)
        N = N_res_in_block
        if channel_exponential:
            self.resnet = nn.Sequential(
                ConvModule(res_init_depth, 2*res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
                ConvModule(2*res_init_depth, 2*res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
                ConvModule(4*res_init_depth, 8*res_init_depth, N, nonlin=self.nonlin, kernel_size=9)
            )
        else:
            self.resnet = nn.Sequential(
                ConvModule(res_init_depth, 2*res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
                ConvModule(2*res_init_depth, 3*res_init_depth, N, nonlin=self.nonlin, kernel_size=9),
                ConvModule(3*res_init_depth, 4*res_init_depth, N, nonlin=self.nonlin, kernel_size=9)
            )

        self.logit = nn.Conv1d(8*init_depth, num_classes, 1, bias=False)
        self.num_classes = 3
        
        print(self)

    def forward(self, x, lens=None):
        if lens is None:
            lens = x.size()[1]
        out = self.forward_features(x)
        
        lens = lens[:, None].expand(len(x), self.num_classes)
        out = self.logit(out)
        out = th.sum(out, dim=-1).squeeze() / lens
        return out 
    
    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_resnet(self, x):
        return self.resnet(x)
    
    def forward_features(self, x):
        out = self.encoder(x)
        out = self.resnet(out)

        return out
    