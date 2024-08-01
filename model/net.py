import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
sys.path.append(parent_dir)
model_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(model_dir)

from axial_atten import AA_kernel
from self_attention import SelfAttentionBlock
from conv_layer import Conv

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(net)
    print('Total number of parameters: %d' % num_params)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1,3,3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                padding=(0,1,1), bias=False, dilation=dilation_rate)

    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                padding=padding, bias=False, dilation=dilation_rate)

def conv2x2(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
            padding=padding, bias=False, dilation=dilation_rate)


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv2x2(inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(inplanes)
        elif norm =='in':
            self.bn1 = nn.InstanceNorm2d(inplanes)
        elif norm =='gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv2x2(planes, planes, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm2d(planes)
        elif norm =='in':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif norm =='gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='in':
                self.shortcut = nn.Sequential(
                    nn.InstanceNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(residue)
        return out

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SEBasicBlock(out_ch, out_ch, kernel_size=(3,3), norm=norm)
    def forward(self, x): 
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out 

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class ReverseAxialAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ReverseAxialAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.out_conv = nn.Conv2d(in_ch, 1, 1, 1)
        # self.out_conv = nn.Sequential(
        #                     nn.Conv2d(in_ch, in_ch//2, 3, 1, 1),
        #                     nn.Conv2d(in_ch//2, 1, 1, 1))
        self.aa_kernel = AA_kernel(out_ch, out_ch)

        self.ra_conv1 = Conv(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv2 = Conv(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv3 = Conv(out_ch,1,3,1,padding=1,bn_acti=True)

    def forward(self, dec_out, enc_out):
        partial_output = self.out_conv(dec_out)
        partial_output_ra = -1*(torch.sigmoid(partial_output)) + 1

        aa_attn = self.aa_kernel(enc_out)
        aa_attn_o = partial_output_ra.expand(-1, self.out_ch, -1, -1).mul(aa_attn)

        ra =  self.ra_conv1(aa_attn_o) 
        ra = self.ra_conv2(ra) 
        ra = self.ra_conv3(ra)

        out = ra + partial_output

        return out

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2, 2), se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm)
        )
        # self.ra_attn = ReverseAxialAttention(in_ch+out_ch, out_ch)

        self.ra_attn = ReverseAxialAttention(in_ch, out_ch)

    def forward(self, x_dec, x_enc):  #x1 from dec and x2 fro encoder
        x_dec = F.interpolate(x_dec, scale_factor=self.scale, mode='nearest')
        # print(f'x_enc:{x_enc.shape} x_dec:{x_dec.shape}')
        out = torch.cat([x_enc, x_dec], dim=1)
        # ra_out = self.ra_attn(out, x_enc)       #with concatenated feature
        ra_out = self.ra_attn(x_dec, x_enc)      #old model --> with only decoder feature
        out = self.conv(out)
        return out, ra_out

class up_nocat(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_nocat, self).__init__()

        self.scale = scale
        self.conv = nn.Sequential(
            conv_block(out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=True)
        out = self.conv(out)

        return out

class literal_conv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, reduction=2, norm='bn'):
        super(literal_conv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, se=se, reduction=reduction, norm=norm)
    def forward(self, x):
        out = self.conv(x)
        return out

class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm2d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm2d(input_num))
            elif norm == 'gn':
                self.add_module('norm_1', nn.GroupNorm(NUM_GROUP, input_num))

        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm2d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm2d(num1))
        elif norm == 'gn':
            self.add_module('norm_2', nn.GroupNorm(NUM_GROUP, num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature
