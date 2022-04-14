import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from option import args
from utils.utils import *


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=True):
        super(ConvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


# Dual-attention Module
class DAM(nn.Module):
    def __init__(self):
        super(DAM, self).__init__()
        self.n_ch = args.num_features

        # Channel attention
        self.c_conv_1 = ConvLayer(self.n_ch, self.n_ch, kernel_size=1, padding=0)
        self.c_conv_2 = ConvLayer(self.n_ch, self.n_ch, kernel_size=1, padding=0)
        self.c_act = nn.Sigmoid()

        # Spatial attention
        self.s_conv_1 = ConvLayer(2 * self.n_ch, self.n_ch)
        self.s_conv_2 = nn.Sequential(
            nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_i, x_r):
        ca = F.adaptive_avg_pool2d(x_i, (1, 1))
        ca = self.c_conv_1(ca)
        ca = self.c_conv_2(ca)
        ca = self.c_act(ca)

        sa = torch.cat((x_i, x_r), dim=1)
        sa = self.s_conv_1(sa)
        sa = self.s_conv_2(sa)

        att_map = ca * sa
        return att_map


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class DAHDRNet(nn.Module):
    def __init__(self):
        super(DAHDRNet, self).__init__()
        self.in_ch = 2 * args.num_channels
        self.out_ch = args.num_channels
        self.num_feat = args.num_features
        self.drdb_layers = args.num_layers
        self.drdb_growth = args.growth

        # Dual Attention Network
        self.conv_1 = ConvLayer(self.in_ch, self.num_feat)
        self.conv_2 = ConvLayer(self.in_ch, self.num_feat)
        self.conv_3 = ConvLayer(self.in_ch, self.num_feat)

        self.conv_21 = ConvLayer(self.num_feat, self.num_feat)
        self.conv_22 = ConvLayer(self.num_feat, self.num_feat)
        self.conv_23 = ConvLayer(self.num_feat, self.num_feat)

        self.dam = DAM()

        # Merging Network
        self.f0 = ConvLayer(3 * self.num_feat, self.num_feat)
        self.drdb1 = DRDB(self.num_feat, self.drdb_layers, self.drdb_growth)
        self.drdb2 = DRDB(self.num_feat, self.drdb_layers, self.drdb_growth)
        self.drdb3 = DRDB(self.num_feat, self.drdb_layers, self.drdb_growth)
        self.f5 = ConvLayer(3 * self.num_feat, self.num_feat)
        self.f6 = ConvLayer(self.num_feat, self.num_feat)
        self.f7 = ConvLayer(self.num_feat, self.out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.conv_1(x1)
        x2 = self.conv_2(x2)
        x3 = self.conv_3(x3)

        x1_1 = self.dam(x1, x2) * x1
        x2_1 = self.conv_21(x2)
        x3_1 = self.dam(x3, x2) * x3

        x1_2 = self.dam(x1_1, x2_1) * x1_1
        x2_2 = self.conv_22(x2_1)
        x3_2 = self.dam(x3_1, x2_1) * x3_1

        x1_3 = self.dam(x1_2, x2_2) * x1_2
        x2_3 = self.conv_23(x2_2)
        x3_3 = self.dam(x3_2, x2_2) * x3_2

        x_cat = torch.cat((x1_3, x2_3, x3_3), dim=1)
        x_cat = self.f0(x_cat)
        f1 = self.drdb1(x_cat)
        f2 = self.drdb2(f1)
        f3 = self.drdb3(f2)
        f_cat = torch.cat((f1, f2, f3), dim=1)
        f_cat = self.f5(f_cat)
        f_res = f_cat + x2
        f_res = self.f6(f_res)
        out = self.f7(f_res)

        return out


if __name__ == '__main__':
    a = torch.ones([1, 6, 500, 500])
    b = torch.ones([1, 6, 500, 500])
    c = torch.ones([1, 6, 500, 500])
    net = DAHDRNet()
    output = net(a, b, c)
    print(output.shape)
