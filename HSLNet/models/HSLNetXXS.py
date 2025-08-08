# -*- coding: utf-8 -*-

import os
from functools import partial

import cv2
import numpy
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math
from HSLNet.moblievit_star import mobile_vit_xx_small

def upsample(x, y):
    return F.interpolate(x, y, mode='bilinear', align_corners=True)

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)

class MP(nn.Module):  # Multi-scale perception (MP) module
    def __init__(self, in_channel, out_channel):
        super(MP, self).__init__()
        self.conv1 = DSConv3x3(out_channel, out_channel, stride=1, dilation=1)
        self.conv2 = DSConv3x3(out_channel, out_channel, stride=1, dilation=2)
        self.conv3 = DSConv3x3(out_channel, out_channel, stride=1, dilation=4)
        self.conv4 = DSConv3x3(out_channel, out_channel, stride=1, dilation=8)

        self.fuse = DSConv3x3(out_channel, out_channel, relu=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        out = self.fuse(out1 + out2 + out3 + out4)

        return out + x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class SMFM(nn.Module):
    def __init__(self, channel, h, w):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channel * 3)
        self.convTo2 = nn.Conv2d(channel * 2, channel, 3, 1, 1)
        self.mp = MP(channel, channel)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, 1, 0),
            nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, r, d):
        c = torch.cat((r, d), dim=1)
        fc = self.convTo2(c)

        s1 = self.sa(r)
        r1 = r * s1

        s2 = self.sa(d)
        d1 = d * s2

        f = torch.cat((r1, d1, fc), dim=1)
        f = self.ca(f) * f
        f = self.conv3(f)

        f = self.mp(f)

        return f

class HIRM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HIRM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()

    def forward(self, fh, fl):
        f1 = fh + fl
        fh1 = fh * self.ca(f1)
        f2 = fh1 + fl
        fh2 = fh1 * self.sa(f2)
        f3 = fh2 + fl

        return f3 + fl * fh

class BConv(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(BConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class MFEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFEM, self).__init__()
        self.DWConv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, groups=out_channel),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channel, out_channel, 1, 1, 0, groups=1),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.gapLayer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.gmpLayer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bconv = nn.Sequential(
            nn.Conv2d(out_channel * 2 + 1, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)
        b1 = self.DWConv(x)
        b2 = x
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        b3 =torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        b3 = self.upsample2(b3)
        out = self.bconv(torch.cat((b1, b2, b3), dim=1))
        return out
class pvt(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(pvt, self).__init__()
        # encoder
        self.rgb_pvt = mobile_vit_xx_small()
        self.depth_pvt = mobile_vit_xx_small()

        self.fusion1 = SMFM(80, 12, 12)
        self.fusion2 = SMFM(64, 24, 24)
        self.fusion3 = SMFM(48, 48, 48)
        self.fusion4 = SMFM(24, 96, 96)

        # fe
        self.conv_x4 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x2 = nn.Conv2d(32, 1, 3, 1, 1)
        self.conv_x1 = nn.Conv2d(32, 1, 3, 1, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.hff1 = HIRM(in_ch=32, out_ch=32)
        self.hff2 = HIRM(in_ch=32, out_ch=32)
        self.hff3 = HIRM(in_ch=32, out_ch=32)
        self.hff4 = HIRM(in_ch=32, out_ch=32)

        self.fe_12 = MFEM(80, 32)
        self.fe_24 = MFEM(64, 32)
        self.fe_48 = MFEM(48, 32)
        self.fe_96 = MFEM(24, 32)

        self.bconv24 = BConv(64, 32, 3, 1, 1)
        self.bconv48 = BConv(64, 32, 3, 1, 1)
        self.bconv96 = BConv(64, 32, 3, 1, 1)

        self.sa = SpatialAttention()
        self.ca = ChannelAttention(32)

    def forward(self, rgb, depth):
        rgb_list = self.rgb_pvt(rgb)
        depth_list = self.depth_pvt(depth)

        r0 = rgb_list[5]
        r1 = rgb_list[4] + r0  # (80,12,12)
        r4 = rgb_list[1]  # (8,24,96,96)
        r3 = rgb_list[2]  # (8,48,48,48)
        r2 = rgb_list[3]  # (8,64,24,24)

        d0 = depth_list[5]
        d1 = depth_list[4] + d0  # (80,12,12)
        d4 = depth_list[1]  # (8,24,96,96)
        d3 = depth_list[2]  # (8,48,48,48)
        d2 = depth_list[3]  # (8,64,24,24)

        # Encoder
        F1 = self.fusion1(r1, d1)  # (80,12,12)
        F2 = self.fusion2(r2, d2)  # (64,24,24)
        F3 = self.fusion3(r3, d3)  # (48,48,48)
        F4 = self.fusion4(r4, d4)  # (24,96,96)

        # Decoder
        x1_fe = self.fe_12(F1)  # (32,12,12)
        x1u2 = self.relu(self.bn1(self.conv1(self.upsample2(x1_fe))))
        x2_fe = self.fe_24(F2)  # (32,24,24)
        x3_fe = self.fe_48(F3)  # (32,48,48)
        x4_fe = self.fe_96(F4)  # (32,96,96)

        de1 = self.hff1(x1u2, x2_fe)
        de1u = self.relu(self.bn1(self.conv1(self.upsample2(de1))))  # 2倍
        de2 = self.hff2(de1u, x3_fe)
        de2u = self.relu(self.bn1(self.conv1(self.upsample2(de2))))  # 2倍
        de3 = self.hff3(de2u, x4_fe)

        c1 = x1_fe
        c2 = self.bconv24(torch.cat((de1, x1u2), dim=1))
        c3 = self.bconv48(torch.cat((de2, de1u), dim=1))
        c4 = self.bconv96(torch.cat((de3, de2u), dim=1))

        o1 = c1
        o2 = c2 + self.upsample2(o1)
        o3 = c3 + self.upsample2(o2)
        o4 = c4 + self.upsample2(o3)

        shape = rgb.size()[2:]  # shape:(384,384)

        out1 = F.interpolate(self.conv_x1(o1), size=shape, mode='bilinear')  # (b,1,384,384)
        out2 = F.interpolate(self.conv_x2(o2), size=shape, mode='bilinear')  # (b,1,384,384)
        out3 = F.interpolate(self.conv_x3(o3), size=shape, mode='bilinear')  # (b,1,384,384)
        out4 = F.interpolate(self.conv_x4(o4), size=shape, mode='bilinear')  # (b,1,384,384)

        return out1, out2, out3, out4

        # ###################################################   end   #########################################

    def load_pre(self, pre_model_rgb, pre_model_depth):
        self.rgb_pvt.load_state_dict(torch.load(pre_model_rgb), strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model_rgb}")
        self.depth_pvt.load_state_dict(torch.load(pre_model_depth), strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model_depth}")



# ########################################### end #####################################################
import torch
from torchvision.models import resnet18
from thop import profile

model = pvt()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input1 = torch.randn(1, 3, 384, 384)
input1 = input1.to(device)
input2 = torch.randn(1, 3, 384, 384)
input2 = input2.to(device)
flops, params = profile(model.cuda(), inputs=(input1, input2,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))

import torch

iterations = 300  # 重复计算的轮次

model = pvt()
device = torch.device("cuda:0")
model.to(device)

random_input1 = torch.randn(1, 3, 384, 384).to(device)
random_input2 = torch.randn(1, 3, 384, 384).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input1, random_input2)

# 测速
times = torch.zeros(iterations)  # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input1, random_input2)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)  # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
