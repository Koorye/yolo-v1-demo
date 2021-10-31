#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: Yolo V1 模型结构

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=2, bias=False, dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type == 'B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class YoloV1(nn.Module):
    """
    Yolo V1结构
    """

    def __init__(self):
        super(YoloV1, self).__init__()

        self.backbone = models.resnet.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.output = nn.Sequential(
            detnet_bottleneck(2048, 256),
            detnet_bottleneck(256, 256),
            detnet_bottleneck(256, 256),

            nn.Conv2d(256, 30, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.output(x)
        x = x.permute(0, 2, 3, 1)
        return x


if __name__ == '__main__':
    yolo_v1 = YoloV1()
    print(yolo_v1)

    x = torch.randn(1, 3, 448, 448)
    print(yolo_v1(x).size())