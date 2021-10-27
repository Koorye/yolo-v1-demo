#!/usr/bin/env python
# --*-- encoding: utf-8 --*--
# @Author: Koorye
# @Date: 2021-10-27
# @Desc: Yolo V1 模型结构

import torch
from torch import nn
from torchvision import models


class YoloV1(nn.Module):
    """
    Yolo V1结构
    [b,3,448,448] -> [b,7,7,30]
    """

    def __init__(self):
        super(YoloV1, self).__init__()

        resnet = models.resnet.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((7,7)),
            nn.Flatten(),

            nn.Linear(7*7*512, 4096),
            nn.LeakyReLU(),
            nn.Dropout(.5),

            nn.Linear(4096, 7*7*30),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.output(x)
        x = x.view(x.size(0), 7, 7, 30)
        return x


if __name__ == '__main__':
    yolo_v1 = YoloV1()
    print(yolo_v1)
    
    x = torch.randn(1, 3, 448, 448)
    print(yolo_v1(x).size())
