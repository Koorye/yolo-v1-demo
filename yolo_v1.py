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
        resnet_out_channel = resnet.fc.in_features
        self.resnet = nn.Sequential(*(list(resnet.children())[:-2]))

        self.conv = nn.Sequential(
            nn.Conv2d(resnet_out_channel, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 7*7*30),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv(x)
        x = self.output(x)
        x = x.view(x.size(0), 7, 7, 30)
        return x


if __name__ == '__main__':
    yolo_v1 = YoloV1()
    print(yolo_v1)
    
    x = torch.randn(1, 3, 448, 448)
    print(yolo_v1(x).size())
