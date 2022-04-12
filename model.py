import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    def __init__(self, features):
        super(BasicBlock, self).__init__()
        kernel_size = 3
        padding = 1

        # self.conv = nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(features, features, kernel_size=3, dilation=3, padding='same'),
        #     nn.BatchNorm2d(features)
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(features)
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=1, dilation=1, padding='same'),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, dilation=3, padding='same'),
            nn.BatchNorm2d(features)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        tmp = self.conv(x) + x
        tmp = self.relu(tmp)
        out = self.dilated(tmp) + tmp
        return self.relu(out)


class net(nn.Module):
    def __init__(self, channels):
        super(net, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(7):
            layers.append(BasicBlock(features))

        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn_res = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn_res(x)
        return out
