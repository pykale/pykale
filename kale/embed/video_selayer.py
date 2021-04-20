# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Python implementation of Squeeze-and-Excitation Layers (SELayer)
Initial implementation: channel-wise (SELayerC)
Improved implementation: temporal-wise (SELayerT), convolution-based channel-wise (SELayerCoC), max-pooling-based
channel-wise (SELayerMC), multi-pooling-based channel-wise (SELayerMAC)

[Redundancy and repeat of code will be reduced in the future.]

References:
    Hu Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In CVPR, pp. 7132-7141. 2018.
    For initial implementation, please go to https://github.com/hujie-frank/SENet
"""

import torch
import torch.nn as nn


class SELayerC(nn.Module):
    """Construct channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerT(nn.Module):
    """Construct temporal-wise SELayer."""

    def __init__(self, channel, reduction=2):
        super(SELayerT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()
        output = x.transpose(1, 2).contiguous()
        y = self.avg_pool(output).view(b, t)
        y = self.fc(y).view(b, t, 1, 1, 1)
        y = y.transpose(1, 2).contiguous()
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerCoC(nn.Module):
    """Construct convolution-based channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super(SELayerCoC, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=channel // reduction)
        # self.conv3 = nn.Conv2d(
        #     in_channels=channel // reduction,
        #     out_channels=channel // reduction,
        #     kernel_size=3,
        #     padding=1,
        #     groups=channel // reduction,
        #     bias=False)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(num_features=channel)

    def forward(self, x):
        b, c, t, _, _ = x.size()  # n, c, t, h, w
        y = self.conv1(x)  # n, c/r, t, h, w
        y = self.bn1(y)  # n, c/r, t, h, w
        y = self.avg_pool(y)  # n, c/r, 1, 1, 1
        y = self.conv2(y)  # n, c, 1, 1, 1
        y = self.bn2(y)  # n, c, 1, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1, 1
        # out = x * y.expand_as(x)  # n, c, t, h, w
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMC(nn.Module):
    """Construct channel-wise SELayer with max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMC, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMAC(nn.Module):
    """Construct channel-wise SELayer with the mix of average pooling and max pooling."""

    def __init__(self, channel, reduction=16):
        super(SELayerMAC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).squeeze().unsqueeze(dim=1)
        y = self.conv(y).squeeze()
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out
