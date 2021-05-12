""" Models for efficient test following PyTorch-Lightning.

References: https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/helpers/boring_model.py
"""

import torch.nn as nn


class VideoBoringModel(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channel, 1024)

    def forward(self, x):
        x = self.avg_pool3d(x).squeeze()
        x = self.fc(x)
        return x

    def output_size(self):
        return self.fc.in_features
