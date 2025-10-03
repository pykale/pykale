# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Python implementation of Squeeze-and-Excitation Layers (SELayer)
Initial implementation: channel-wise (SELayerC)
Improved implementation: temporal-wise (SELayerT), max-pooling-based channel-wise (SELayerMC),
multi-pooling-based channel-wise (SELayerMAC)

[Redundancy and repeat of code will be reduced in the future.]

References:
    Hu Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." In CVPR, pp. 7132-7141. 2018.
    For initial implementation, please go to https://github.com/hujie-frank/SENet
"""

import torch
import torch.nn as nn


def get_selayer(attention):
    """Returns a SELayer class based on the attention identifier.

    Args:
        attention: Name of the SELayer implementation to retrieve.

    Returns:
        SELayer: Subclass corresponding to the requested attention name.

    Raises:
        ValueError: If the provided attention name is unsupported.
    """

    selayer_map = {
        "SELayerC": SELayerC,
        "SELayerT": SELayerT,
        "SELayerMC": SELayerMC,
        "SELayerMAC": SELayerMAC,
    }

    try:
        return selayer_map[attention]
    except KeyError as exc:
        raise ValueError(f"Wrong MODEL.ATTENTION. Current:{attention}") from exc


class SELayer(nn.Module):
    """Base class for squeeze-and-excitation layers.

    Args:
        channel: Total number of channels expected by the layer.
        reduction: Reduction ratio applied inside the excitation block.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction

    def forward(self, x):
        squeezed = self._squeeze(x)
        excited = self._excite(squeezed)
        scale = self._reshape(excited, x)
        return self._scale_residual(x, scale)

    def _scale_residual(self, x, scale):
        scale = scale - 0.5
        return x + x * scale.expand_as(x)

    def _squeeze(self, x):
        raise NotImplementedError()

    def _excite(self, squeezed):
        raise NotImplementedError()

    def _reshape(self, excited, x):
        return excited.view(x.size(0), self.channel, 1, 1, 1)

    @staticmethod
    def _make_se_fc(channel, reduction):
        """Builds a squeeze-excitation projection independent of layer state."""

        return nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )


class SELayerC(SELayer):
    """Construct channel-wise SELayer."""

    def __init__(self, channel, reduction=16):
        super().__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = self._make_se_fc(channel, reduction)

    def _squeeze(self, x):
        return self.avg_pool(x).view(x.size(0), self.channel)

    def _excite(self, squeezed):
        return self.fc(squeezed)


class SELayerT(SELayer):
    """Construct temporal-wise SELayer."""

    def __init__(self, channel, reduction=2):
        super().__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = self._make_se_fc(channel, reduction)

    def _squeeze(self, x):
        batch_size, _, time_steps, _, _ = x.size()
        output = x.transpose(1, 2).contiguous()
        return self.avg_pool(output).view(batch_size, time_steps)

    def _excite(self, squeezed):
        return self.fc(squeezed)

    def _reshape(self, excited, x):
        batch_size, time_steps = excited.size()
        scale = excited.view(batch_size, time_steps, 1, 1, 1)
        return scale.transpose(1, 2).contiguous()


class SELayerMC(SELayer):
    """Construct channel-wise SELayer with max pooling."""

    def __init__(self, channel, reduction=16):
        super().__init__(channel, reduction)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = self._make_se_fc(channel, reduction)

    def _squeeze(self, x):
        return self.max_pool(x).view(x.size(0), self.channel)

    def _excite(self, squeezed):
        return self.fc(squeezed)


class SELayerMAC(SELayer):
    """Construct channel-wise SELayer with the mix of average pooling and max pooling."""

    def __init__(self, channel, reduction=16):
        super().__init__(channel, reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 2), bias=False)
        self.fc = self._make_se_fc(channel, reduction)

    def _squeeze(self, x):
        batch_size = x.size(0)
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).contiguous()
        y = y.view(batch_size, self.channel, 2)
        y = self.conv(y.unsqueeze(1))
        return y.view(batch_size, self.channel)

    def _excite(self, squeezed):
        return self.fc(squeezed)
