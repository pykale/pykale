"""Implements LeNet.

Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    """Initialize LeNet.

    Args:
        input_channels (int): Input channel number.
        output_channels (int): Output channel number for block.
        additional_layers (int): Number of additional blocks for LeNet.
        output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
        linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
        squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        additional_layers,
        output_each_layer=False,
        linear=None,
        squeeze_output=True,
    ):
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.conv_layers = [nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=2, bias=False)]
        self.batch_norms = [nn.BatchNorm2d(output_channels)]
        self.global_pools = [GlobalPooling2D()]

        for i in range(additional_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    (2 ** i) * output_channels, (2 ** (i + 1)) * output_channels, kernel_size=3, padding=1, bias=False
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(output_channels * (2 ** (i + 1))))
            self.global_pools.append(GlobalPooling2D())

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.global_pools = nn.ModuleList(self.global_pools)
        self.squeeze_output = squeeze_output
        self.linear = None

        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        """Apply LeNet to layer input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        intermediate_outputs = []
        output = x
        for i in range(len(self.conv_layers)):
            output = F.relu(self.batch_norms[i](self.conv_layers[i](output)))
            output = F.max_pool2d(output, 2)
            global_pool = self.global_pools[i](output)
            intermediate_outputs.append(global_pool)

        if self.linear is not None:
            output = self.linear(output)
        intermediate_outputs.append(output)

        if self.output_each_layer:
            if self.squeeze_output:
                return [t.squeeze() for t in intermediate_outputs]
            return intermediate_outputs

        if self.squeeze_output:
            return output.squeeze()
        return output


class GlobalPooling2D(nn.Module):
    """Implements 2D Global Pooling."""

    def __init__(self):
        """Initializes GlobalPooling2D Module."""
        super(GlobalPooling2D, self).__init__()

    def forward(self, x):
        """Apply 2D Global Pooling to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x
