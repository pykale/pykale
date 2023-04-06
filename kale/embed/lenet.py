import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    """Implements LeNet.

    Adapted from centralnet code https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """

    def __init__(
        self, in_channels, args_channels, additional_layers, output_each_layer=False, linear=None, squeeze_output=True
    ):
        """Initialize LeNet.

        Args:
            in_channels (int): Input channel number.
            args_channels (int): Output channel number for block.
            additional_layers (int): Number of additional blocks for LeNet.
            output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
            linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
            squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.
        """
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.convs = [nn.Conv2d(in_channels, args_channels, kernel_size=5, padding=2, bias=False)]
        self.bns = [nn.BatchNorm2d(args_channels)]
        self.gps = [GlobalPooling2D()]
        for i in range(additional_layers):
            self.convs.append(
                nn.Conv2d(
                    (2 ** i) * args_channels, (2 ** (i + 1)) * args_channels, kernel_size=3, padding=1, bias=False
                )
            )
            self.bns.append(nn.BatchNorm2d(args_channels * (2 ** (i + 1))))
            self.gps.append(GlobalPooling2D())
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.gps = nn.ModuleList(self.gps)
        self.sq_out = squeeze_output
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
        tempouts = []
        out = x
        for i in range(len(self.convs)):
            out = F.relu(self.bns[i](self.convs[i](out)))
            out = F.max_pool2d(out, 2)
            gp = self.gps[i](out)
            tempouts.append(gp)

        if self.linear is not None:
            out = self.linear(out)
        tempouts.append(out)
        if self.output_each_layer:
            if self.sq_out:
                return [t.squeeze() for t in tempouts]
            return tempouts
        if self.sq_out:
            return out.squeeze()
        return out


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
