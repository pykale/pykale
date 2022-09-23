import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    r"""
    The MLP decoder module, which comprises four fully connected neural networks. It's a common decoder for decoding
    drug-target encoding information.

    Args:
        in_dim (int): Dimension of input feature.
        hidden_dim (int): Dimension of hidden layers.
        out_dim (int): Dimension of output layer.
        dropout_rate (float): dropout rate during training.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, 1)
        torch.nn.init.normal_(self.fc4.weight)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MultiRelaInnerProductDecoder(torch.nn.Module):
    """
    Build `DistMult
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf>`_ factorization as GripNet decoder in PoSE dataset.
    Copy-paste with slight modifications from https://github.com/NYXFLOWER/GripNet
    """

    def __init__(self, in_channels: int, num_edge_type: int):
        super(MultiRelaInnerProductDecoder, self).__init__()
        self.num_edge_type = num_edge_type
        self.in_channels = in_channels
        self.weight = torch.nn.Parameter(torch.Tensor(num_edge_type, in_channels))

        self.reset_parameters()

    def forward(self, x, edge_index: torch.Tensor, edge_type: torch.Tensor, sigmoid: bool = True) -> torch.Tensor:
        """
        Args:
            x: input node feature embeddings.
            edge_index: edge index in COO format with shape [2, num_edges].
            edge_type: The one-dimensional relation type/index for each target edge in edge_index.
            sigmoid: use sigmoid function or not.
        """
        value = (x[edge_index[0]] * x[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_channels))

    def __repr__(self) -> str:
        return "{}: DistMultLayer(in_channels={}, num_relations={})".format(
            self.__class__.__name__, self.in_channels, self.num_edge_type
        )
