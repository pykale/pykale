# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Construct a message passing network using PyTorch Geometric for the MOGONET method. MOGONET is a multiomics fusion
framework for cancer classification and biomarker identification that utilizes supervised graph convolutional networks
for omics datasets.

This code is written by refactoring the MOGONET code (https://github.com/txWang/MOGONET/blob/main/models.py) within
the 'MessagePassing' base class provided in the PyTorch Geometric.

Reference:
Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021). MOGONET integrates multi-omics data
using graph convolutional networks allowing patient classification and biomarker identification. Nature communications.
https://www.nature.com/articles/s41467-021-23774-w
"""

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import torch_sparse
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import xavier_normal_
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor


class MogonetGCNConv(MessagePassing):
    r"""Create message passing layers for the MOGONET method. Each layer is defined as:

    .. math::
        H^{(l+1)}=f(H^{(l)}, A) = \sigma(AH^{(l)}W^{(l)})

    where :math:`\mathbf{H^{(l)}}` is the input of the :math:`l`-th layer and :math:`\mathbf{W^{(l)}}` is the weight
    matrix of the :math:`l`-th layer. :math:`\sigma(.)` denotes a non-linear activation function.

    For more information please refer to the MOGONET paper.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. (default: ``True``)
        aggr (string or list or Aggregation, optional): The aggregation scheme to use,
            *e.g.*, ``"add"``, ``"sum"``, ``"mean"``, ``"min"``, ``"max"`` or ``"mul"``.
        **kwargs (optional): Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        aggr: Optional[Union[str, List[str], Aggregation]] = "add",
        **kwargs,
    ) -> None:
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all parameters of the model."""
        xavier_normal_(self.weight.data)

        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x: Tensor, edge_index: SparseTensor) -> Tensor:
        # The format of edge_index is SparseTensor which performs fast sparse-matrix multiplication
        x = torch.mm(x, self.weight)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        r"""Construct messages from node :math:`j` to node :math:`i` for each edge in ``edge_index``."""
        return x_j

    def message_and_aggregate(self, adj_t: Union[SparseTensor, Tensor], x: Tensor) -> Tensor:
        r"""Fuse computations of :func:`message` and :func:`aggregate` into a single function."""
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out: Tensor) -> Tensor:
        r"""Update node embeddings for each node :math:`i \in \mathcal{V}`."""
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class MogonetGCN(Module):
    r"""Create the structure of the graph convolutional network in the MOGONET method.
    For more information please refer to the MOGONET paper.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (List[int]): A list of sizes of hidden layers.
        dropout (float): Probability of an element to be zeroed.
    """

    def __init__(self, in_channels: int, hidden_channels: List[int], dropout: float) -> None:
        super().__init__()
        self.conv1 = MogonetGCNConv(in_channels, hidden_channels[0])
        self.conv2 = MogonetGCNConv(hidden_channels[0], hidden_channels[1])
        self.conv3 = MogonetGCNConv(hidden_channels[1], hidden_channels[2])
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: SparseTensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, 0.25)

        return x
