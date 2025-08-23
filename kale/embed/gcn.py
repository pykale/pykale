from math import pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_cluster import radius_graph
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter, scatter_add

from kale.embed.materials_equivariant import (
    CosineCutoff,
    EquiMessagePassing,
    ExpNormalSmearing,
    FTE,
    NeighborEmb,
    rbf_emb,
    S_vector,
)


class GCNEncoderLayer(MessagePassing):
    r"""
    Modification of PyTorch Geometirc's nn.GCNConv, which reduces the computational cost of GCN layer for
    `GripNet <https://github.com/NYXFLOWER/GripNet>`_ model.
    The graph convolutional operator from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ (ICLR 2017) paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Note: For more information please see Pytorch Geomertic's `nn.GCNConv
    <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.message_passing>`_ docs.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True, **kwargs):
        super(GCNEncoderLayer, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.fill_(0)

        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        """
        Add self-loops and apply symmetric normalization
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x (torch.Tensor): The input node feature embedding.
            edge_index (torch.Tensor): Graph edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor, optional): The one-dimensional relation weight for each edge in
                :obj:`edge_index` (default: None).
        """
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}".format(self.cached_num_edges, edge_index.size(1))
                )

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


class RGCNEncoderLayer(MessagePassing):
    r"""
    Modification of PyTorch Geometric's nn.RGCNConv, which reduces the computational and memory
    cost of RGCN encoder layer for `GripNet <https://github.com/NYXFLOWER/GripNet>`_ model.
    The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Note: For more information please see Pytorch Geomertic’s `nn.RGCNConv
    <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.message_passing>`_ docs.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of edge relations.
        num_bases (int): Use bases-decoposition regulatization scheme and num_bases denotes the number of bases.
        after_relu (bool): Whether input embedding is activated by relu function or not.
        bias (bool): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases, after_relu, bias=False, **kwargs):
        super(RGCNEncoderLayer, self).__init__(aggr="mean", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.after_relu = after_relu

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.att.data.normal_(std=1 / np.sqrt(self.num_bases))

        if self.after_relu:
            self.root.data.normal_(std=2 / self.in_channels)
            self.basis.data.normal_(std=2 / self.in_channels)

        else:
            self.root.data.normal_(std=1 / np.sqrt(self.in_channels))
            self.basis.data.normal_(std=1 / np.sqrt(self.in_channels))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, edge_index, edge_type, range_list):
        """
        Args:
            x (torch.Tensor): The input node feature embedding.
            edge_index (torch.Tensor): Graph edge index in COO format with shape [2, num_edges].
            edge_type (torch.Tensor): The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
            range_list (torch.Tensor): The index range list of each edge type with shape [num_types, 2].
        """
        return self.propagate(edge_index, x=x, edge_type=edge_type, range_list=range_list)

    def message(self, x_j, edge_index, edge_type, range_list):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        # w = w[edge_type, :, :]
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        out_list = []
        for et in range(range_list.shape[0]):
            start, end = range_list[et]

            tmp = torch.matmul(x_j[start:end, :], w[et])

            # xxx = x_j[start: end, :]
            # tmp = checkpoint(torch.matmul, xxx, w[et])

            out_list.append(tmp)

        # TODO: test this
        return torch.cat(out_list)

    def update(self, aggr_out, x):
        out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return "{}({}, {}, num_relations={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_relations
        )


class GCNEncoder(nn.Module):
    r"""
    The GraphDTA's GCN encoder module, which comprises three graph convolutional layers and one full connected layer.
    The model is a variant of DeepDTA and is applied to encoding drug molecule graph information. The original paper
    is  `"GraphDTA: Predicting drug–target binding affinity with graph neural networks"
    <https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btaa921/5942970>`_ .

    Args:
        in_channel (int): Dimension of each input node feature.
        out_channel (int): Dimension of each output node feature.
        dropout_rate (float): dropout rate during training.
    """

    def __init__(self, in_channel=78, out_channel=128, dropout_rate=0.2):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, in_channel)
        self.conv2 = GCNConv(in_channel, in_channel * 2)
        self.conv3 = GCNConv(in_channel * 2, in_channel * 4)
        self.fc = nn.Linear(in_channel * 4, out_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        x = self.fc(x)
        x = self.dropout(x)
        return x


class MolecularGCN(nn.Module):
    """
    A molecular feature extractor using a Graph Convolutional Network (GCN).

    This class implements a GCN to extract features from molecular graphs. It includes an initial
    linear transformation followed by a series of graph convolutional layers. The output is a
    fixed-size feature vector for each molecule.

    Args:
        in_feats (int): Number of input features each node has.
        dim_embedding (int): Dimensionality of the embedding space after the initial linear transformation.
        padding (bool): Whether to apply padding (set certain weights to zero).
        hidden_feats (list of int): A list specifying the number of hidden units for each GCN layer.
        activation (callable, optional): Activation function to apply after each GCN layer.
    """

    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            # If padding is enabled, set the last row of the weight matrix to zeros (for any padded (dummy) nodes)
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        self.gcn_layers = nn.ModuleList()
        self.activations = []
        prev_dim = dim_embedding
        for hidden_dim in hidden_feats:
            self.gcn_layers.append(GCNConv(prev_dim, hidden_dim))
            self.activations.append(activation)
            prev_dim = hidden_dim

        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        x, edge_index = batch_graph.x, batch_graph.edge_index
        x = self.init_transform(x)

        for gcn_layer, activation in zip(self.gcn_layers, self.activations):
            x = gcn_layer(x, edge_index)
            if activation is not None:
                x = activation(x)

        # Expect all graphs to be padded to the same number of nodes
        batch_size = batch_graph.num_graphs
        x = x.view(batch_size, -1, self.output_feats)

        return x


