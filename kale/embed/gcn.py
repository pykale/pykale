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


class CGCNNEncoderLayer(nn.Module):
    r"""
    CGCNN-style edge-gated graph convolution for crystal graphs.
    An implementation of paper `Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of
    Material Properties<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.

    Args:
        atom_fea_len (int): Node/atom feature dimension.
        nbr_fea_len (int):  Edge/neighbor feature dimension.
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super(CGCNNEncoderLayer, self).__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(N, M, -1), atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, total_gated_fea.shape[-1])).view(N, M, -1)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGCN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network (CGCNN).
    An implementation of paper `Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of
    Material Properties<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`.
    Each crystal is provided as:
      - atom features ``data.atom_fea``,
      - dense neighbor lists ``data.nbr_fea_idx``,
      - neighbor edge features ``data.nbr_fea`` (e.g., distance embeddings).

    Args:
        orig_atom_fea_len (int): Input atom feature dim.
        nbr_fea_len (int): Neighbor/edge feature dim.
        atom_fea_len (int): Hidden atom feature dim after embedding. Default: 64.
        n_conv (int): Number of CGCNN layers. Default: 3.
        h_fea_len (int): Hidden dim before output. Default: 128.
        n_h (int): # of hidden FC layers after pooling. Default: 1.
        classification (bool): If True, 2-way classification; else regression.
    """

    def __init__(
        self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1, classification=False
    ):
        super(CrystalGCN, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [CGCNNEncoderLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
            self.logsoftmax = nn.LogSoftmax()
            self.dropout = nn.Dropout()
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data):
        device = next(self.parameters()).device
        # atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
        atom_fea = data.atom_fea.to(device)
        nbr_fea = data.nbr_fea.to(device)
        nbr_fea_idx = data.nbr_fea_idx.to(device)
        crystal_atom_idx = data.crystal_atom_idx

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)  # add position information
        # or here posi_fea = fc()
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features
        """
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class LEFTNetZ(nn.Module):
    def __init__(
        self,
        atom_fea_dim,
        num_targets,
        otf_graph=False,
        use_pbc=True,
        output_dim=0,
        cutoff=6.0,
        num_layers=4,
        readout="mean",
        hidden_channels=128,
        num_radial=96,
        y_mean=0,
        y_std=1,
        eps=1e-10,
    ):
        """
        LEFTNet-Z model for predicting properties of materials based on their crystal structure.
        Equivariant GNN for crystal graphs with periodic boundary conditions (PBC) support. It builds a radius graph from atomic positions, uses RBF distance
        encodings and equivariant message passing. Outputs graph-level properties via readout.

        The code is implemented based on the paper `A new perspective on building efficient and expressive
        3D equivariant graph neural networks<https://openreview.net/pdf?id=hWPNYWkYPN>`

        Args:
            atom_fea_dim (int): Input atomic feature dimension (e.g., size of Z/property encoding).
            num_targets (int): Graph-level output dimension (if ``output_dim==0``, this value is used).
            otf_graph (bool): Whether to use on-the-fly graph construction. (Stored but unused here.)
            use_pbc (bool): Whether to enable periodic boundary conditions. (Stored but unused here.)
            output_dim (int): If > 0, overrides ``num_targets`` for the final layer size. Default: 0.
            cutoff (float): Radius cutoff (Å) for neighbor search. Default: 6.0.
            num_layers (int): Number of equivariant message passing blocks. Default: 4.
            readout (str): Readout reducer: {"mean","sum",...}. Default: "mean".
            hidden_channels (int): Hidden channel size for invariant/equivariant states. Default: 128.
            num_radial (int): Number of RBFs for encoding distances. Default: 96.
            y_mean (float): Target normalization mean. Default: 0.
            y_std (float): Target normalization std. Default: 1.
            eps (float): Numerical epsilon to avoid division-by-zero. Default: 1e-10.
        """
        super(LEFTNetZ, self).__init__()
        self.y_std = y_std
        self.y_mean = y_mean
        self.eps = eps
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff
        self.num_targets = num_targets
        self.readout = readout

        self.use_pbc = use_pbc
        self.otf_graph = otf_graph

        self.input_dim = atom_fea_dim

        self.radial_emb = rbf_emb(num_radial, cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(num_radial, hidden_channels), nn.SiLU(inplace=True), nn.Linear(hidden_channels, hidden_channels)
        )

        self.neighbor_emb = NeighborEmb(self.hidden_channels, self.input_dim)

        self.S_vector = S_vector(hidden_channels)

        self.lin = nn.Sequential(
            nn.Linear(3, hidden_channels // 4), nn.SiLU(inplace=True), nn.Linear(hidden_channels // 4, 1)
        )

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(EquiMessagePassing(hidden_channels, num_radial).jittable())
            self.FTEs.append(FTE(hidden_channels))

        if output_dim != 0:
            self.num_targets = output_dim

        self.last_layer = nn.Linear(hidden_channels, self.num_targets)
        # self.out_forces = EquiOutput(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()
        for layer in self.radial_lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def prop_setup(self, data, device):
        """
        Setup for the property encoding. for LEFTNet-Z, only the atomic number is used as the input feature.
        num_classes = 95 for the first 95 elements are used in the periodic table.
        """
        z = data.atom_num.long().to(device)
        z_encoded = F.one_hot(z, num_classes=95).float()
        return z_encoded

    # @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        device = next(self.parameters()).device
        pos = data.positions.to(device)
        batch = data.batch_idx.to(device)
        z = self.prop_setup(data, device)

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=1000)
        j, i = edge_index
        vecs = pos[j] - pos[i]
        dist = vecs.norm(dim=-1)

        radial_emb = self.radial_emb(dist)  # RBF, shape: (num_edges, num_radial=32), embedding for raw distance
        radial_hidden = self.radial_lin(radial_emb)  # MLP
        rbounds = 0.5 * (
            torch.cos(dist * pi / self.cutoff) + 1.0
        )  # for soft cutoff, smooth decay, value closer to 1 means stronger relationship, closer to 0 means weaker

        radial_emb = radial_emb.to(device)
        radial_hidden = radial_hidden.to(device)
        rbounds = rbounds.to(device)

        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden  # further stengthen the representation of radial_emb

        # init invariant node features
        # shape: (num_nodes, hidden_channels)
        s = self.neighbor_emb(
            z, edge_index, radial_hidden
        )  # z (num_nodes, atom_encoding), z_emb (num_nodes, hidden_channels); s=z_emb+neighbor_emb, shape: (num_nodes, hidden_channels)

        # init equivariant node features
        # shape: (num_nodes, 3, hidden_channels)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)

        # build edge-wise frame
        edge_diff = vecs
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)  # normalize the edge_diff
        edge_cross = torch.cross(pos[i], pos[j])
        edge_cross = edge_cross / ((torch.sqrt(torch.sum((edge_cross) ** 2, 1).unsqueeze(1))) + self.eps)
        edge_vertical = torch.cross(edge_diff, edge_cross)
        # shape: (num_edges, 3, 3)
        edge_frame = torch.cat(
            (edge_diff.unsqueeze(-1), edge_cross.unsqueeze(-1), edge_vertical.unsqueeze(-1)), dim=-1
        )  # normalised edge, edge_cross, edge_vertical

        node_frame = 0

        # LSE: local 3D substructure encoding
        # S_i_j shape: (num_nodes, 3, hidden_channels)
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization2 = torch.sum(S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1)
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)
        scalar4 = (
            self.lin(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1)

        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)

        for i in range(self.num_layers):
            ds, dvec = self.message_layers[i](s, vec, edge_index, radial_emb, edge_weight, edge_diff)

            s = s + ds
            vec = vec + dvec

            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec, node_frame)

            s = s + ds
            vec = vec + dvec

        s = self.last_layer(s)
        s = scatter(s, batch, dim=0, reduce=self.readout)
        s = s * self.y_std + self.y_mean
        return s

    def forward(self, data):
        return self._forward(data)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class LEFTNetProp(LEFTNetZ):
    """
    LEFTNet with property-based atomic encoding. Only differs in ``prop_setup``:
    it uses ``data.atom_fea`` (precomputed atomic properties) instead of Z one-hot.
    """

    def __init__(
        self,
        atom_fea_dim,
        num_targets,
        otf_graph=False,
        use_pbc=True,
        output_dim=0,
        cutoff=8.0,
        num_layers=4,
        readout="mean",
        hidden_channels=128,
        num_radial=96,
        y_mean=0,
        y_std=1,
        eps=1e-10,
    ):
        super(LEFTNetProp, self).__init__(
            atom_fea_dim=atom_fea_dim,
            num_targets=num_targets,
            otf_graph=otf_graph,
            use_pbc=use_pbc,
            output_dim=output_dim,
            cutoff=cutoff,
            num_layers=num_layers,
            readout=readout,
            hidden_channels=hidden_channels,
            num_radial=num_radial,
            y_mean=y_mean,
            y_std=y_std,
            eps=eps,
        )

    def prop_setup(self, data, device):
        """
        Setup for the property encoding. for LEFTNet-Prop, a one-hot encoding contains atomic properties is used as the atomic feature..
        """

        return data.atom_fea.to(device)


class CartNet(torch.nn.Module):
    """

    CartNet model from Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation.
    This is an implementation of the paper "Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation<https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00352g>"
    The code is adapted from the `CartNet repository<https://github.com/imatge-upc/CartNet/tree/main>`.
    Args:
        dim_in (int): Dimensionality of the input features.
        dim_rbf (int): Dimensionality of the radial basis function embeddings.
        num_layers (int): Number of CartNet layers in the model.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 5.0.
        invariant (bool, optional): If `True`, enforces rotational invariance in the encoder. Default is `False`.
        temperature (bool, optional): If `True`, includes temperature information in the encoder. Default is `True`.
        use_envelope (bool, optional): If `True`, applies an envelope function to the interactions. Default is `True`.
        cholesky (bool, optional): If `True`, uses a Cholesky head for the output. If `False`, uses a scalar head. Default is `True`.
    Methods:
        forward(batch):
            Performs a forward pass of the model.
            Args:
                batch: A batch of input data.
            Returns:
                pred: The model's predictions.
                true: The ground truth values corresponding to the input batch.
    """

    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True,
        use_envelope: bool = True,
        atom_types: bool = True,
    ):
        super().__init__()

        self.encoder = GeometricGraphEncoder(
            dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, temperature=temperature, atom_types=atom_types
        )
        self.dim_in = dim_in

        layers = []
        for _ in range(num_layers):
            layers.append(
                CartNetLayer(
                    dim_in=dim_in,
                    use_envelope=use_envelope,
                    radius=radius,
                )
            )
        self.layers = torch.nn.Sequential(*layers)

        self.head = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_in // 2, bias=True),
            nn.SiLU(inplace=True),
            pyg_nn.Linear(dim_in // 2, 1, bias=True),
        )

    def forward(self, batch):
        batch = self.encoder(batch)

        for layer in self.layers:
            batch = layer(batch)
        dim_size = int(batch.batch_idx.max().item() + 1)
        x = self.head(batch.x)
        pred = scatter(x, batch.batch_idx.to(x.device), dim=0, reduce="mean", dim_size=dim_size)

        return pred, batch.target


class GeometricGraphEncoder(nn.Module):
    """
    Geometry-aware graph encoder.

    Builds edges from 3D coordinates using a radius cutoff, encodes node
    features from categorical types or provided embeddings, and encodes
    edge features from radial basis functions (RBF) and optional direction
    vectors. Supports an invariant mode (distance-only edges) and a
    directional mode for equivariant message passing.

    Args:
        dim_in (int): Output dimension of encoded node features.
        dim_rbf (int): Number of radial basis functions for distance encoding.
        radius (float): Cutoff distance for neighbor search.
        invariant (bool): If True, edges use distances only. If False,
            append unit direction vectors.
        temperature (bool): If True, include temperature in node encoding.
        use_node_type (bool): If True, embed node types (e.g., atomic numbers).

    Inputs (batch):
        - `pos` (Tensor [N, 3]): Cartesian coordinates for nodes.
        - Optional `node_type` (LongTensor [N]): Categorical node IDs (e.g., atomic numbers).
        - Optional `x` (Tensor [N, F]): Pre-computed node features.
        - Optional `temperature` (Tensor): Per-node or per-graph temperatures.
        - `batch` (LongTensor [N]): Graph IDs for each node.

    Outputs:
        - `x` (Tensor [N, dim_in]): Encoded node features.
        - `edge_index` (LongTensor [2, E]): Graph connectivity.
        - `edge_attr` (Tensor [E, Fe]): Encoded edge features.
        - `cart_dist` (Tensor [E]): Pairwise distances.
        - `cart_dir` (Tensor [E, 3]): Direction vectors (if invariant=False).
    """

    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True,
        atom_types: bool = True,
    ):
        super(GeometricGraphEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_rbf = dim_rbf
        self.radius = radius
        self.invariant = invariant
        self.temperature = temperature
        self.atom_types = atom_types
        if self.atom_types:
            self.embedding = nn.Embedding(119, self.dim_in * 2)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        elif not self.temperature:
            self.embedding = nn.Embedding(1, self.dim_in)

        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in * 2, bias=True)
        elif self.atom_types:
            self.bias = nn.Parameter(torch.zeros(self.dim_in * 2))
        self.activation = nn.SiLU(inplace=True)

        if self.temperature or self.atom_types:
            self.encoder_atom = nn.Sequential(
                self.activation, pyg_nn.Linear(self.dim_in * 2, self.dim_in), self.activation
            )
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3

        self.encoder_edge = nn.Sequential(
            pyg_nn.Linear(dim_edge, self.dim_in * 2),
            self.activation,
            pyg_nn.Linear(self.dim_in * 2, self.dim_in),
            self.activation,
        )

        self.rbf = ExpNormalSmearing(0.0, radius, dim_rbf, False)

    def forward(self, batch):
        batch.device = next(self.parameters()).device
        data = batch.atom_num.long().to(batch.device)
        batch_idx = batch.batch_idx.to(batch.device)
        pos = batch.positions.to(batch.device)

        batch.edge_index = radius_graph(pos, r=self.radius, batch=batch_idx, max_num_neighbors=1000)
        j, i = batch.edge_index
        vec = pos[j] - pos[i]
        dist = vec.norm(dim=-1)
        batch.cart_dist = dist
        batch.cart_dir = vec / dist.unsqueeze(-1)

        x = self.embedding(data) + self.bias

        batch.x = self.encoder_atom(x)

        if self.invariant:  # cfg.invariant is False
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch


class CartNetLayer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the CartNet architecture.
    Args:
        dim_in (int): Dimension of the input node features.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 8.0.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.

    """

    def __init__(
        self,
        dim_in: int,
        radius: float = 8.0,
        use_envelope: bool = True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True)
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in * 3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in * 3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )

        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, radius)

    def forward(self, batch):
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist

        x_in = x
        e_in = e

        x, e = self.propagate(
            edge_index,
            Xx=x,
            Ee=e,
            He=dist,
        )

        batch.x = self.activation(x) + x_in

        batch.edge_attr = e_in + e

        return batch

    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))

        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1) * e_ij
        else:
            sigma_ij = e_ij

        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))

        out = scatter(sigma_ij * sender, index, 0, None, dim_size, reduce="sum")

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)

        e_out = self.e
        del self.e

        return x, e_out
