import torch.nn.functional as F
import torch
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from examples.pose_gripnet.utils import *


# Copy-paste with slight modification from torch_geometric.nn.GCNConv
class GCNEncoderLayer(MessagePassing):
    r"""
    Modification of PyTorch Geometirc's nn.GCNConv, which reduces the computational cost of GCN layer for GripNet model.
    The graph convolutional operator from the `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Note: For more information please see Pytorch Geomertic's nn.GCNConv docs.

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
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(GCNEncoderLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

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
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x (torch.Tensor): The input node feature embedding.
            edge_index (torch.Tensor): Graph edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor, optional): The one-dimensional relation weight for each edge in
                :obj:`edge_index`.
        """
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
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
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


# Copy-paste with slight modification from torch_geometric.nn.RGCNConv
class RGCNEncoderLayer(MessagePassing):
    r"""
    Modification of PyTorch Geometirc's nn.RGCNConv, which reduces the computational and memory
    cost of RGCN encoder layer for GripNet model. The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Note: For more information please see Pytorch Geomertic’s nn.RGCNConv docs.

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

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_relations,
                 num_bases,
                 after_relu,
                 bias=False,
                 **kwargs):
        super(RGCNEncoderLayer, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.after_relu = after_relu

        self.basis = Parameter(
            torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Parameter(torch.Tensor(num_relations, num_bases))
        self.root = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

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
        return self.propagate(
            edge_index, x=x, edge_type=edge_type, range_list=range_list)

    def message(self, x_j, edge_index, edge_type, range_list):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_relations, self.in_channels, self.out_channels)
        # w = w[edge_type, :, :]
        # out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        out_list = []
        for et in range(range_list.shape[0]):
            start, end = range_list[et]

            tmp = torch.matmul(x_j[start: end, :], w[et])

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
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


# Copy-paste with slight modification from https://github.com/NYXFLOWER/GripNet
class HomoGraph(Module):
    r"""
    The supervertex module in GripNet. Each supervertex is a subgraph containing nodes with the
    same category or at least keep semantically-coherent. The supervertex can be regarded as homogeneous graph and
    information is propagated between them.

    Args:
        nhid_list (list): Dimensions list of hidden layers e.g. [hidden_1, hidden_2, ... hidden_n]
        requires_grad (bool, optional): Requires gradient for initial embedding (default :obj:`True`)
        start_graph (bool, optional): If set to :obj:`True`, this supervertex is the
            start point of the whole information propagation. (default :obj:`False`)
        in_dim (int, optional): the size of input sample for start graph. (default :obj:`None`)
        multi_relational: If set to :obj: 'True', the supervertex is a multi relation graph. (default :obj:`False`)
        n_rela (int, optional): Number of edge relations if supervertex is a multi relation graph. (default :obj:`None`)
        n_base (int, optional): Number of bases if supervertex is a multi relation graph. (default :obj:`None`)
    """

    def __init__(self, nhid_list, requires_grad=True, start_graph=False,
                 in_dim=None, multi_relational=False, n_rela=None, n_base=32):
        super(HomoGraph, self).__init__()
        self.multi_relational = multi_relational
        self.start_graph = start_graph
        self.out_dim = nhid_list[-1]
        self.n_cov = len(nhid_list) - 1

        if start_graph:
            self.embedding = torch.nn.Parameter(torch.Tensor(in_dim, nhid_list[0]))
            self.embedding.requires_grad = requires_grad
            self.reset_parameters()

        if multi_relational:
            assert n_rela is not None
            after_relu = [False if i == 0 else True for i in
                          range(len(nhid_list) - 1)]
            self.conv_list = torch.nn.ModuleList([
                RGCNEncoderLayer(nhid_list[i], nhid_list[i + 1], n_rela, n_base, after_relu[i])
                for i in range(len(nhid_list) - 1)])
        else:
            self.conv_list = torch.nn.ModuleList([
                GCNEncoderLayer(nhid_list[i], nhid_list[i + 1], cached=True)
                for i in range(len(nhid_list) - 1)])

    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, homo_edge_index, edge_weight=None, edge_type=None,
                range_list=None, if_catout=False):
        """"""
        if self.start_graph:
            x = self.embedding

        if if_catout:
            tmp = []
            tmp.append(x)

        if self.multi_relational:
            assert edge_type is not None
            assert range_list is not None

        for net in self.conv_list[:-1]:
            x = net(x, homo_edge_index, edge_type, range_list) \
                if self.multi_relational \
                else net(x, homo_edge_index, edge_weight)
            x = F.relu(x, inplace=True)
            if if_catout:
                tmp.append(x)

        x = self.conv_list[-1](x, homo_edge_index, edge_type, range_list) \
            if self.multi_relational \
            else self.conv_list[-1](x, homo_edge_index, edge_weight)

        x = F.relu(x, inplace=True)
        if if_catout:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)
        return x


class InterGraph(Module):
    r"""
    The superedges module in GripNet. Each superedges is a bipartite subgraph containing nodes from two categories
    forming two nodes set, connected by edges between them. The superedge can be regards as a heterogeneous graph
    connecting different supervertexs. It achieves efficient information flow propagation from all parents supervetices
    to target supervertex.

    Args:
        source_dim (int): Embedding dimensions of each source node.
        target_dim (int): Embedding dimensions of each target node.
        n_target (int): Numbers of target nodes.
        target_feat_dim (int, optional): Initial dimensions of each target node. (default: 32)
        requires_grad (bool, optional): Require gradient for the part of initial target node embedding.
            (default: obj:`True`)
    """

    def __init__(self, source_dim, target_dim, n_target, target_feat_dim=32,
                 requires_grad=True):
        super(InterGraph, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.target_feat_dim = target_feat_dim
        self.n_target = n_target
        self.target_feat = torch.nn.Parameter(
            torch.Tensor(n_target, target_feat_dim))

        self.target_feat.requires_grad = requires_grad

        self.conv = GCNEncoderLayer(source_dim, target_dim, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.target_feat.data.normal_()

    def forward(self, x, inter_edge_index, edge_weight=None, if_relu=True, mod='cat'):
        """"""
        n_source = x.shape[0]
        tmp = inter_edge_index + 0
        tmp[1, :] += n_source

        x = torch.cat(
            [x, torch.zeros((self.n_target, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_source:, :]
        if if_relu:
            x = F.relu(x)
        if mod == 'cat':
            x = torch.cat([x, torch.abs(self.target_feat)], dim=1)
        else:
            assert x.shape[1] == self.target_feat.shape[1]
            x = x + torch.abs(self.target_feat)
        return x


class multiRelaInnerProductDecoder(Module):
    def __init__(self, in_dim, num_et):
        super(multiRelaInnerProductDecoder, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, edge_type, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]] * self.weight[edge_type]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))


class multiClassInnerProductDecoder(Module):
    def __init__(self, in_dim, num_class):
        super(multiClassInnerProductDecoder, self).__init__()
        self.num_class = num_class
        self.in_dim = in_dim
        self.weight = Parameter(torch.Tensor(self.in_dim, self.num_class))

        self.reset_parameters()

    def forward(self, z, node_list, softmax=True):
        # value = (z[node_list] * self.weight[node_label]).sum(dim=1)
        # value = torch.sigmoid(value) if sigmoid else value

        pred = torch.matmul(z[node_list], self.weight)
        pred = torch.softmax(pred, dim=1) if softmax else pred

        return pred

    def reset_parameters(self):
        stdv = np.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        # self.weight.data.normal_()


class GripNet(Module):
    def __init__(self, gg_nhids_gcn, gd_out, dd_nhids_gcn, n_d_node, n_g_node, n_dd_edge_type):
        super(GripNet, self).__init__()
        self.n_d_node = n_d_node
        self.n_g_node = n_g_node
        self.gg = HomoGraph(gg_nhids_gcn, start_graph=True, in_dim=self.n_g_node)
        self.gd = InterGraph(sum(gg_nhids_gcn), gd_out[0], self.n_d_node, target_feat_dim=gd_out[-1])
        self.dd = HomoGraph(dd_nhids_gcn, multi_relational=True, n_rela=n_dd_edge_type)
        self.dmt = multiRelaInnerProductDecoder(sum(dd_nhids_gcn), n_dd_edge_type)

    def forward(self, g_feat, gg_edge_index, gg_edge_weight, gd_edge_index, dd_idx, dd_et, dd_range, device):
        z = self.gg(g_feat, gg_edge_index, edge_weight=gg_edge_weight, if_catout=True)
        z = self.gd(z, gd_edge_index)
        z = self.dd(z, dd_idx, edge_type=dd_et, range_list=dd_range, if_catout=True)
        pos_index = dd_idx
        neg_index = negative_sampling(dd_idx, self.n_d_node).to(device)
        pos_score = self.dmt(z, pos_index, dd_et)
        neg_score = self.dmt(z, neg_index, dd_et)
        return pos_score, neg_score
