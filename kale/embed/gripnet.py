"""
The GripNet is an efficient framework to learn node representations on heterogeneous graphs for the
downstream link prediction, node classification, and visualization. The code is based on
the `GripNet
<https://github.com/NYXFLOWER/GripNet>`_ source repo.
"""

import torch
import torch.nn.functional as F
from torch.nn import Module

from kale.embed.gcn import GCNEncoderLayer, RGCNEncoderLayer
from kale.prepdata.supergraph_construct import SuperVertexParaSetting


# Copy-paste with slight modification from https://github.com/NYXFLOWER/GripNet
class GripNetSuperVertex(Module):
    r"""
    The supervertex module in GripNet. Each supervertex is a subgraph containing nodes with the
    same category or at least keep semantically-coherent. The supervertex can be regarded as homogeneous graph and
    information is propagated between them.

    Args:
        channels_list (list): Channels list of hidden layers e.g. [hidden_1, hidden_2, ... hidden_n]
        requires_grad (bool, optional): Requires gradient for initial embedding (default :obj:`True`)
        start_graph (bool, optional): If set to :obj:`True`, this supervertex is the
            start point of the whole information propagation. (default :obj:`False`)
        in_channels (int, optional): Size of each input sample for start graph. (default :obj:`None`)
        multi_relational: If set to :obj: 'True', the supervertex is a multi relation graph. (default :obj:`False`)
        num_relations (int, optional): Number of edge relations if supervertex is a multi relation graph. (default :obj:`None`)
        num_bases (int, optional): Number of bases if supervertex is a multi relation graph. (default :obj:`None`)
    """

    def __init__(
        self,
        channels_list,
        requires_grad=True,
        start_graph=False,
        in_channels=None,
        multi_relational=False,
        num_relations=None,
        num_bases=32,
    ):
        super(GripNetSuperVertex, self).__init__()
        self.multi_relational = multi_relational
        self.start_graph = start_graph
        self.out_dim = channels_list[-1]
        self.n_cov = len(channels_list) - 1

        if start_graph:
            self.embedding = torch.nn.Parameter(torch.Tensor(in_channels, channels_list[0]))
            self.embedding.requires_grad = requires_grad
            self.reset_parameters()

        if multi_relational:
            assert num_relations is not None
            after_relu = [False if i == 0 else True for i in range(len(channels_list) - 1)]
            self.conv_list = torch.nn.ModuleList(
                [
                    RGCNEncoderLayer(channels_list[i], channels_list[i + 1], num_relations, num_bases, after_relu[i])
                    for i in range(len(channels_list) - 1)
                ]
            )
        else:
            self.conv_list = torch.nn.ModuleList(
                [
                    GCNEncoderLayer(channels_list[i], channels_list[i + 1], cached=True)
                    for i in range(len(channels_list) - 1)
                ]
            )

    def reset_parameters(self):
        self.embedding.data.normal_()

    def forward(self, x, homo_edge_index, edge_weight=None, edge_type=None, range_list=None, if_catout=False):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            homo_edge_index (torch.Tensor): edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor): one-dimensional relation weight for each edge.
            edge_type (torch.Tensor): one-dimensional relation type for each edge in.
            range_list (list): the index range list of each edge type with shape [num_types, 2].
            if_catout (bool): whether to concatenate each layer's output.
        """
        if self.start_graph:
            x = self.embedding

        if if_catout:
            tmp = []
            tmp.append(x)

        if self.multi_relational:
            assert edge_type is not None
            assert range_list is not None

        for net in self.conv_list[:-1]:
            x = (
                net(x, homo_edge_index, edge_type, range_list)
                if self.multi_relational
                else net(x, homo_edge_index, edge_weight)
            )
            x = F.relu(x, inplace=True)
            if if_catout:
                tmp.append(x)

        x = (
            self.conv_list[-1](x, homo_edge_index, edge_type, range_list)
            if self.multi_relational
            else self.conv_list[-1](x, homo_edge_index, edge_weight)
        )

        x = F.relu(x, inplace=True)
        if if_catout:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)
        return x


class GripNetSuperEdges(Module):
    r"""
    The superedges module in GripNet. Each superedges is a bipartite subgraph containing nodes from two categories
    forming two nodes set, connected by edges between them. The superedge can be regards as a heterogeneous graph
    connecting different supervertexs. It achieves efficient information flow propagation from all parents supervetices
    to target supervertex.

    Args:
        source_dim (int): Embedding dimensions of each source node.
        target_dim (int): Embedding dimensions of each target node aggregated from source nodes.
        n_target (int): Numbers of target nodes.
        target_feat_dim (int, optional): Initial dimensions of each target node for internal layer. (default: 32)
        requires_grad (bool, optional): Require gradient for the part of initial target node embedding.
            (default: :obj:`True`)
    """

    def __init__(self, source_dim, target_dim, n_target, target_feat_dim=32, requires_grad=True):
        super(GripNetSuperEdges, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.target_feat_dim = target_feat_dim
        self.n_target = n_target
        self.target_feat = torch.nn.Parameter(torch.Tensor(n_target, target_feat_dim))

        self.target_feat.requires_grad = requires_grad

        self.conv = GCNEncoderLayer(source_dim, target_dim, cached=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.target_feat.data.normal_()

    def forward(self, x, inter_edge_index, edge_weight=None, if_relu=True, mod="cat"):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            inter_edge_index (torch.Tensor): edge index in COO format with shape [2, num_edges].
            edge_weight (torch.Tensor): one-dimensional relation weight for each edge.
            if_relu (bool): use relu function or not.
            mod (string): the aggregation schema to use (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        """
        n_source = x.shape[0]
        tmp = inter_edge_index + 0
        tmp[1, :] += n_source

        x = torch.cat([x, torch.zeros((self.n_target, x.shape[1])).to(x.device)], dim=0)
        x = self.conv(x, tmp, edge_weight)[n_source:, :]
        if if_relu:
            x = F.relu(x)
        if mod == "cat":
            x = torch.cat([x, torch.abs(self.target_feat)], dim=1)
        else:
            assert x.shape[1] == self.target_feat.shape[1]
            x = x + torch.abs(self.target_feat)
        return x


class TypicalGripNetEncoder(Module):
    r"""
    A typical GripNet architecture with one external aggregation feature layer (GCNs) and one internal layer (RGCNs).
    The information propagates from one source nodes set to one target nodes set. You can also
    define self topological ordering of the supervertices the specific graph belongs to. For more details about GripNet,
    please see the original implementation `code
    <https://github.com/NYXFLOWER/GripNet>`_, and the original `paper
    <https://arxiv.org/abs/2010.15914>`_.

    Args:
        source_channels_list (list): Channels list of source nodes' hidden layers e.g. [channel_1, channel_2, ... channel_n]
        inter_channels_list (list): Channels list of superedge between source and target node sets with length 2.
        target_channels_list (list): Channels list of target nodes' hidden layers e.g. [channel_1, channel_2, ... channel_n]
        num_target_nodes (int): Numbers of target nodes.
        num_source_nodes (int): Numbers of source nodes.
        num_target_edge_relations (int): Number of edge relations of target supervertex.
    """

    def __init__(
        self,
        source_channels_list,
        inter_channels_list,
        target_channels_list,
        num_target_nodes,
        num_source_nodes,
        num_target_edge_relations,
    ):
        super(TypicalGripNetEncoder, self).__init__()
        self.n_target_node = num_target_nodes
        self.n_source_node = num_source_nodes
        self.source_graph = GripNetSuperVertex(source_channels_list, start_graph=True, in_channels=self.n_source_node)
        self.s2t_graph = GripNetSuperEdges(
            sum(source_channels_list),
            inter_channels_list[0],
            self.n_target_node,
            target_feat_dim=inter_channels_list[-1],
        )
        self.target_graph = GripNetSuperVertex(
            target_channels_list, multi_relational=True, num_relations=num_target_edge_relations
        )

    def forward(
        self,
        source_x,
        source_edge_index,
        source_edge_weight,
        inter_edge_index,
        target_edge_index,
        target_edge_relations,
        target_edge_range,
    ):
        """
        Args:
            source_x (torch.Tensor): The input source node feature embedding.
            source_edge_index (torch.Tensor): Source edge index in COO format with shape [2, num_edges].
            source_edge_weight (torch.Tensor): The one-dimensional relation weight
                for each edge in source graph.
            inter_edge_index: Source-target edge index in COO format with shape [2, num_edges].
            target_edge_index: Target edge index in COO format with shape [2, num_edges].
            target_edge_relations: The one-dimensional relation type for each target edge in
                :obj:`edge_index`.
            target_edge_range: The index range list of each target edge type with shape [num_types, 2].
        """
        z = self.source_graph(source_x, source_edge_index, edge_weight=source_edge_weight, if_catout=True)
        z = self.s2t_graph(z, inter_edge_index)
        z = self.target_graph(
            z, target_edge_index, edge_type=target_edge_relations, range_list=target_edge_range, if_catout=True
        )
        return z


class GripNetInternalModule(Module):
    """
    The internal module of a supervertex, which is composed of an internal feature layer and multiple internal 
    aggregation layers.

    Args:
        in_dim (int): the dimension of node features on this supervertex.
        n_edge_type (int): the number of edge types on this supervertex.
        if_start_svertex (bool): if this supervertex is a start supervertex on the supergraph.
        setting (SuperVertexParaSetting): supervertex parameter settings.
    """

    def __init__(self, in_dim: int, n_edge_type: int, if_start_svertex: bool, setting: SuperVertexParaSetting) -> None:
        super(GripNetInternalModule, self).__init__()
        # in and out dimension
        self.in_dim = in_dim
        self.out_dim = setting.inter_agg_dim[-1]

        self.n_edge_type = n_edge_type
        self.if_multirelational = 1 if n_edge_type > 1 else 0
        self.if_start_svertex = if_start_svertex
        self.setting = setting

        self.__init_internal_feat_layer__()
        self.__init_internal_agg_layer__()

    def __init_internal_feat_layer__(self):
        """internal feature layer"""

        self.embedding = torch.nn.Parameter(torch.Tensor(self.in_dim, self.setting.inter_feat_dim))
        self.embedding.requires_grad = True

        # reset parameters to be normally distributed
        self.embedding.data.normal_()

    def __init_internal_agg_layer__(self):
        """internal aggregation layers"""

        # compute the dim of input of the first internal aggregation layer
        self.in_agg_dim = self.setting.inter_feat_dim
        if not self.if_start_svertex:
            assert self.setting.mode in [
                "cat",
                "add",
            ], f"The mode {self.setting.mode} is not supported. Please use cat or add."

            if self.setting.mode == "cat":
                assert self.setting.exter_agg_dim, "The exter_agg_dim is not set."
                self.in_agg_dim += sum(self.setting.exter_agg_dim.values())
            else:
                tmp = set([self.in_agg_dim] + list(self.setting.exter_agg_dim.values()))
                assert len(tmp) == 1, "The in_agg_dim should be the same as any element in exter_agg_dim."

        # create and initialize the internal aggregation layers
        self.n_internal_agg_layer = len(self.setting.inter_agg_dim)
        tmp_dim = [self.in_agg_dim] + self.setting.inter_agg_dim

        if self.setting.if_catout:
            self.out_dim = sum(tmp_dim)

        if self.if_multirelational:
            # using RGCN if there are multiple edge types
            after_relu = [False if i == 0 else True for i in range(self.n_internal_agg_layer)]
            self.internal_agg_layers = torch.nn.ModuleList(
                [
                    RGCNEncoderLayer(
                        tmp_dim[i], tmp_dim[i + 1], self.n_edge_type, self.setting.num_bases, after_relu[i]
                    )
                    for i in range(self.n_internal_agg_layer)
                ]
            )
        else:
            # using GCN if there is only one edge type
            self.internal_agg_layers = torch.nn.ModuleList(
                [GCNEncoderLayer(tmp_dim[i], tmp_dim[i + 1], cached=True) for i in range(self.n_internal_agg_layer)]
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor = None,
        range_list: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Args:
            x (torch.Tensor): the input node feature embedding. It should be the sum or concat of the outputs of the internal
            feature layer and all external aggregation layers.
            edge_index (torch.Tensor): edge index in COO format with shape [2, #edges].
            edge_type (torch.Tensor, optional): one-dimensional relation type for each edge, indexed from 0. 
            Defaults to None.
            range_list (torch.Tensor, optional): The index range list of each edge type with shape [num_types, 2]. Defaults to None.
            edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
        """

        if self.setting.if_catout:
            tmp = []
            tmp.append(x)

        if self.if_multirelational:
            assert edge_type is not None
            assert range_list is not None

        for net in self.internal_agg_layers[:-1]:
            x = (
                net(x, edge_index, edge_type, range_list)
                if self.if_multirelational
                else net(x, edge_index, edge_weight)
            )
            x = F.relu(x, inplace=True)
            if self.setting.if_catout:
                tmp.append(x)

        x = (
            self.internal_agg_layers[-1](x, edge_index, edge_type, range_list)
            if self.if_multirelational
            else self.internal_agg_layers[-1](x, edge_index, edge_weight)
        )

        x = F.relu(x, inplace=True)
        if self.setting.if_catout:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)

        return x

    def __repr__(self):
        return f"GripNetInternalModule({self.in_dim}, {self.out_dim}): LayerList(\n(internal_feat_layer): Embedding({self.in_dim}, {self.setting.inter_feat_dim})\n(internal_agg_layers): {self.internal_agg_layers}"


class GripNetExternalModule(Module):
    """The internal module of a supervertex, which is an external feature layer.

    Args:
        in_dim (int): the dimension of the input node feature embedding.
        out_dim (int): the dimension of the output node feature embedding.
        n_out_node (int): the number of output nodes.
    """

    def __init__(self, in_dim: int, out_dim: int, n_out_node: int) -> None:
        super(GripNetExternalModule, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_out_node = n_out_node

        self.external_agg_layer = GCNEncoderLayer(in_dim, out_dim, cached=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, if_relu=True):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            edge_index (torch.Tensor): edge index in COO format with shape [2, #edges].
            edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
            if_relu (bool, optional): if use ReLU before returning node feature embeddings. Defaults to True.
        """

        n_source, n_feat = x.shape
        bigraph_edge_index = edge_index + 0
        bigraph_edge_index[1, :] += n_source

        x = torch.cat([x, torch.zeros((self.n_out_node, n_feat)).to(x.device)], dim=0)
        x = self.external_agg_layer(x, bigraph_edge_index, edge_weight)[n_source:, :]

        if if_relu:
            x = F.relu(x, inplace=True)

        return x
