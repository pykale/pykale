"""
The GripNet proposed in the `"GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs"
    <https://doi.org/10.1016/j.patcog.2022.108973>`_ (PatternRecognit 2022) paper, which is an efficient
    framework to learn node representations on heterogeneous graphs for the downstream link prediction,
    node classification, and visualization. The code is based on the https://github.com/NYXFLOWER/GripNet.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn import Module

from kale.embed.gcn import GCNEncoderLayer, RGCNEncoderLayer
from kale.prepdata.supergraph_construct import SuperGraph, SuperVertex, SuperVertexParaSetting


class GripNetInternalModule(Module):
    r"""
    The internal module of a supervertex, which is composed of an internal feature layer and multiple internal
    aggregation layers.

    Args:
        in_channels (int): the dimension of node features on this supervertex.
        num_edge_type (int): the number of edge types on this supervertex.
        start_supervertex (bool): whether this supervertex is a start supervertex on the supergraph.
        setting (SuperVertexParaSetting): supervertex parameter settings.
    """

    def __init__(
        self, in_channels: int, num_edge_type: int, start_supervertex: bool, setting: SuperVertexParaSetting
    ) -> None:
        super(GripNetInternalModule, self).__init__()
        # in and out dimension
        self.in_channels = in_channels
        self.out_channels = setting.inter_agg_channels_list[-1]

        self.num_edge_type = num_edge_type
        self.multirelational = 1 if num_edge_type > 1 else 0
        self.start_supervertex = start_supervertex
        self.setting = setting

        self.__init_inter_feat_layer__()
        self.__init_inter_agg_layer__()

    def __init_inter_feat_layer__(self):
        """internal feature layer"""

        self.embedding = torch.nn.Parameter(torch.Tensor(self.in_channels, self.setting.inter_feat_channels))
        self.embedding.requires_grad = True

        # reset parameters to be normally distributed
        self.embedding.data.normal_()

    def __init_inter_agg_layer__(self):
        """internal aggregation layers"""

        # compute the dim of input of the first internal aggregation layer
        self.in_agg_channels = self.setting.inter_feat_channels
        if not self.start_supervertex:
            if self.setting.mode == "cat":
                if not self.setting.exter_agg_channels_dict:
                    error_msg = "`exter_agg_channels_dict` is not set."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                self.in_agg_channels += sum(self.setting.exter_agg_channels_dict.values())
            elif self.setting.mode == "add":
                tmp = set([self.in_agg_channels] + list(self.setting.exter_agg_channels_dict.values()))
                if len(tmp) != 1:
                    error_msg = "`in_agg_channels` should be the same as any element in `exter_agg_channels_dict`."
                    logging.error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_msg = "`mode` value is invalid. Use 'cat' or 'add'."
                logging.error(error_msg)
                raise ValueError(error_msg)

        # create and initialize the internal aggregation layers
        self.num_inter_agg_layer = len(self.setting.inter_agg_channels_list)
        tmp_dim = [self.in_agg_channels] + self.setting.inter_agg_channels_list

        if self.setting.concat_output:
            self.out_channels = sum(tmp_dim)

        if self.multirelational:
            # using RGCN if there are multiple edge types
            after_relu = [False if i == 0 else True for i in range(self.num_inter_agg_layer)]
            self.inter_agg_layers = torch.nn.ModuleList(
                [
                    RGCNEncoderLayer(
                        tmp_dim[i], tmp_dim[i + 1], self.num_edge_type, self.setting.num_bases, after_relu[i]
                    )
                    for i in range(self.num_inter_agg_layer)
                ]
            )
        else:
            # using GCN if there is only one edge type
            self.inter_agg_layers = torch.nn.ModuleList(
                [GCNEncoderLayer(tmp_dim[i], tmp_dim[i + 1], cached=True) for i in range(self.num_inter_agg_layer)]
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
            x (torch.Tensor): the input node feature embedding.
            edge_index (torch.Tensor): edge index in COO format with shape [2, #edges].
            edge_type (torch.Tensor, optional): one-dimensional relation type for each edge, indexed from 0.
                Defaults to None.
            range_list (torch.Tensor, optional): The index range list of each edge type with shape [num_types, 2].
                Defaults to None.
            edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.

        Note: The internal feature layer is computed in the `forward` function of GripNet class. If the supervertex
        is not a start supervertex, `x` should be the sum or concat of the outputs of the internal feature
        layer and all external aggregation layers.
        """

        if self.setting.concat_output:
            tmp = []
            tmp.append(x)

        if self.multirelational:
            if edge_type is None or range_list is None:
                error_msg = "`edge_type` and `range_list` are not set."
                logging.error(error_msg)
                raise ValueError(error_msg)

        # internal feature aggregation layers
        for net in self.inter_agg_layers[:-1]:
            x = net(x, edge_index, edge_type, range_list) if self.multirelational else net(x, edge_index, edge_weight)
            x = F.relu(x, inplace=True)
            if self.setting.concat_output:
                tmp.append(x)

        x = (
            self.inter_agg_layers[-1](x, edge_index, edge_type, range_list)
            if self.multirelational
            else self.inter_agg_layers[-1](x, edge_index, edge_weight)
        )

        x = F.relu(x, inplace=True)
        if self.setting.concat_output:
            tmp.append(x)
            x = torch.cat(tmp, dim=1)

        return x

    def __repr__(self):
        tmp = [f"\n    ({i}): {l}" for i, l in enumerate(self.inter_agg_layers)]

        return "{}: ModuleList(\n  (0): InternalFeatureLayer: Embedding({}, {})\n  (1): InternalFeatureAggregationModule: ModuleList({}\n  )\n)".format(
            self.__class__.__name__, self.in_channels, self.setting.inter_feat_channels, "".join(tmp)
        )


class GripNetExternalModule(Module):
    """The internal module of a supervertex, which is an external feature layer.

    Args:
        in_channels (int): Size of each input sample. In GripNet, it should be the dimension of the output embedding of the
            corresponding parent supervertex.
        out_channels (int): Size of each output sample. In GripNet, it is the dimension of the output embedding of
            the supervertex.
        num_out_node (int): the number of output nodes.
    """

    def __init__(self, in_channels: int, out_channels: int, num_out_node: int) -> None:
        super(GripNetExternalModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_out_node = num_out_node

        self.exter_agg_layer = GCNEncoderLayer(in_channels, out_channels, cached=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None, use_relu=True):
        """
        Args:
            x (torch.Tensor): the input node feature embedding.
            edge_index (torch.Tensor): edge index in COO format with shape [2, #edges].
            edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
            use_relu (bool, optional): whether to use ReLU before returning node feature embeddings. Defaults to True.
        """

        n_source, n_feat = x.shape
        bigraph_edge_index = edge_index + 0
        bigraph_edge_index[1, :] += n_source

        x = torch.cat([x, torch.zeros((self.num_out_node, n_feat)).to(x.device)], dim=0)
        x = self.exter_agg_layer(x, bigraph_edge_index, edge_weight)[n_source:, :]

        if use_relu:
            x = F.relu(x, inplace=True)

        return x

    def __repr__(self):
        return "{}: {}".format(self.__class__.__name__, self.exter_agg_layer.__repr__())


class GripNet(Module):
    r"""The GripNet model.

    Args:
        supergraph (SuperGraph): the supergraph.

    Reference:
        Xu, H., Sang, S., Bai, P., Li, R., Yang, L. and Lu, H., 2022. GripNet: Graph Information
        Propagation on Supergraph for Heterogeneous Graphs. Pattern Recognition, p.108973.
    """

    def __init__(self, supergraph: SuperGraph) -> None:
        super(GripNet, self).__init__()

        self.supergraph = supergraph
        self.task_supervertex_name = supergraph.topological_order[-1]

        self.out_embed_dict: Dict[str, torch.Tensor] = {}

        self.__check_supergraph__()
        self.__init_supervertex_module_dict__()

        self.out_channels = self.__get_out_channels__()

    def __check_supergraph__(self) -> None:
        """check whether the input supergraph has parameter settings"""

        if self.supergraph.supervertex_setting_dict is None:
            error_msg = "`supervertex_setting_dict` is not set."
            logging.error(error_msg)
            raise ValueError(error_msg)

    def __init_supervertex_module_dict__(self):
        self.supervertex_module_dict: Dict[str, torch.nn.ModuleList] = {}
        for supervertex_name in self.supergraph.topological_order:
            supervertex = self.supergraph.supervertex_dict[supervertex_name]
            setting = self.supergraph.supervertex_setting_dict[supervertex_name]
            self.__init_module_supervertex__(supervertex, setting)

    def __init_module_supervertex__(self, supervertex: SuperVertex, setting: SuperVertexParaSetting):
        module_list = torch.nn.ModuleList()

        if not supervertex.start_supervertex:
            # add the external modules from all parent supervertices
            for in_name in supervertex.in_supervertex_list:
                in_channels = self.supervertex_module_dict[in_name][-1].out_channels

                if setting.exter_agg_channels_dict is None:
                    error_msg = "`exter_agg_channels_dict` is not set."
                    logging.error(error_msg)
                    raise ValueError(error_msg)

                out_channels = setting.exter_agg_channels_dict[in_name]
                module_list.append(GripNetExternalModule(in_channels, out_channels, supervertex.num_node))

        # add the internal module
        module_list.append(
            GripNetInternalModule(
                supervertex.num_node_feat, supervertex.num_edge_type, supervertex.start_supervertex, setting
            )
        )

        self.supervertex_module_dict[supervertex.name] = module_list

    def __get_out_channels__(self):
        task_supervertex = self.supergraph.topological_order[-1]
        return self.supervertex_module_dict[task_supervertex][-1].out_channels

    def forward(self):
        if self.supergraph.supervertex_setting_dict is None:
            error_msg = "`supervertex_setting_dict` is not set."
            logging.error(error_msg)
            raise ValueError(error_msg)

        for supervertex_name in self.supergraph.topological_order:
            mode = self.supergraph.supervertex_setting_dict[supervertex_name].mode
            self.__forward_supervertex__(supervertex_name, mode)

        return self.out_embed_dict[self.task_supervertex_name]

    def __forward_supervertex__(self, supervertex_name: str, mode: str):

        supervertex = self.supergraph.supervertex_dict[supervertex_name]
        model = self.supervertex_module_dict[supervertex_name]

        # internal feature layer
        x = torch.matmul(supervertex.node_feat, model[-1].embedding)

        # external feature aggregation layers
        if mode == "add":
            for idx in range(len(supervertex.in_supervertex_list)):
                parent_name = supervertex.in_supervertex_list[idx]
                parent_x = self.out_embed_dict[parent_name]
                superedge = self.supergraph.superedge_dict[(parent_name, supervertex_name)]

                x += model[idx](parent_x, superedge.edge_index, superedge.edge_weight)
        else:
            tmp = [x]
            for idx in range(len(supervertex.in_supervertex_list)):
                parent_name = supervertex.in_supervertex_list[idx]
                parent_x = self.out_embed_dict[parent_name]
                superedge = self.supergraph.superedge_dict[(parent_name, supervertex_name)]

                tmp.append(model[idx](parent_x, superedge.edge_index, superedge.edge_weight))
            x = torch.cat(tmp, dim=1)

        # internal feature aggregation layers
        if supervertex.num_edge_type > 1:
            x = model[-1](
                x, supervertex.edge_index, supervertex.edge_type, supervertex.range_list, supervertex.edge_weight
            )
        else:
            x = model[-1](x, supervertex.edge_index, supervertex.edge_weight)

        self.out_embed_dict[supervertex_name] = x

    def __repr__(self):
        return "{}: ModuleDict(\n{})".format(self.__class__.__name__, self.supervertex_module_dict)
