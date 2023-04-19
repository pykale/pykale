# =============================================================================
# Author: Hao Xu, xu-hao@ucsd.edu or nyx0flower@gmail.com
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""
The supergraph structure from the `"GripNet: Graph Information Propagation on Supergraph for Heterogeneous Graphs"
    <https://doi.org/10.1016/j.patcog.2022.108973>`_ (PatternRecognit 2022) paper.
"""

import logging
from typing import Dict, List

import networkx as nx
import torch


class SuperVertex(object):
    r"""
    The supervertex structure in GripNet. Each supervertex is a subgraph containing nodes of the same category
        that are semantically-coherent. Supervertices can be homogeneous or heterogeneous.

    Args:
        name (str): the name of the supervertex.
        node_feat (torch.Tensor): node features of the supervertex with shape [#nodes, #features]. We recommend
            using `torch.sparse.FloatTensor()` if the node feature matrix is sparse.
        edge_index (torch.Tensor): edge indices in COO format with shape [2, #edges].
        edge_type (torch.Tensor, optional): one-dimensional relation type for each edge, indexed from 0. Defaults to None.
        edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.

    Examples:
        >>> import torch
        >>> node_feat = torch.randn(4, 20)
        >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        >>> edge_type = torch.tensor([0, 0, 1, 1])
        >>> edge_weight = torch.randn(4)
        >>> # create a supervertex with homogeneous edges
        >>> supervertex_homo = SuperVertex(node_feat, edge_index)
        >>> # create a supervertex with heterogeneous edges
        >>> supervertex_hete = SuperVertex(node_feat, edge_index, edge_type)
        >>> # create a supervertex with weighted edges
        >>> supervertex_weight1 = SuperVertex(node_feat, edge_index, edge_weight=edge_weight)
        >>> supervertex_weight2 = SuperVertex(node_feat, edge_index, edge_type, edge_weight)
    """

    def __init__(
        self,
        name: str,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> None:

        self.name = name
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_weight = edge_weight

        # get the number of nodes, node features and edges
        self.num_node, self.num_node_feat = node_feat.shape
        self.num_edge = edge_index.shape[1]

        # initialize in-supervertex and out-supervertex lists
        self.in_supervertex_list: List[str] = []
        self.out_supervertex_list: List[str] = []
        self.start_supervertex = True

        self.__process_edges__()

    def __process_edges__(self):
        r"""
        process the edges of the supervertex.
        """
        # get the number of edge types
        if self.edge_type is None:
            self.num_edge_type = 1
        else:
            unique_edge_type = self.edge_type.unique()
            self.num_edge_type = unique_edge_type.shape[0]

            # check whether the index of edge type is continuous and starts from 0
            if self.num_edge_type != unique_edge_type.max() + 1:
                error_msg = "`edge_type` is not set correctly. It should be continuous and start from 0."
                logging.error(error_msg)
                raise ValueError(error_msg)

            # sort the edges and edge types
            sort_index = torch.argsort(self.edge_type)
            self.edge_index = self.edge_index[:, sort_index]
            self.edge_type = self.edge_type[sort_index]

            self.__get_range_list__()

    def __get_range_list__(self):
        """get the range of edge types"""
        idx = 0
        range_list = [[0, 0]]

        for i, edge_type in enumerate(self.edge_type):
            if edge_type != idx:
                idx = edge_type
                range_list[-1][1] = i
                range_list.append([i, i])

        range_list[-1][1] = i + 1

        self.range_list = torch.tensor(range_list)

    def __repr__(self) -> str:
        return f"SuperVertex(\n    name={self.name}, \n    node_feat={self.node_feat.shape}, \n    edge_index={self.edge_index.shape}, \n    num_edge_type={self.num_edge_type})"

    def add_in_supervertex(self, vertex_name: str):
        self.in_supervertex_list.append(vertex_name)

    def add_out_supervertex(self, vertex_name: str):
        self.out_supervertex_list.append(vertex_name)


class SuperEdge(object):
    r"""
    The superedge structure in GripNet. Each superedge is a bipartite subgraph containing nodes from two categories
    forming two node sets, connected by edges between them. A superedge can be regarded as a heterogeneous graph
    connecting two supervertices.

    Args:
        source_supervertex (str): the name of the source supervertex.
        target_supervertex (str): the name of the target supervertex.
        edge_index (torch.Tensor): edge indices in COO format with shape [2, #edges]. The first row is the index of
            source nodes, and the second row is the index of target nodes.
        edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
    """

    def __init__(
        self,
        source_supervertex: str,
        target_supervertex: str,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> None:

        self.direction = (source_supervertex, target_supervertex)
        self.source_supervertex = source_supervertex
        self.target_supervertex = target_supervertex
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def __repr__(self) -> str:
        return f"SuperEdges(\n    edge_direction={self.source_supervertex}->{self.target_supervertex}, \n    edge_index={self.edge_index.shape})"


class SuperVertexParaSetting(object):
    r"""Parameter settings for each supervertex.

        Args:
            supervertex_name (str): the name of the supervertex.
            inter_feat_channels (int): the dimension of the output of the internal feature layer.
            inter_agg_channels_list (List[int]): the output dimensions of a sequence of internal aggregation layers.
            exter_agg_channels_dict (Dict[str, int], optional): the dimension of received message vector from parent supervertices.
                Defaults to None.
            mode (str, optional): the allowed gripnet mode--'cat' or 'add'. Defaults to None.
            num_bases (int, optional): the number of bases used for basis-decomposition if the
                supervertex is multi-relational. Defaults to 32.
            concat_output (bool, optional): whether to concatenate the output of each layers. Defaults to True.
        """

    def __init__(
        self,
        supervertex_name: str,
        inter_feat_channels: int,
        inter_agg_channels_list: List[int],
        exter_agg_channels_dict: Dict[str, int] = None,
        mode: str = None,
        num_bases: int = 32,
        concat_output: bool = True,
    ) -> None:
        self.supervertex_name = supervertex_name
        self.inter_feat_channels = inter_feat_channels
        self.inter_agg_channels_list = inter_agg_channels_list
        self.mode = mode
        self.num_bases = num_bases
        self.concat_output = concat_output
        self.exter_agg_channels_dict = exter_agg_channels_dict

        # check whether the mode is valid
        if self.mode is not None and self.mode not in ["cat", "add"]:
            error_msg = "`mode` value is invalid. It should be 'cat' or 'add'."
            logging.error(error_msg)
            raise ValueError(error_msg)


class SuperGraph(object):
    r"""
    The supergraph structure in GripNet. Each supergraph is a directed acyclic graph (DAG) containing
    supervertices and superedges.

    Args:
        supervertex_list (list[SuperVertex]): a list of supervertices.
        superedge_list (list[SuperEdge]): a list of superedges.
        supervertex_para_setting (dict[str, SuperVertexParaSetting], Optional): the parameter settings for each supervertex.
    """

    def __init__(
        self,
        supervertex_list: List[SuperVertex],
        superedge_list: List[SuperEdge],
        supervertex_setting_dict: Dict[str, SuperVertexParaSetting] = None,
    ) -> None:

        self.supervertex_dict = {supervertex.name: supervertex for supervertex in supervertex_list}
        self.superedge_dict = {superedge.direction: superedge for superedge in superedge_list}
        self.supervertex_setting_dict = supervertex_setting_dict

        self.__process_supergraph__()
        self.__update_supervertex__()

    def __process_supergraph__(self):
        r"""
        Process the graph of the supergraph.
        """
        # initialize the supergraph
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.superedge_dict.keys())

        # check whether the graph is a DAG
        if not nx.is_directed_acyclic_graph(self.G):
            error_msg = "The supergraph is not a directed acyclic graph."
            logging.error(error_msg)
            raise TypeError(error_msg)

        self.num_supervertex = self.G.number_of_nodes()
        self.num_superedge = self.G.number_of_edges()

        # get the topological order of the supergraph
        self.topological_order = list(nx.topological_sort(self.G))

    def __update_supervertex__(self):
        r"""
        Update the supervertices according to the superedges of the supergraph.
        """
        # update the in- and out-supervertex lists of each supervertex
        for n1, n2 in self.G.edges():
            self.supervertex_dict[n2].add_in_supervertex(n1)
            self.supervertex_dict[n1].add_out_supervertex(n2)

        # update if the supervertex is a start supervertex of the supergraph
        for _, sv in self.supervertex_dict.items():
            if sv.in_supervertex_list:
                sv.start_supervertex = False

    def set_supergraph_para_setting(self, supervertex_setting_list: List[SuperVertexParaSetting]):
        """Set the parameters of the supergraph.

        Args:
            supervertex_setting_list (list[SuperVertexParaSetting]): a list of parameter settings for each supervertex.
        """
        self.supervertex_setting_dict = {sv.supervertex_name: sv for sv in supervertex_setting_list}

    def __repr__(self) -> str:
        return f"SuperGraph(\n  svertex_dict={self.supervertex_dict.values()}, \n  sedge_dict={self.superedge_dict.values()}, \n  G={self.G}), \n  topological_order={self.topological_order}"
