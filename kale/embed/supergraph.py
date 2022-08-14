from typing import List

import networkx as nx
import torch


class SuperVertex(object):
    def __init__(
        self,
        name: str,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> None:
        r"""
        The supervertex structure in GripNet. Each supervertex is a subgraph containing nodes with the same category
        and at least keep semantically-coherent. Supervertices can be homogeneous or heterogeneous.

        Args:
            name (str): the name of the supervertex.
            node_feat (torch.Tensor): node features of the supervertex with shape [#nodes, #features]. We recommend
            using `torch.sparse.FloatTensor()` if the node feature matrix is sparse.
            edge_index (torch.Tensor): edge indices in COO format with shape [2, #edges].
            edge_type (torch.Tensor, optional): one-dimensional relation type for each edge, indexed from 0. Defaults to None.
            edge_weight (torch.Tensor, optional): one-dimensional weight for each edge. Defaults to None.
        """
        self.name = name
        self.node_feat = node_feat
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_weight = edge_weight

        # get the number of nodes, node features and edges
        self.n_node, self.n_node_feat = node_feat.shape
        self.n_edge = edge_index.shape[1]

        # initialize in-supervertex and out-supervertex lists
        self.in_supervertex_list: List[str] = []
        self.out_supervertex_list: List[str] = []
        self.if_start_supervertex = True

        self.__process_edges__()

    def __process_edges__(self):
        r"""
        process the edges of the supervertex.
        """
        # get the number of edge types
        if self.edge_type is None:
            self.n_edge_type = 1
        else:
            unique_edge_type = self.edge_type.unique()
            self.n_edge_type = unique_edge_type.shape[0]

            # check if the index of edge type is continuous and starts from 0
            assert (
                self.n_edge_type == unique_edge_type.max() + 1
            ), "The index of edge type is not continuous and starts from 0."

    def __repr__(self) -> str:
        return f"SuperVertex(\n    name={self.name}, \n    node_feat={self.node_feat.shape}, \n    edge_index={self.edge_index.shape}, \n    n_edge_type={self.n_edge_type})"

    def add_in_supervertex(self, vertex_name: str):
        self.in_supervertex_list.append(vertex_name)

    def add_out_supervertex(self, vertex_name: str):
        self.out_supervertex_list.append(vertex_name)

    def set_name(self, vertex_name: str):
        self.name = vertex_name


class SuperEdge(object):
    def __init__(self, source_supervertex: str, target_supervertex: str, edge_index: torch.Tensor) -> None:
        """
        The superedge structure in GripNet. Each superedge is a bipartite subgraph containing nodes from two categories
        forming two node sets, connected by edges between them. The superedge can be regards as a heterogeneous graph
        connecting two supervertices.

        Args:
            source_supervertex (str): the name of the source supervertex.
            target_supervertex (str): the name of the target supervertex.
            edge_index (torch.Tensor): edge indices in COO format with shape [2, #edges]. The first row is the index of
            source nodes, and the second row is the index of target nodes.
        """
        self.direction = (source_supervertex, target_supervertex)
        self.source_supervertex = source_supervertex
        self.target_supervertex = target_supervertex
        self.edge_index = edge_index

    def __repr__(self) -> str:
        return f"SuperEdges(\n    edge_direction={self.source_supervertex}->{self.target_supervertex}, \n    edge_index={self.edge_index.shape})"


class SuperGraph(object):
    def __init__(self, supervertex_list: List[SuperVertex], superedge_list: List[SuperEdge]) -> None:
        r"""
        The supergraph structure in GripNet. Each supergraph is a directed acyclic graph (DAG) containing supervertices and superedges.

        Args:
            supervertex_list (list[SuperVertex]): a list of supervertices.
            superedge_list (list[SuperEdge]): a list of superedges.
        """
        self.supervertex_dict = {sv.name: sv for sv in supervertex_list}
        self.superedge_dict = {se.direction: se for se in superedge_list}

        self.__process_supergraph__()
        self.__update_supervertex__()

    def __process_supergraph__(self):
        r"""
        Process the graph of the supergraph.
        """
        # initialize the supergraph
        self.G = nx.DiGraph()
        self.G.add_edges_from(self.superedge_dict.keys())

        # check if the graph is a DAG
        assert nx.is_directed_acyclic_graph(self.G), "The supergraph is not a directed acyclic graph."

        self.n_supervertex = self.G.number_of_nodes()
        self.n_superedge = self.G.number_of_edges()

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
                sv.if_start_supervertex = False

    def __repr__(self) -> str:
        return f"SuperGraph(\n  svertex_dict={self.supervertex_dict.values()}, \n  sedge_dict={self.superedge_dict.values()}, \n  G={self.G}), \n  topological_order={self.topological_order}"


if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(1111)

    # create three supervertices
    node_feat = torch.randn(4, 20)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_type = torch.tensor([0, 0, 1, 1])

    sv1 = SuperVertex("1", node_feat, edge_index)
    sv2 = SuperVertex("2", node_feat, edge_index, edge_type)
    sv3 = SuperVertex("3", node_feat, edge_index, edge_type)

    # determine the supervertices among them
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])

    se1 = SuperEdge(sv1.name, sv2.name, edge_index)
    se2 = SuperEdge(sv2.name, sv3.name, edge_index)

    # create a supergraph
    sg = SuperGraph([sv1, sv2, sv3], [se1, se2])
