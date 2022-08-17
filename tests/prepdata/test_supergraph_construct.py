import torch

from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting

# create three supervertices
node_feat = torch.randn(4, 20)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
edge_type = torch.tensor([0, 0, 1, 1])
edge_weight = torch.randn(4)


def test_supervertex():
    """Test the supervertex construction."""

    supervertex1 = SuperVertex("1", node_feat, edge_index)
    supervertex2 = SuperVertex("2", node_feat, edge_index, edge_type)
    supervertex3 = SuperVertex("3", node_feat, edge_index, edge_type, edge_weight)

    assert supervertex1.__repr__()
    assert supervertex2.edge_weight is None
    assert supervertex3.edge_weight is not None

    supervertex1.add_in_supervertex("2")
    assert supervertex1.in_supervertex_list == ["2"]

    supervertex2.add_out_supervertex("3")
    assert supervertex2.out_supervertex_list == ["3"]


supervertex1 = SuperVertex("1", node_feat, edge_index)
supervertex2 = SuperVertex("2", node_feat, edge_index, edge_type)
supervertex3 = SuperVertex("3", node_feat, edge_index, edge_type)

# determine the supervertices among them
edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])


def test_superedge():
    """Test the superedge construction."""

    superedge1 = SuperEdge("1", "2", edge_index)
    superedge2 = SuperEdge("2", "3", edge_index, edge_weight)

    assert superedge1.__repr__()
    assert superedge1.edge_weight is None
    assert superedge2.edge_weight is not None


superedge1 = SuperEdge(supervertex1.name, supervertex3.name, edge_index)
superedge2 = SuperEdge(supervertex2.name, supervertex3.name, edge_index)


def test_supergraph():
    """Test the supergraph construction."""

    # create a supergraph
    supergraph = SuperGraph([supervertex1, supervertex2, supervertex3], [superedge1, superedge2])

    assert supergraph.num_supervertex == 3
    assert supergraph.num_superedge == 2
    assert supergraph.topological_order == ["1", "2", "3"] or supergraph.topological_order == ["2", "1", "3"]
    assert set(supergraph.supervertex_dict["3"].in_supervertex_list) == {"1", "2"}

    assert supergraph.__repr__()


def test_supervertex_para_setting():
    """Test the supervertex parameter setting with and without exter_agg_dim and mode."""

    setting1 = SuperVertexParaSetting("2", 20, [10, 10])
    setting2 = SuperVertexParaSetting("3", 20, [15, 10], exter_agg_dim={"sv1": 20, "sv2": 20}, mode="cat")

    supergraph = SuperGraph([supervertex2, supervertex3], [superedge2], [setting1, setting2])
    assert supergraph.supervertex_setting_dict
