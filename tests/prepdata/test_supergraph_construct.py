import torch

from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting

# create three supervertices
node_feat = torch.randn(4, 20)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
edge_type = torch.tensor([0, 0, 1, 1])

sv1 = SuperVertex("1", node_feat, edge_index)
sv2 = SuperVertex("2", node_feat, edge_index, edge_type)
sv3 = SuperVertex("3", node_feat, edge_index, edge_type)

# determine the supervertices among them
edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])

se1 = SuperEdge(sv1.name, sv3.name, edge_index)
se2 = SuperEdge(sv2.name, sv3.name, edge_index)


def test_supergraph():
    """Test the supergraph construction."""

    # create a supergraph
    sg = SuperGraph([sv1, sv2, sv3], [se1, se2])

    assert sg.n_supervertex == 3
    assert sg.n_superedge == 2
    assert sg.topological_order == ["1", "2", "3"] or sg.topological_order == ["2", "1", "3"]
    assert set(sg.supervertex_dict["3"].in_supervertex_list) == {"1", "2"}


def test_supervertex_para_setting():
    """Test the supervertex parameter setting with and without exter_agg_dim and mode."""

    SuperVertexParaSetting("start_svertex", 20, [10, 10])
    SuperVertexParaSetting("task_svertex", 20, [15, 10], exter_agg_dim={"sv1": 20, "sv2": 20}, mode="cat")
