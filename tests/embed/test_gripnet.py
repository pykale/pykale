import pytest
import torch

from kale.embed.gripnet import GripNet
from kale.predict.decode import GripNetLinkPrediction
from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting


@pytest.mark.parametrize("mode,test_in_channels,test_out_channels", [("cat", 90, 115), ("add", 30, 55)])
def test_gripnet(mode, test_in_channels, test_out_channels):
    """GripNet and SuperGraph Test"""
    # create three supervertices
    node_feat = torch.randn(4, 20)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    supervertex1 = SuperVertex("1", node_feat, edge_index)
    supervertex2 = SuperVertex("2", node_feat, edge_index, edge_type)
    supervertex3 = SuperVertex("3", node_feat, edge_index, edge_type)

    # determine the superedges among them
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])

    superedge1 = SuperEdge(supervertex1.name, supervertex3.name, edge_index)
    superedge2 = SuperEdge(supervertex2.name, supervertex3.name, edge_index)

    # create a supergraph
    supergraph = SuperGraph([supervertex1, supervertex2, supervertex3], [superedge1, superedge2])

    setting1 = SuperVertexParaSetting("1", 20, [10, 10])
    setting2 = SuperVertexParaSetting("2", 20, [10, 10])
    setting3 = SuperVertexParaSetting("3", 30, [15, 10], exter_agg_channels_dict={"1": 30, "2": 30}, mode=mode)

    supergraph.set_supergraph_para_setting([setting1, setting2, setting3])
    gripnet = GripNet(supergraph)

    assert (
        gripnet.supervertex_module_dict["3"][-1].inter_agg_layers[0].in_channels == test_in_channels
    ), "ValueError: invalid exter_agg_channels_dict settings in the task vertex."

    y = gripnet()
    error_message = "ValueError: dimension mismatch in the task vertex"

    assert y.shape[1] == test_out_channels, error_message

    if mode == "cat":
        assert gripnet.out_embed_dict["1"].shape[1] == 20 + 10 + 10, error_message
        assert gripnet.out_embed_dict["2"].shape[1] == 20 + 10 + 10, error_message

    # general tests
    assert supervertex1.__repr__()
    assert superedge1.__repr__()
    assert gripnet.__repr__() is not None

    # test GripNetLinkPrediction predictor
    GripNetLinkPrediction(supergraph, 0.1)
