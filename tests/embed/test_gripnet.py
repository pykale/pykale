import os

import pytest
import torch

from kale.embed.gripnet import GripNet, TypicalGripNetEncoder
from kale.prepdata.supergraph_construct import SuperEdge, SuperGraph, SuperVertex, SuperVertexParaSetting
from kale.utils.download import download_file_by_url

pose_url = "https://github.com/pykale/data/raw/main/graphs/pose_pyg_2.pt"


@pytest.fixture(scope="module")
def pose_data(download_path):
    download_file_by_url(pose_url, download_path, "pose.pt", "pt")
    return torch.load(os.path.join(download_path, "pose.pt"))


def test_gripnet_encoder(pose_data):
    gg_layers = [32, 16, 16]
    gd_layers = [16, 32]
    dd_layers = [sum(gd_layers), 16]
    gripnet = TypicalGripNetEncoder(
        gg_layers, gd_layers, dd_layers, pose_data.n_d_node, pose_data.n_g_node, pose_data.n_dd_edge_type
    )
    gripnet(
        pose_data.g_feat,
        pose_data.gg_edge_index,
        pose_data.edge_weight,
        pose_data.gd_edge_index,
        pose_data.train_idx,
        pose_data.train_et,
        pose_data.train_range,
    )
    assert gripnet.source_graph.conv_list[0].__repr__() == "GCNEncoderLayer(32, 16)"
    assert gripnet.source_graph.conv_list[1].__repr__() == "GCNEncoderLayer(16, 16)"
    assert gripnet.s2t_graph.conv.__repr__() == "GCNEncoderLayer(64, 16)"
    assert gripnet.target_graph.conv_list[0].__repr__() == "RGCNEncoderLayer(48, 16, num_relations=854)"


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
