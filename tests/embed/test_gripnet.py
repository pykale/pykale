import os

import pytest
import torch

from kale.embed.gripnet import GripNetExternalModule, GripNetInternalModule, TypicalGripNetEncoder
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


# create three supervertices
node_feat = torch.randn(4, 20)
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)

supervertex1 = SuperVertex("1", node_feat, edge_index)
supervertex2 = SuperVertex("2", node_feat, edge_index, edge_type)
supervertex3 = SuperVertex("3", node_feat, edge_index, edge_type)

# determine the supervertices among them
edge_index = torch.tensor([[0, 1, 2, 3], [1, 1, 3, 3]])

superedge1 = SuperEdge(supervertex1.name, supervertex3.name, edge_index)
superedge2 = SuperEdge(supervertex2.name, supervertex3.name, edge_index)

# create a supergraph
supergraph = SuperGraph([supervertex1, supervertex2, supervertex3], [superedge1, superedge2])


def test_gripnet_internal_module1():
    """GripNet Internal Module Test for start supervertex"""

    setting1 = SuperVertexParaSetting("start_svertex", 20, [10, 10])
    internal_module1 = GripNetInternalModule(
        supervertex1.n_node_feat, supervertex1.n_edge_type, supervertex1.if_start_supervertex, setting1
    )

    x = torch.randn(20, 20)
    y = internal_module1(x, supervertex3.edge_index)

    assert y.shape[1] == internal_module1.out_dim


def test_gripnet_internal_module2():
    """GripNet Internal Module Test for end supervertex"""

    setting2 = SuperVertexParaSetting("task_svertex", 20, [15, 10], exter_agg_dim={"1": 20, "2": 20}, mode="cat")
    internal_module2 = GripNetInternalModule(
        supervertex3.n_node_feat, supervertex3.n_edge_type, supervertex3.if_start_supervertex, setting2
    )

    x = torch.randn(20, 20)
    x1 = torch.matmul(x, internal_module2.embedding)
    x2 = torch.randn(20, 40)
    xx = torch.cat((x1, x2), dim=1)

    range_list = torch.LongTensor([[0, 2], [2, 4]])
    edge_weight = torch.randn(4)

    y = internal_module2(xx, supervertex3.edge_index, supervertex3.edge_type, range_list, edge_weight)

    assert y.shape[1] == internal_module2.out_dim


def test_gripnet_external_module():
    """GripNet External Module Test"""

    external_module = GripNetExternalModule(8, 7, 5)
    x = torch.randn((4, 8))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 4, 3, 4]])
    y = external_module(x, edge_index)

    assert y.shape[0] == 5
    assert y.shape[1] == 7
