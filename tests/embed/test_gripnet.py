import os

import pytest
import torch

from kale.embed.gripnet import TypicalGripNetEncoder
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
