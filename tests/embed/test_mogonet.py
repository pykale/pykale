import pytest
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

from kale.embed.mogonet import MogonetGCN, MogonetGCNConv


@pytest.fixture(scope="module")
def test_data():
    x = torch.Tensor([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]])
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 2]])
    edge_weight = torch.tensor([0.5, 0.2, 0.3, 0.4])
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)

    return Data(x=x, edge_index=edge_index, adj_t=adj_t)


def test_mogonet_gcn_conv_model_shape():
    in_channels = 2
    out_channels = 8
    conv = MogonetGCNConv(in_channels=in_channels, out_channels=out_channels)

    # Test the shape of the module
    assert conv.in_channels == in_channels
    assert conv.out_channels == out_channels

    # Test the shape of the weight tensor
    assert conv.weight.shape == (in_channels, out_channels)


def test_mogonet_gcn_conv_no_bias():
    in_channels = 2
    out_channels = 8
    conv = MogonetGCNConv(in_channels=in_channels, out_channels=out_channels, bias=False)
    assert conv.bias is None


def test_mogonet_gcn_conv_reset_parameters():
    in_channels = 2
    out_channels = 8
    conv = MogonetGCNConv(in_channels=in_channels, out_channels=out_channels)
    # Set parameters to a fixed value
    with torch.no_grad():
        for param in conv.parameters():
            param.fill_(1.0)
    # Reset the parameters
    conv.reset_parameters()
    # Check that the parameters are now different
    for param in conv.parameters():
        assert torch.any(param != 1.0)


def test_mogonet_gcn_conv_output(test_data):
    in_channels = 2
    out_channels = 8
    num_samples = 3
    conv = MogonetGCNConv(in_channels=in_channels, out_channels=out_channels)

    # Test the forward pass
    x = test_data.x
    adj_t = test_data.adj_t
    output = conv(x, adj_t)
    assert output.shape == (num_samples, out_channels)

    # Test the message function
    x_j = torch.randn(3, 5)
    msg = conv.message(x_j)
    assert msg.shape == (3, 5)

    # Test the message_and_aggregate function
    agg = conv.message_and_aggregate(adj_t, x)
    assert agg.shape == (num_samples, in_channels)

    # Test the update function
    aggr_out = torch.randn(num_samples, out_channels)
    updated = conv.update(aggr_out)
    assert updated.shape == aggr_out.shape


def test_mogonet_gcn_model_shape():
    model = MogonetGCN(in_channels=2, hidden_channels=[4, 5, 6], dropout=0.5)
    assert isinstance(model.conv1, MogonetGCNConv)
    assert isinstance(model.conv2, MogonetGCNConv)
    assert isinstance(model.conv3, MogonetGCNConv)

    assert model.conv1.weight.shape == (2, 4)
    assert model.conv2.weight.shape == (4, 5)
    assert model.conv3.weight.shape == (5, 6)


def test_mogonet_gcn_output(test_data):
    model = MogonetGCN(in_channels=2, hidden_channels=[4, 5, 6], dropout=0.5)
    x = test_data.x
    adj_t = test_data.adj_t
    output = model(x, adj_t)
    assert output.shape == (3, 6)
