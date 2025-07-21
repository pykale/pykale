import torch
import torch.nn.functional as F

from kale.embed.gcn import GCNEncoder, MolecularGCN
from tests.helpers.mock_graph import create_mock_batch_graph


def test_molecular_gcn_with_activation():
    in_feats = 10
    dim_embedding = 16
    hidden_feats = [16, 8]
    batch_size = 2
    model = MolecularGCN(in_feats, dim_embedding=dim_embedding, hidden_feats=hidden_feats, activation=F.relu)
    batch_graph = create_mock_batch_graph(batch_size=batch_size, in_feats=in_feats)
    output = model(batch_graph)
    assert output.shape[-1] == hidden_feats[-1]


def test_molecular_gcn_minimal_inputs():
    model = MolecularGCN(1, dim_embedding=1, hidden_feats=[1])
    model.eval()
    batch_graph = create_mock_batch_graph(batch_size=2, num_nodes=1, num_edges=1, in_feats=1)
    out = model(batch_graph)
    assert out.shape == (2, 1, 1)


def test_gcn_encoder():
    gcn_encoder = GCNEncoder(in_channel=8, out_channel=32).eval()
    assert gcn_encoder.conv1.__repr__() == "GCNConv(8, 8)"
    assert gcn_encoder.conv2.__repr__() == "GCNConv(8, 16)"
    assert gcn_encoder.conv3.__repr__() == "GCNConv(16, 32)"
    N1, N2 = 4, 5
    x = torch.randn(N1 + N2, 8)
    batch = torch.tensor([0 for _ in range(N1)] + [1 for _ in range(N2)])
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3, 3, 5, 7, 8], [1, 2, 3, 0, 0, 0, 2, 6, 6, 0]], dtype=torch.long)
    row, col = edge_index
    from torch_sparse import SparseTensor

    adj = SparseTensor(row=row, col=col, value=None, sparse_sizes=(9, 9))

    out1 = gcn_encoder(x, edge_index, batch)
    assert out1.size() == (2, 32)
    # assert the consistency between types of LongTensor and SparseTensor.
    assert torch.allclose(gcn_encoder(x, adj.t(), batch), out1, atol=1e-6)
