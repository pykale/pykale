import torch

from kale.embed.seq_nn import CNNEncoder, GCNEncoder


def test_cnn_encoder():
    num_embeddings, embedding_dim = 64, 128
    sequence_length = 85
    num_filters = 32
    filter_length = 8
    cnn_encoder = CNNEncoder(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        num_kernels=num_filters,
        kernel_length=filter_length,
    ).eval()
    # assert cnn encoder shape
    assert cnn_encoder.embedding.weight.size() == (num_embeddings + 1, embedding_dim)
    assert cnn_encoder.conv1.__repr__() == "Conv1d(85, 32, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv2.__repr__() == "Conv1d(32, 64, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv3.__repr__() == "Conv1d(64, 96, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.global_max_pool.__repr__() == "AdaptiveMaxPool1d(output_size=1)"

    input_batch = torch.randint(high=num_embeddings, size=(8, sequence_length))
    output_encoding = cnn_encoder(input_batch)
    # assert output shape
    assert output_encoding.size() == (8, 96)


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
