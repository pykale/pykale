import torch

from kale.predict.decode import MLPDecoder


def test_mlp_decoder():
    in_dim, hidden_dim, out_dim = 8, 16, 32
    mlp_decoder = MLPDecoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    assert mlp_decoder.fc1.weight.size() == (hidden_dim, in_dim)
    assert mlp_decoder.fc3.weight.size() == (out_dim, hidden_dim)
    assert mlp_decoder.fc4.weight.size() == (1, out_dim)
    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.size() == (16, 1)
