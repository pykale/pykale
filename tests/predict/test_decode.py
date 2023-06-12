import torch

from kale.predict.decode import MLPDecoder


def test_mlp_decoder():
    # Test with additional layers
    in_dim, hidden_dim, out_dim = 8, 16, 32
    include_additional_layers = True
    dropout_rate = 0.1
    mlp_decoder = MLPDecoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout_rate=dropout_rate,
        include_additional_layers=include_additional_layers,
    )
    assert mlp_decoder.fc1.weight.size() == (hidden_dim, in_dim)
    assert mlp_decoder.fc2.weight.size() == (hidden_dim, hidden_dim)
    assert mlp_decoder.fc3.weight.size() == (out_dim, hidden_dim)
    assert mlp_decoder.fc4.weight.size() == (1, out_dim)
    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.size() == (16, 1)

    # Test without additional layers
    in_dim, hidden_dim, out_dim = 8, 16, 2
    include_additional_layers = False
    mlp_decoder = MLPDecoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout_rate=dropout_rate,
        include_additional_layers=include_additional_layers,
    )
    assert mlp_decoder.fc1.weight.size() == (hidden_dim, in_dim)
    assert mlp_decoder.fc2.weight.size() == (out_dim, hidden_dim)
    assert not hasattr(mlp_decoder, "fc3")  # There should be no fc3 layer
    assert not hasattr(mlp_decoder, "fc4")  # There should be no fc4 layer
    input_batch = torch.randn((16, in_dim))
    output = mlp_decoder(input_batch)
    assert output.size() == (16, out_dim)
