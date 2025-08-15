import torch

from kale.embed.nn import FCNet, RandomLayer


def test_random_layer_forward():
    input_dim_list = [64, 64]
    output_dim = 256
    batch_size = 32

    # Initialize RandomLayer model
    model = RandomLayer(input_dim_list, output_dim)

    # Create a mock input list
    input_list = [torch.randn(batch_size, dim) for dim in input_dim_list]

    # Forward pass through the model
    output = model(input_list)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == torch.Size([batch_size, output_dim]), "Output shape should match batch size and output_dim"


def test_fcnet_forward():
    dims = [64, 128, 64]
    batch_size = 32

    # Initialize FCNet model
    model = FCNet(dims, dropout=1)

    # Create mock input
    input_data = torch.randn(batch_size, dims[0])

    # Forward pass through the model
    output = model(input_data)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape == torch.Size([batch_size, dims[-1]]), "Output shape should match batch size and last dim"


def test_fcnet_minimal_inputs():
    # FCNet
    model = FCNet([1, 1])
    inp = torch.randn(2, 1)
    out = model(inp)
    assert out.shape == (2, 1)
