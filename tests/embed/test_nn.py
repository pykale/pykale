import torch

from kale.embed.nn import RandomLayer


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
