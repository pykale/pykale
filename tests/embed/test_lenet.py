import torch

from kale.embed.lenet import GlobalPooling2D, LeNet


def test_lenet_output_shapes():
    input_channels = 3
    output_channels = 6
    additional_layers = 2
    lenet = LeNet(input_channels, output_channels, additional_layers)

    x = torch.randn(16, 3, 32, 32)
    output = lenet(x)
    assert output.shape == (16, 24, 4, 4), "Unexpected output shape"


def test_global_pooling_2d():
    global_pooling = GlobalPooling2D()
    x = torch.randn(16, 8, 16, 16)
    output = global_pooling(x)
    assert output.shape == (16, 8), "Unexpected output shape from GlobalPooling2D"
