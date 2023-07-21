import pytest
import torch

from kale.embed.image_cnn import (
    LeNet,
    ResNet18Feature,
    ResNet34Feature,
    ResNet50Feature,
    ResNet101Feature,
    ResNet152Feature,
    SimpleCNNBuilder,
    SmallCNNFeature,
)

BATCH_SIZE = 64

# the default input shape is batch_size * num_channel * height * weight
INPUT_BATCH = torch.randn(BATCH_SIZE, 3, 32, 32)
PARAM = [
    (ResNet18Feature, 512),
    (ResNet34Feature, 512),
    (ResNet50Feature, 2048),
    (ResNet101Feature, 2048),
    (ResNet152Feature, 2048),
]


def test_smallcnnfeature_shapes():
    model = SmallCNNFeature()
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 128)


def test_simplecnnbuilder_shapes():
    model = SimpleCNNBuilder(
        conv_layers_spec=[[16, 3], [32, 3], [64, 3], [32, 1], [64, 3], [128, 3], [256, 3], [64, 1]]
    )
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 64, 8, 8)


@pytest.mark.parametrize("param", PARAM)
def test_shapes(param):
    model, out_size = param
    model = model(pretrained=False)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, out_size)


def test_lenet_output_shapes():
    input_channels = 3
    output_channels = 6
    additional_layers = 2
    lenet = LeNet(input_channels, output_channels, additional_layers)

    x = torch.randn(16, 3, 32, 32)
    output = lenet(x)
    assert output.shape == (16, 24, 4, 4), "Unexpected output shape"
