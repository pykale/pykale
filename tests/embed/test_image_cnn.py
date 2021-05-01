import pytest
import torch

from kale.embed.image_cnn import (
    ResNet18Feature,
    ResNet34Feature,
    ResNet50Feature,
    ResNet101Feature,
    ResNet152Feature,
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


@pytest.mark.parametrize("param", PARAM)
def test_shapes(param):
    model, out_size = param
    model = model(pretrained=False)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, out_size)
