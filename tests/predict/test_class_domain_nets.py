import pytest
import torch

from kale.predict.class_domain_nets import (
    ClassNetSmallImage,
    ClassNetVideo,
    ClassNetVideoConv,
    DomainNetSmallImage,
    DomainNetVideo,
    SoftmaxNet,
)
from kale.utils.seed import set_seed

set_seed(36)
BATCH_SIZE = 2
# The default input shape for basic ClassNet and DomainNet is batch_size * dimension. However, for ClassNetVideoConv,
# the input is the output of the I3D last average pooling layer and the shape is
# batch_size * num_channel * frame_per_segment * height * weight.
INPUT_BATCH = torch.randn(BATCH_SIZE, 128)
INPUT_BATCH_AVERAGE = torch.randn(BATCH_SIZE, 1024, 1, 1, 1)
CLASSNET_MODEL = [ClassNetSmallImage, ClassNetVideo]
DOMAINNET_MODEL = [DomainNetSmallImage, DomainNetVideo]


def test_softmaxnet_shapes():
    model = SoftmaxNet(input_dim=128, n_classes=8)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 8)


@pytest.mark.parametrize("model", CLASSNET_MODEL)
def test_classnet_shapes(model):
    model = model(input_size=128, n_class=8)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 8)


def test_classnetvideoconv_shapes():
    model = ClassNetVideoConv(n_class=8)
    model.eval()
    output_batch = model(INPUT_BATCH_AVERAGE)
    assert output_batch.size() == (BATCH_SIZE, 8, 1, 1, 1)


@pytest.mark.parametrize("model", DOMAINNET_MODEL)
def test_domainnet_shapes(model):
    model = model(input_size=128)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (BATCH_SIZE, 2)
