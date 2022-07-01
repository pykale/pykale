import numpy.testing as testing
import pytest
import torch
import torch.nn as nn

from kale.embed.attention_cnn import CNNTransformer, SelfAttention, TransformerBlock, TransformerSENet
from kale.prepdata.tensor_reshape import seq_to_spatial
from kale.utils.seed import set_seed


def test_shapes():
    set_seed(36)

    CNN_OUT_HEIGHT = 32
    CNN_OUT_WIDTH = 32
    CNN_OUT_CHANNELS = 256

    cnn = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=CNN_OUT_CHANNELS, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=2)
    )
    cnn_transformer = CNNTransformer(
        cnn=cnn,
        cnn_output_shape=(-1, CNN_OUT_CHANNELS, CNN_OUT_HEIGHT, CNN_OUT_WIDTH),
        num_layers=4,
        num_heads=4,
        dim_feedforward=1024,
        dropout=0.1,
        output_type="spatial",
        positional_encoder=None,
    )

    cnn_transformer.eval()

    BATCH_SIZE = 2

    input_batch = torch.randn((BATCH_SIZE, 3, 64, 64))

    out_spatial = cnn_transformer(input_batch)
    cnn_transformer.output_type = "sequence"
    out_seq = cnn_transformer(input_batch)

    assert out_spatial.size() == (BATCH_SIZE, CNN_OUT_CHANNELS, CNN_OUT_HEIGHT, CNN_OUT_WIDTH)
    assert out_seq.size() == (CNN_OUT_HEIGHT * CNN_OUT_WIDTH, BATCH_SIZE, CNN_OUT_CHANNELS)

    # simply reshape them both to have same shape
    out_spatial_2 = seq_to_spatial(out_seq, CNN_OUT_HEIGHT, CNN_OUT_WIDTH)

    testing.assert_almost_equal(out_spatial.detach().numpy(), out_spatial_2.detach().numpy())


@pytest.mark.parametrize("causal", [True, False])
def test_SelfAttention(causal):
    INPUT_BATCH = torch.randn(2, 8, 8)
    model = SelfAttention(8, 2, 0.1, 0.1, causal, 9)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (2, 8, 8)


def test_TransformerBlock():
    INPUT_BATCH = torch.randn(2, 8, 8)
    model = TransformerBlock(8, 8, 0.1, 0.1, 0.1, 9, 10, False)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (2, 8, 8)


def test_TransformerSENet():
    INPUT_BATCH = torch.randn(2, 8, 32)
    model = TransformerSENet(input_size=32, n_channel=10, output_size=5)
    model.eval()
    output_batch = model(INPUT_BATCH)
    assert output_batch.size() == (2, 8, 5)
