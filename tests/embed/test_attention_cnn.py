import numpy.testing as testing
import pytest
import torch
import torch.nn as nn

from kale.embed.attention import BANLayer
from kale.embed.cnn import CNNTransformer
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


def test_ban_layer_forward():
    v_dim = 64
    q_dim = 64
    h_dim = 128
    h_out = 8
    batch_size = 32
    v_seq_len = 10
    q_seq_len = 12

    # Initialize BANLayer model
    model = BANLayer(v_dim, q_dim, h_dim, h_out)

    # Create mock input tensors for v and q
    v = torch.randn(batch_size, v_seq_len, v_dim)
    q = torch.randn(batch_size, q_seq_len, q_dim)

    # Forward pass through the model
    logits, att_maps = model(v, q)

    # Check output types and shape
    assert isinstance(logits, torch.Tensor), "Logits should be a tensor"
    assert isinstance(att_maps, torch.Tensor), "Attention maps should be a tensor"
    assert logits.shape == torch.Size([batch_size, h_dim]), "Logits shape should match batch size and hidden_dim"
    assert att_maps.shape == torch.Size([batch_size, h_out, v_seq_len, q_seq_len]), "Attention maps shape should match"


def test_ban_layer_attention_pooling():
    v_dim = 16
    q_dim = 16
    h_dim = 32
    h_out = 2
    batch_size = 2
    seq_len = 3
    model = BANLayer(v_dim, q_dim, h_dim, h_out)
    v = torch.randn(batch_size, seq_len, h_dim * model.num_att_maps)
    q = torch.randn(batch_size, seq_len, h_dim * model.num_att_maps)
    att_map = torch.randn(batch_size, seq_len, seq_len)
    pooled = model.attention_pooling(v, q, att_map)
    assert pooled.shape[0] == batch_size


@pytest.mark.parametrize(
    "v_dim, q_dim, hidden_dim, num_out_heads, num_att_maps, batch_size, seq_len_v, seq_len_q",
    [
        # This set triggers the `else` branch (uses `h_net`)
        (32, 32, 16, 64, 3, 2, 10, 12),
        # This set triggers the `if num_out_heads <= self.c` branch (uses h_mat + einsum)
        (16, 16, 32, 8, 3, 4, 5, 6),
    ],
)
def test_ban_layer_forward_hout_leq_c(
    v_dim, q_dim, hidden_dim, num_out_heads, num_att_maps, batch_size, seq_len_v, seq_len_q
):
    model = BANLayer(v_dim, q_dim, hidden_dim, num_out_heads=num_out_heads, num_att_maps=num_att_maps)
    v = torch.randn(batch_size, seq_len_v, v_dim)
    q = torch.randn(batch_size, seq_len_q, q_dim)
    logits, att_maps = model(v, q)
    assert logits.shape[0] == batch_size
    assert att_maps.shape[0] == batch_size
    # Also test with softmax=True
    logits_sm, att_maps_sm = model(v, q, softmax=True)
    assert logits_sm.shape[0] == batch_size
    assert att_maps_sm.shape[0] == batch_size
