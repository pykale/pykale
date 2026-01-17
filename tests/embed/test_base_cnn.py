"""
Unit tests for BaseCNN utility class in kale.embed.base_cnn module.

Tests cover activation functions, convolution block creation, weight initialization,
pooling operations, embedding layers, and utility functions.
"""
import pytest
import torch
import torch.nn as nn

from kale.embed.base_cnn import BaseCNN


def _create_simple_cnn(in_channels=3):
    """Factory function to create a simple test CNN using BaseCNN utilities."""

    class SimpleCNN(BaseCNN):
        """Simple test CNN that uses BaseCNN utilities."""

        def __init__(self, in_channels=3):
            super().__init__()
            self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
                in_channels=in_channels,
                out_channels_list=[32, 64, 128],
                kernel_sizes=[3, 3, 3],
                conv_type="2d",
            )
            self._initialize_weights(method="kaiming")

        def forward(self, x):
            for conv, bn in zip(self.conv_layers, self.batch_norms):
                x = self._apply_activation(bn(conv(x)), "relu")
                x = self._apply_pooling(x, pool_type="max", pool_size=2)
            return self._flatten_features(x)

    return SimpleCNN(in_channels=in_channels)


# =============================================================================
# Activation Tests
# =============================================================================


def test_basecnn_apply_activation_relu():
    """Test ReLU activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    activated = model._apply_activation(x, "relu")
    assert activated.shape == x.shape
    assert torch.all(activated >= 0)  # ReLU should make all values non-negative


def test_basecnn_apply_activation_sigmoid():
    """Test sigmoid activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    activated = model._apply_activation(x, "sigmoid")
    assert activated.shape == x.shape
    assert torch.all(activated >= 0) and torch.all(activated <= 1)


def test_basecnn_apply_activation_invalid():
    """Test error handling for invalid activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError, match="Unsupported activation function"):
        model._apply_activation(x, "invalid_activation")


def test_basecnn_apply_activation_tanh():
    """Test tanh activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    activated = model._apply_activation(x, "tanh")
    assert activated.shape == x.shape
    assert torch.all(activated >= -1) and torch.all(activated <= 1)


def test_basecnn_apply_activation_leaky_relu():
    """Test leaky_relu activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    activated = model._apply_activation(x, "leaky_relu")
    assert activated.shape == x.shape


def test_basecnn_apply_activation_elu():
    """Test ELU activation."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    activated = model._apply_activation(x, "elu")
    assert activated.shape == x.shape


# =============================================================================
# Convolution Block Creation Tests
# =============================================================================


def test_basecnn_create_sequential_conv_blocks_2d():
    """Test creating sequential 2D convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=3, out_channels_list=[32, 64, 128], kernel_sizes=[3, 3, 3], conv_type="2d"
    )
    assert len(conv_layers) == 3
    assert len(batch_norms) == 3
    assert conv_layers[0].in_channels == 3
    assert conv_layers[0].out_channels == 32


def test_basecnn_create_sequential_conv_blocks_1d():
    """Test creating sequential 1D convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=3, out_channels_list=[32, 64], kernel_sizes=[3, 3], conv_type="1d"
    )
    assert len(conv_layers) == 2
    assert isinstance(conv_layers[0], nn.Conv1d)


def test_basecnn_create_sequential_conv_blocks_no_batch_norm():
    """Test creating sequential convolutional blocks without batch norm."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=3, out_channels_list=[32, 64], kernel_sizes=[3, 3], conv_type="2d", use_batch_norm=False
    )
    assert len(conv_layers) == 2
    # When use_batch_norm=False, Identity modules are returned
    assert len(batch_norms) == 2
    assert all(isinstance(bn, nn.Identity) for bn in batch_norms)


def test_basecnn_create_doubling_conv_blocks():
    """Test creating doubling convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms, global_pools = model._create_doubling_conv_blocks(
        input_channels=3, base_channels=32, num_layers=3
    )
    assert len(conv_layers) == 3
    assert conv_layers[0].out_channels == 32
    assert conv_layers[1].out_channels == 64
    assert conv_layers[2].out_channels == 128


def test_basecnn_create_progressive_conv_blocks():
    """Test creating progressive convolutional blocks with custom multipliers."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_progressive_conv_blocks(
        in_channels=3, base_channels=32, num_layers=3, multipliers=[1, 2, 4], kernel_sizes=3, conv_type="2d"
    )
    assert len(conv_layers) == 3
    assert conv_layers[0].out_channels == 32
    assert conv_layers[1].out_channels == 64
    assert conv_layers[2].out_channels == 128


def test_basecnn_create_sequential_conv_blocks_invalid_conv_type():
    """Test error handling for invalid conv_type."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="conv_type must be '1d' or '2d'"):
        model._create_sequential_conv_blocks(3, [32, 64], [3, 3], conv_type="3d")


def test_basecnn_create_sequential_conv_blocks_empty_channels():
    """Test error handling for empty out_channels_list."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="out_channels_list cannot be empty"):
        model._create_sequential_conv_blocks(3, [], [3], conv_type="2d")


def test_basecnn_create_sequential_conv_blocks_negative_in_channels():
    """Test error handling for negative in_channels."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="in_channels must be positive"):
        model._create_sequential_conv_blocks(-1, [32, 64], [3, 3], conv_type="2d")


def test_basecnn_create_sequential_conv_blocks_kernel_mismatch():
    """Test error handling for mismatched kernel_sizes length."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="kernel_sizes length.*must match"):
        model._create_sequential_conv_blocks(3, [32, 64, 128], [3, 3], conv_type="2d")  # Only 2 kernels for 3 layers


def test_basecnn_create_sequential_conv_blocks_stride_mismatch():
    """Test error handling for mismatched strides length."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="strides length.*must match"):
        model._create_sequential_conv_blocks(
            3, [32, 64, 128], [3, 3, 3], strides=[1, 1], conv_type="2d"
        )  # Only 2 strides for 3 layers


def test_basecnn_create_sequential_conv_blocks_padding_mismatch():
    """Test error handling for mismatched paddings length."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="paddings length.*must match"):
        model._create_sequential_conv_blocks(
            3, [32, 64, 128], [3, 3, 3], paddings=[1, 1], conv_type="2d"
        )  # Only 2 paddings for 3 layers


def test_basecnn_create_progressive_conv_blocks_multiplier_mismatch():
    """Test error handling for mismatched multipliers length."""
    model = BaseCNN()
    with pytest.raises(ValueError, match="Length of multipliers.*must match num_layers"):
        model._create_progressive_conv_blocks(
            in_channels=3, base_channels=32, num_layers=3, multipliers=[1, 2], kernel_sizes=3, conv_type="2d"
        )  # Only 2 multipliers for 3 layers


# =============================================================================
# Weight Initialization Tests
# =============================================================================


def test_basecnn_initialize_weights_kaiming():
    """Test Kaiming weight initialization."""

    class TestCNN(BaseCNN):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self._initialize_weights(method="kaiming")

    model = TestCNN()
    assert model.conv.weight is not None


def test_basecnn_initialize_weights_xavier():
    """Test Xavier weight initialization."""

    class TestCNN(BaseCNN):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self._initialize_weights(method="xavier")

    model = TestCNN()
    assert model.conv.weight is not None


def test_basecnn_initialize_weights_normal():
    """Test normal weight initialization."""

    class TestCNN(BaseCNN):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self._initialize_weights(method="normal")

    model = TestCNN()
    assert model.conv.weight is not None


def test_basecnn_initialize_weights_uniform():
    """Test uniform weight initialization."""

    class TestCNN(BaseCNN):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)
            self._initialize_weights(method="uniform")

    model = TestCNN()
    assert model.conv.weight is not None


def test_basecnn_initialize_weights_invalid_method():
    """Test error handling for invalid weight initialization method."""

    class TestCNN(BaseCNN):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)

    model = TestCNN()
    with pytest.raises(ValueError, match="Unsupported initialization method"):
        model._initialize_weights(method="invalid_method")


# =============================================================================
# Utility Functions Tests
# =============================================================================


def test_basecnn_flatten_features():
    """Test feature flattening."""
    model = BaseCNN()
    x = torch.randn(2, 64, 4, 4)
    flattened = model._flatten_features(x)
    assert flattened.dim() == 2
    assert flattened.shape == (2, 64 * 4 * 4)


def test_basecnn_get_conv_output_size():
    """Test output size calculation for convolution."""
    model = BaseCNN()
    # For a 28x28 input with 3x3 kernel, stride 1, padding 0:
    # Output = floor((28 - 3 + 2*0)/1 + 1) = 26
    output_size = model._get_conv_output_size(input_size=28, kernel_size=3, padding=0, stride=1)
    assert output_size == 26


# =============================================================================
# Pooling Tests
# =============================================================================


def test_basecnn_apply_pooling_max_2d():
    """Test max pooling for 2D tensors."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    pooled = model._apply_pooling(x, pool_type="max", pool_size=2)
    assert pooled.shape == (2, 3, 4, 4)


def test_basecnn_apply_pooling_avg_2d():
    """Test average pooling for 2D tensors."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    pooled = model._apply_pooling(x, pool_type="avg", pool_size=2)
    assert pooled.shape == (2, 3, 4, 4)


def test_basecnn_apply_pooling_adaptive_max():
    """Test adaptive max pooling."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    pooled = model._apply_pooling(x, pool_type="adaptive_max", adaptive_output_size=(4, 4))
    assert pooled.shape == (2, 3, 4, 4)


def test_basecnn_apply_pooling_1d():
    """Test max pooling for 1D tensors."""
    model = BaseCNN()
    x = torch.randn(2, 3, 16)
    pooled = model._apply_pooling(x, pool_type="max", pool_size=2)
    assert pooled.shape == (2, 3, 8)


def test_basecnn_apply_pooling_adaptive_avg_2d():
    """Test adaptive average pooling for 2D."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    pooled = model._apply_pooling(x, pool_type="adaptive_avg", adaptive_output_size=(4, 4))
    assert pooled.shape == (2, 3, 4, 4)


def test_basecnn_apply_pooling_adaptive_avg_1d():
    """Test adaptive average pooling for 1D."""
    model = BaseCNN()
    x = torch.randn(2, 3, 16)
    pooled = model._apply_pooling(x, pool_type="adaptive_avg", adaptive_output_size=8)
    assert pooled.shape == (2, 3, 8)


def test_basecnn_apply_pooling_avg_1d():
    """Test average pooling for 1D."""
    model = BaseCNN()
    x = torch.randn(2, 3, 16)
    pooled = model._apply_pooling(x, pool_type="avg", pool_size=2)
    assert pooled.shape == (2, 3, 8)


def test_basecnn_apply_pooling_adaptive_error():
    """Test error handling for adaptive pooling without output size."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match="adaptive_output_size must be specified"):
        model._apply_pooling(x, pool_type="adaptive_max")


def test_basecnn_apply_pooling_invalid_type():
    """Test error handling for invalid pool_type."""
    model = BaseCNN()
    x = torch.randn(2, 3, 8, 8)
    with pytest.raises(ValueError, match="Unsupported pool_type"):
        model._apply_pooling(x, pool_type="invalid_pool")


# =============================================================================
# Embedding Tests
# =============================================================================


def test_basecnn_create_embedding_layer():
    """Test creating embedding layer."""
    model = BaseCNN()
    embedding = model._create_embedding_layer(num_embeddings=100, embedding_dim=128)
    assert isinstance(embedding, nn.Embedding)
    assert embedding.weight.shape == (100, 128)


def test_basecnn_create_embedding_layer_with_padding():
    """Test creating embedding layer with padding."""
    model = BaseCNN()
    embedding = model._create_embedding_layer(num_embeddings=100, embedding_dim=128, padding_idx=0)
    assert embedding.padding_idx == 0


# =============================================================================
# Integration Test
# =============================================================================


def test_basecnn_simple_cnn_forward():
    """Test forward pass through a simple CNN using BaseCNN."""
    model = _create_simple_cnn(in_channels=3)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.dim() == 2  # Should be flattened
    assert output.shape[0] == 4  # Batch size preserved
