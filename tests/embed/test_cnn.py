"""
Unit tests for CNN module including BaseCNN, CNNEncoder, ProteinCNN,
ContextCNNGeneric, and CNNTransformer.
"""
import pytest
import torch
import torch.nn as nn

from kale.embed.cnn import BaseCNN, CNNEncoder, CNNTransformer, ContextCNNGeneric, ProteinCNN
from kale.embed.image_cnn import SimpleCNNBuilder


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
# BaseCNN Tests
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
    assert torch.all(activated >= 0) and torch.all(activated <= 1)  # Sigmoid output range


def test_basecnn_apply_activation_invalid():
    """Test invalid activation function."""
    model = BaseCNN()
    x = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError):
        model._apply_activation(x, "invalid_activation")


def test_basecnn_create_sequential_conv_blocks_2d():
    """Test creating 2D convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=3, out_channels_list=[32, 64, 128], kernel_sizes=3, conv_type="2d"
    )
    assert len(conv_layers) == 3
    assert len(batch_norms) == 3
    assert isinstance(conv_layers[0], nn.Conv2d)
    assert isinstance(batch_norms[0], nn.BatchNorm2d)


def test_basecnn_create_sequential_conv_blocks_1d():
    """Test creating 1D convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=1, out_channels_list=[16, 32], kernel_sizes=3, conv_type="1d"
    )
    assert len(conv_layers) == 2
    assert isinstance(conv_layers[0], nn.Conv1d)
    assert isinstance(batch_norms[0], nn.BatchNorm1d)


def test_basecnn_create_sequential_conv_blocks_no_batch_norm():
    """Test creating convolutional blocks without batch normalization."""
    model = BaseCNN()
    conv_layers, batch_norms = model._create_sequential_conv_blocks(
        in_channels=3, out_channels_list=[32, 64], kernel_sizes=3, conv_type="2d", use_batch_norm=False
    )
    assert len(conv_layers) == 2
    assert isinstance(batch_norms[0], nn.Identity)


def test_basecnn_create_doubling_conv_blocks():
    """Test creating doubling convolutional blocks."""
    model = BaseCNN()
    conv_layers, batch_norms, global_pools = model._create_doubling_conv_blocks(
        input_channels=3, base_channels=32, num_layers=3, first_kernel_size=3, subsequent_kernel_size=3
    )
    assert len(conv_layers) == 3
    assert len(global_pools) == 3
    assert conv_layers[0].out_channels == 32
    assert conv_layers[1].out_channels == 64
    assert conv_layers[2].out_channels == 128


def test_basecnn_create_progressive_conv_blocks():
    """Test creating progressive convolutional blocks with custom multipliers."""
    model = BaseCNN()
    # Test with [1, 2, 3] multipliers
    conv_layers, batch_norms = model._create_progressive_conv_blocks(
        in_channels=3, base_channels=32, num_layers=3, multipliers=[1, 2, 3], kernel_sizes=3, conv_type="2d"
    )
    assert len(conv_layers) == 3
    assert conv_layers[0].out_channels == 32  # 32 * 1
    assert conv_layers[1].out_channels == 64  # 32 * 2
    assert conv_layers[2].out_channels == 96  # 32 * 3

    # Test with [1, 2, 4, 8] multipliers for 1D convolutions
    conv_layers2, batch_norms2 = model._create_progressive_conv_blocks(
        in_channels=16, base_channels=64, num_layers=4, multipliers=[1, 2, 4, 8], kernel_sizes=3, conv_type="1d"
    )
    assert len(conv_layers2) == 4
    assert conv_layers2[0].out_channels == 64  # 64 * 1
    assert conv_layers2[1].out_channels == 128  # 64 * 2
    assert conv_layers2[2].out_channels == 256  # 64 * 4
    assert conv_layers2[3].out_channels == 512  # 64 * 8
    assert isinstance(conv_layers2[0], nn.Conv1d)

    # Test with varying list of kernel sizes
    conv_layers3, _ = model._create_progressive_conv_blocks(
        in_channels=8, base_channels=16, num_layers=3, multipliers=[1, 3, 5], kernel_sizes=[3, 5, 7], conv_type="2d"
    )
    assert conv_layers3[0].kernel_size == (3, 3)
    assert conv_layers3[1].kernel_size == (5, 5)
    assert conv_layers3[2].kernel_size == (7, 7)
    assert conv_layers3[0].out_channels == 16  # 16 * 1
    assert conv_layers3[1].out_channels == 48  # 16 * 3
    assert conv_layers3[2].out_channels == 80  # 16 * 5


def test_basecnn_initialize_weights_kaiming():
    """Test Kaiming weight initialization."""
    model = _create_simple_cnn(in_channels=3)
    # Check that weights have been initialized
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            assert m.weight is not None
            # Weights should not be all zeros
            assert not torch.all(m.weight == 0)


def test_basecnn_flatten_features():
    """Test feature flattening."""
    model = BaseCNN()
    x = torch.randn(8, 64, 7, 7)  # (batch, channels, height, width)
    x_flat = model._flatten_features(x)
    assert x_flat.shape == (8, 64 * 7 * 7)


def test_basecnn_apply_pooling_max_2d():
    """Test max pooling for 2D tensors."""
    model = BaseCNN()
    x = torch.randn(8, 32, 28, 28)
    x_pooled = model._apply_pooling(x, pool_type="max", pool_size=2)
    assert x_pooled.shape == (8, 32, 14, 14)


def test_basecnn_apply_pooling_avg_2d():
    """Test average pooling for 2D tensors."""
    model = BaseCNN()
    x = torch.randn(8, 32, 28, 28)
    x_pooled = model._apply_pooling(x, pool_type="avg", pool_size=2)
    assert x_pooled.shape == (8, 32, 14, 14)


def test_basecnn_apply_pooling_adaptive_max():
    """Test adaptive max pooling."""
    model = BaseCNN()
    x = torch.randn(8, 64, 14, 14)
    x_pooled = model._apply_pooling(x, pool_type="adaptive_max", adaptive_output_size=7)
    assert x_pooled.shape == (8, 64, 7, 7)


def test_basecnn_apply_pooling_1d():
    """Test pooling for 1D tensors."""
    model = BaseCNN()
    x = torch.randn(8, 32, 100)  # (batch, channels, length)
    x_pooled = model._apply_pooling(x, pool_type="max", pool_size=2)
    assert x_pooled.shape == (8, 32, 50)


def test_basecnn_create_embedding_layer():
    """Test creating embedding layer."""
    model = BaseCNN()
    embedding = model._create_embedding_layer(num_embeddings=1000, embedding_dim=128)
    assert isinstance(embedding, nn.Embedding)
    assert embedding.num_embeddings == 1000
    assert embedding.embedding_dim == 128


def test_basecnn_create_embedding_layer_with_padding():
    """Test creating embedding layer with padding index."""
    model = BaseCNN()
    embedding = model._create_embedding_layer(num_embeddings=1000, embedding_dim=128, padding_idx=0)
    assert embedding.padding_idx == 0


def test_basecnn_get_conv_output_size():
    """Test calculating convolutional output size."""
    model = BaseCNN()
    # Test with same padding
    output_size = model._get_conv_output_size(input_size=28, kernel_size=3, padding=1, stride=1)
    assert output_size == 28

    # Test with stride 2
    output_size = model._get_conv_output_size(input_size=28, kernel_size=3, padding=1, stride=2)
    assert output_size == 14

    # Test without padding
    output_size = model._get_conv_output_size(input_size=28, kernel_size=3, padding=0, stride=1)
    assert output_size == 26


def test_basecnn_simple_cnn_forward():
    """Test forward pass through a simple CNN using BaseCNN."""
    model = _create_simple_cnn(in_channels=3)
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    assert output.dim() == 2  # Should be flattened
    assert output.shape[0] == 4  # Batch size preserved


# =============================================================================
# CNNEncoder Tests
# =============================================================================


def test_cnn_encoder():
    """Test CNNEncoder initialization and forward pass."""
    num_embeddings, embedding_dim = 64, 128
    sequence_length = 85
    num_filters = 32
    filter_length = 8
    cnn_encoder = CNNEncoder(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sequence_length=sequence_length,
        num_kernels=num_filters,
        kernel_length=filter_length,
    ).eval()
    # assert cnn encoder shape
    assert cnn_encoder.embedding.weight.size() == (num_embeddings + 1, embedding_dim)
    assert cnn_encoder.conv1.__repr__() == "Conv1d(85, 32, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv2.__repr__() == "Conv1d(32, 64, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.conv3.__repr__() == "Conv1d(64, 96, kernel_size=(8,), stride=(1,))"
    assert cnn_encoder.global_max_pool.__repr__() == "AdaptiveMaxPool1d(output_size=1)"

    input_batch = torch.randint(high=num_embeddings, size=(8, sequence_length))
    output_encoding = cnn_encoder(input_batch)
    # assert output shape
    assert output_encoding.size() == (8, 96)


def test_cnn_encoder_repr():
    """Test CNNEncoder __repr__ method for debugging."""
    model = CNNEncoder(num_embeddings=63, embedding_dim=128, sequence_length=100, num_kernels=32, kernel_length=8)
    repr_str = repr(model)
    assert "CNNEncoder" in repr_str
    assert "embedding_dim=128" in repr_str
    assert "output_features=96" in repr_str  # num_kernels * 3


# =============================================================================
# ProteinCNN Tests
# =============================================================================


def test_protein_cnn_forward():
    """Test ProteinCNN forward pass with typical inputs."""
    embedding_dim = 128
    num_filters = [32, 64, 128]
    kernel_size = [3, 3, 3]
    sequence_length = 200
    batch_size = 64

    # Initialize ProteinCNN model
    model = ProteinCNN(embedding_dim, num_filters, kernel_size)

    # Create a mock protein input (protein sequence input)
    protein_input = torch.randint(0, 25, (batch_size, sequence_length))  # Random protein sequences

    # Forward pass through the model
    output = model(protein_input)

    # Check output types and shape
    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.shape[0] == batch_size, "Output batch size should match input batch size"


def test_protein_cnn_minimal_inputs():
    """Test ProteinCNN with minimal configuration."""
    # ProteinCNN
    model = ProteinCNN(1, [1, 1, 1], [1, 1, 1], padding=False)
    model.eval()
    inp = torch.randint(0, 1, (2, 1))
    out = model(inp)
    assert out.shape[0] == 2


def test_protein_cnn_repr():
    """Test ProteinCNN __repr__ method for debugging."""
    model = ProteinCNN(embedding_dim=64, num_filters=[32, 64, 96], kernel_size=[4, 6, 8])
    repr_str = repr(model)
    assert "ProteinCNN" in repr_str
    assert "embedding_dim=64" in repr_str
    assert "output_features=96" in repr_str  # Last filter size


# =============================================================================
# Additional Coverage Tests for Missing Branches
# =============================================================================


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


def test_cnn_encoder_output_size():
    """Test CNNEncoder output_size method."""
    model = CNNEncoder(num_embeddings=100, embedding_dim=128, sequence_length=1000, num_kernels=32, kernel_length=4)
    output_size = model.output_size()
    assert isinstance(output_size, int) or isinstance(output_size, tuple)
    # CNNEncoder returns (96,) as tuple
    if isinstance(output_size, tuple):
        assert len(output_size) == 1
        assert output_size[0] == 96  # num_kernels * 3
    else:
        assert output_size == 96


def test_cnn_encoder_with_different_kernels():
    """Test CNNEncoder with different kernel configurations."""
    model = CNNEncoder(num_embeddings=50, embedding_dim=64, sequence_length=500, num_kernels=16, kernel_length=8)
    x = torch.randint(0, 50, (2, 500))
    output = model(x)
    assert output.shape[0] == 2
    assert output.shape[1] == 48  # num_kernels * 3


def test_protein_cnn_output_size():
    """Test ProteinCNN output_size method."""
    model = ProteinCNN(embedding_dim=64, num_filters=[32, 64, 96], kernel_size=[4, 6, 8])
    output_size = model.output_size()
    assert output_size == 96  # Last filter size (returns int, not tuple)


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
# ContextCNNGeneric and CNNTransformer Tests
# =============================================================================


class TestContextCNNGeneric:
    """Test ContextCNNGeneric class functionality."""

    def test_context_cnn_generic_spatial_output(self):
        """Test ContextCNNGeneric with spatial output type."""
        # Create a CNN that outputs spatial features
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()  # Set to eval mode to avoid batch norm issues

        # Get actual output shape from a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        # Create a simple contextualizer
        num_channels = cnn_output_shape[1]
        contextualizer = nn.Sequential(
            nn.LayerNorm(num_channels),
        )

        # Create ContextCNNGeneric with spatial output
        model = ContextCNNGeneric(
            cnn=cnn, cnn_output_shape=cnn_output_shape, contextualizer=contextualizer, output_type="spatial"
        )
        model.eval()

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        # Output should be spatial (4D tensor)
        assert output.dim() == 4, f"Expected 4D output for spatial, got {output.dim()}D"
        assert output.shape[1] == num_channels, f"Expected {num_channels} channels, got {output.shape[1]}"

    def test_context_cnn_generic_sequence_output(self):
        """Test ContextCNNGeneric with sequence output type."""
        # Create a CNN that outputs spatial features
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        # Get actual output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]
        contextualizer = nn.Sequential(
            nn.LayerNorm(num_channels),
        )

        # Create ContextCNNGeneric with sequence output
        model = ContextCNNGeneric(
            cnn=cnn, cnn_output_shape=cnn_output_shape, contextualizer=contextualizer, output_type="sequence"
        )
        model.eval()

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        # Output should be sequence (3D tensor: batch, seq_len, channels)
        assert output.dim() == 3, f"Expected 3D output for sequence, got {output.dim()}D"
        assert output.shape[2] == num_channels, f"Expected {num_channels} channels, got {output.shape[2]}"

    def test_context_cnn_generic_invalid_output_type(self):
        """Test ContextCNNGeneric with invalid output type."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3]])
        cnn_output_shape = (1, 32, 30, 30)  # approximate shape
        contextualizer = nn.Sequential(nn.LayerNorm(32))

        # Should raise AssertionError for invalid output type
        with pytest.raises(AssertionError, match="parameter 'output_type' must be one of"):
            ContextCNNGeneric(
                cnn=cnn, cnn_output_shape=cnn_output_shape, contextualizer=contextualizer, output_type="invalid"
            )

    def test_context_cnn_generic_batch_processing(self):
        """Test ContextCNNGeneric with different batch sizes."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]
        contextualizer = nn.Sequential(nn.LayerNorm(num_channels))

        model = ContextCNNGeneric(
            cnn=cnn, cnn_output_shape=cnn_output_shape, contextualizer=contextualizer, output_type="spatial"
        )
        model.eval()

        # Test with batch size 2
        x2 = torch.randn(2, 3, 32, 32)
        output2 = model(x2)
        assert output2.shape[0] == 2

        # Test with batch size 4
        x4 = torch.randn(4, 3, 32, 32)
        output4 = model(x4)
        assert output4.shape[0] == 4


class TestCNNTransformer:
    """Test CNNTransformer class functionality."""

    def test_cnn_transformer_initialization(self):
        """Test CNNTransformer initialization and structure."""
        # Create a CNN with spatial output
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn_output_shape = (1, 64, 28, 28)  # approximate

        # Create CNNTransformer
        model = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            output_type="sequence",
        )

        # Check that model has the expected components
        assert hasattr(model, "cnn")
        assert hasattr(model, "contextualizer")
        assert model.output_type == "sequence"

    def test_cnn_transformer_forward_sequence(self):
        """Test CNNTransformer forward pass with sequence output."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]
        model = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            output_type="sequence",
        )
        model.eval()

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        # Output should be sequence (3D tensor: seq_len, batch, channels)
        # Note: spatial_to_seq outputs (seq_len, batch_size, channels) format
        assert output.dim() == 3, f"Expected 3D output for sequence, got {output.dim()}D"
        assert output.shape[1] == 2, f"Expected batch size 2, got {output.shape[1]}"
        assert output.shape[2] == num_channels, f"Expected {num_channels} channels, got {output.shape[2]}"

    def test_cnn_transformer_forward_spatial(self):
        """Test CNNTransformer forward pass with spatial output."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]
        model = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            output_type="spatial",
        )
        model.eval()

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        # Output should be spatial (4D)
        assert output.dim() == 4
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == num_channels  # channels

    def test_cnn_transformer_with_custom_positional_encoder(self):
        """Test CNNTransformer with custom positional encoder."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]
        # Use identity as custom positional encoder (skip positional encoding)
        custom_pos_encoder = nn.Identity()

        model = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            output_type="sequence",
            positional_encoder=custom_pos_encoder,
        )
        model.eval()

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = model(x)

        # Should still work with custom encoder
        assert output.dim() == 3
        assert output.shape[2] == num_channels

    def test_cnn_transformer_different_configurations(self):
        """Test CNNTransformer with different hyperparameter configurations."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3], [64, 3]])
        cnn.eval()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = cnn(dummy_input)
            cnn_output_shape = dummy_output.shape

        num_channels = cnn_output_shape[1]

        # Test with single layer
        model1 = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=1,
            num_heads=2,
            dim_feedforward=128,
            dropout=0.0,
            output_type="sequence",
        )
        model1.eval()

        x = torch.randn(2, 3, 32, 32)
        output1 = model1(x)
        assert output1.shape[2] == num_channels

        # Test with more layers and heads
        model2 = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=4,
            num_heads=8,
            dim_feedforward=512,
            dropout=0.2,
            output_type="spatial",
        )
        model2.eval()

        output2 = model2(x)
        assert output2.shape[1] == num_channels

    def test_cnn_transformer_xavier_initialization(self):
        """Test that CNNTransformer applies Xavier initialization to transformer."""
        cnn = SimpleCNNBuilder(num_channels=3, conv_layers_spec=[[32, 3]])
        cnn_output_shape = (1, 32, 30, 30)

        model = CNNTransformer(
            cnn=cnn,
            cnn_output_shape=cnn_output_shape,
            num_layers=2,
            num_heads=4,
            dim_feedforward=256,
            dropout=0.1,
            output_type="sequence",
        )

        # Check that transformer parameters exist and are initialized
        # (Xavier uniform initialization is applied in __init__)
        transformer_params = [p for p in model.contextualizer.parameters() if p.dim() > 1]
        assert len(transformer_params) > 0, "Should have multi-dimensional parameters in transformer"

        # Verify parameters are initialized (not all zeros)
        for param in transformer_params:
            assert not torch.all(param == 0), "Parameters should be initialized (not all zeros)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
