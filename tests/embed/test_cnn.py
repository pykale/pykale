"""
Unit tests for concrete CNN architectures: CNNEncoder, ProteinCNN,
ContextCNNGeneric, and CNNTransformer.
"""
import pytest
import torch
import torch.nn as nn

from kale.embed.cnn import CNNEncoder, CNNTransformer, ContextCNNGeneric, ProteinCNN
from kale.embed.image_cnn import SimpleCNNBuilder

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
