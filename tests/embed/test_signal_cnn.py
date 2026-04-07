"""
Unit tests for signal CNN module including SignalVAEEncoder.
"""
import torch

from kale.embed.signal_cnn import SignalVAEEncoder


def test_signal_vae_encoder_forward():
    """Test SignalVAEEncoder forward pass with default parameters."""
    # Test configuration
    batch_size = 3
    input_dim = 60000
    latent_dim = 16

    # Create dummy 1D signal input (batch_size, channels, input_dim)
    x = torch.randn(batch_size, 1, input_dim)

    # Initialize encoder
    encoder = SignalVAEEncoder(input_dim=input_dim, latent_dim=latent_dim)

    # Forward pass
    mean, log_var = encoder(x)

    # Check output shapes
    assert mean.shape == (batch_size, latent_dim)
    assert log_var.shape == (batch_size, latent_dim)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(log_var, torch.Tensor)


def test_signal_vae_encoder_custom_dimensions():
    """Test SignalVAEEncoder with custom input and latent dimensions."""
    batch_size = 5
    input_dim = 32000
    latent_dim = 128

    x = torch.randn(batch_size, 1, input_dim)
    encoder = SignalVAEEncoder(input_dim=input_dim, latent_dim=latent_dim)

    mean, log_var = encoder(x)

    assert mean.shape == (batch_size, latent_dim)
    assert log_var.shape == (batch_size, latent_dim)


def test_signal_vae_encoder_backward_compatibility():
    """Test that individual conv layer attributes are accessible for backward compatibility."""
    encoder = SignalVAEEncoder(input_dim=60000, latent_dim=256)

    # Check that individual layers are accessible
    assert hasattr(encoder, "conv1")
    assert hasattr(encoder, "conv2")
    assert hasattr(encoder, "conv3")

    # Verify layer types and configurations
    assert isinstance(encoder.conv1, torch.nn.Conv1d)
    assert encoder.conv1.in_channels == 1
    assert encoder.conv1.out_channels == 16
    assert encoder.conv1.kernel_size == (3,)
    assert encoder.conv1.stride == (2,)

    assert encoder.conv2.in_channels == 16
    assert encoder.conv2.out_channels == 32

    assert encoder.conv3.in_channels == 32
    assert encoder.conv3.out_channels == 64


def test_signal_vae_encoder_output_size():
    """Test SignalVAEEncoder output_size() method."""
    input_dim = 60000
    latent_dim = 128
    encoder = SignalVAEEncoder(input_dim=input_dim, latent_dim=latent_dim)

    output_size = encoder.output_size()
    expected_size = 64 * (input_dim // 8)

    assert output_size == expected_size
    assert isinstance(output_size, int)


def test_signal_vae_encoder_repr():
    """Test SignalVAEEncoder __repr__ method for debugging."""
    input_dim = 60000
    latent_dim = 256
    encoder = SignalVAEEncoder(input_dim=input_dim, latent_dim=latent_dim)

    repr_str = repr(encoder)

    assert "SignalVAEEncoder" in repr_str
    assert f"latent_dim={latent_dim}" in repr_str
    assert "input_features=" in repr_str


def test_signal_vae_encoder_deterministic():
    """Test that SignalVAEEncoder produces consistent outputs for same inputs."""
    input_dim = 60000
    latent_dim = 64
    batch_size = 2

    x = torch.randn(batch_size, 1, input_dim)
    encoder = SignalVAEEncoder(input_dim=input_dim, latent_dim=latent_dim)
    encoder.eval()

    # Run forward pass twice with same input
    mean1, log_var1 = encoder(x)
    mean2, log_var2 = encoder(x)

    # Results should be identical
    assert torch.allclose(mean1, mean2)
    assert torch.allclose(log_var1, log_var2)
