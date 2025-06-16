import torch

from kale.embed.signal_cnn import SignalVAEEncoder


def test_signal_vae_encoder_forward():
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
