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
    mu, logvar = encoder(x)

    # Check output shapes
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert isinstance(mu, torch.Tensor)
    assert isinstance(logvar, torch.Tensor)
