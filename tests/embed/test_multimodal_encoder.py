from unittest.mock import patch

import torch

from kale.embed.multimodal_encoder import SignalImageVAE


def test_bimodal_vae_full_coverage():
    # Small test config for speed and memory
    batch_size = 2
    image_channels = 1
    image_size = 224  # So output after 3x stride=2 is 28x28
    signal_length = 64
    latent_dim = 4

    image = torch.randn(batch_size, image_channels, image_size, image_size)
    signal = torch.randn(batch_size, 1, signal_length)

    model = SignalImageVAE(image_input_channels=image_channels, signal_input_dim=signal_length, latent_dim=latent_dim)

    # --- Test prior_expert (cpu branch) ---
    mean, log_var = model.prior_expert((1, batch_size, latent_dim), use_cuda=False)
    assert mean.shape == (1, batch_size, latent_dim)
    assert torch.allclose(log_var, torch.zeros_like(log_var))

    # --- Test prior_expert (mocked cuda branch, even on CPU) ---
    with patch.object(torch.Tensor, "cuda", lambda x: x):
        mu_cuda, log_var_cuda = model.prior_expert((1, batch_size, latent_dim), use_cuda=True)
        assert mu_cuda.shape == (1, batch_size, latent_dim)
        assert log_var_cuda.shape == (1, batch_size, latent_dim)
        # These are not .is_cuda (since we're faking), but they exist

    # --- Test reparametrize, both training and eval mode ---
    dummy_mu = torch.zeros(batch_size, latent_dim)
    dummy_log_var = torch.zeros(batch_size, latent_dim)
    model.train()
    z_train = model.reparametrize(dummy_mu, dummy_log_var)
    assert z_train.shape == (batch_size, latent_dim)
    model.eval()
    z_eval = model.reparametrize(dummy_mu, dummy_log_var)
    assert torch.allclose(z_eval, dummy_mu)

    # --- Test forward (with both modalities) ---
    model.train()
    img_recon, sig_recon, mean, log_var = model(image=image, signal=signal)
    assert img_recon.shape[0] == batch_size
    assert sig_recon.shape[0] == batch_size
    assert mean.shape == (batch_size, latent_dim)
    assert log_var.shape == (batch_size, latent_dim)

    # --- Test infer with only one modality at a time ---
    mu_img, log_var_img = model.infer(image=image)
    assert mu_img.shape == (batch_size, latent_dim)
    mu_sig, log_var_sig = model.infer(signal=signal)
    assert mu_sig.shape == (batch_size, latent_dim)
