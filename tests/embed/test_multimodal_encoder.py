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

    # --- Test prior_expert (cuda branch resolves to the cuda device, even on CPU-only hosts) ---
    prior_stub = torch.zeros(1, batch_size, latent_dim)  # built before patching torch.zeros
    with patch("kale.embed.multimodal_encoder.torch.zeros", return_value=prior_stub) as mock_zeros:
        mu_cuda, log_var_cuda = model.prior_expert((1, batch_size, latent_dim), use_cuda=True)
        assert mu_cuda.shape == (1, batch_size, latent_dim)
        assert log_var_cuda.shape == (1, batch_size, latent_dim)
        # Both the mean and log-variance tensors must be created on the cuda device, so check every
        # call rather than only the last one - otherwise a regression that placed just one of them
        # correctly would slip through.
        assert mock_zeros.call_count == 2
        assert all(call.kwargs["device"] == "cuda" for call in mock_zeros.call_args_list)

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


class TestPriorExpertDevice:
    """Device placement of the prior expert (issue #542)."""

    def test_explicit_device_is_honoured(self):
        """An explicit device is applied regardless of the use_cuda flag."""
        mean, log_var = SignalImageVAE.prior_expert((1, 2, 4), device="cpu")

        assert mean.device.type == "cpu"
        assert log_var.device.type == "cpu"

    def test_use_cuda_false_stays_on_cpu(self):
        """Backward compatibility: the use_cuda flag still selects CPU when False."""
        mean, log_var = SignalImageVAE.prior_expert((1, 2, 4), use_cuda=False)

        assert mean.device.type == "cpu"
        assert log_var.device.type == "cpu"

    def test_prior_matches_model_device(self):
        """infer() places the prior on the same device as the model parameters.

        Previously the prior was built from a boolean flag and always landed on the default CUDA
        device, so a model on a non-default device received a prior on the wrong one.
        """
        model = SignalImageVAE(image_input_channels=1, signal_input_dim=64, latent_dim=4)
        model.eval()
        expected = next(model.parameters()).device

        image = torch.rand(2, 1, 224, 224)
        signal = torch.rand(2, 1, 64)
        mean, log_var = model.infer(image=image, signal=signal)

        assert mean.device == expected
        assert log_var.device == expected

    def test_infer_forwards_model_device_to_prior_expert(self):
        """infer() forwards the model's actual device, not a boolean CUDA flag.

        The flag collapsed every CUDA device to the default one, so a model on a non-default
        device (e.g. cuda:1) or on MPS received its prior on the wrong device. Asserting the
        forwarded argument is what makes that regression detectable on a CPU-only host.
        """
        model = SignalImageVAE(image_input_channels=1, signal_input_dim=64, latent_dim=4)
        model.eval()
        expected = next(model.parameters()).device
        prior = (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))

        with patch.object(SignalImageVAE, "prior_expert", return_value=prior) as spy:
            model.infer(image=torch.rand(2, 1, 224, 224), signal=torch.rand(2, 1, 64))

        assert spy.call_args.kwargs["device"] == expected
