# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

"""
multimodal_encoder.py

Contains multimodal encoder models, such as BimodalVAE, for joint representation learning.
"""

import torch
import torch.nn as nn

from kale.embed.feature_fusion import ProductOfExperts
from kale.embed.image_cnn import ImageVAEEncoder
from kale.embed.signal_cnn import SignalVAEEncoder
from kale.predict.decode import ImageVAEDecoder, SignalVAEDecoder


class SignalImageVAE(nn.Module):
    """
    SignalImageVAE performs joint variational autoencoding for two input modality image and signal).

    The architecture comprises separate encoders and decoders for each modality (e.g., an image encoder/decoder and a signal encoder/decoder),
    and uses a Product of Experts (PoE) mechanism to fuse the latent distributions from each modality as well as a universal prior expert.
    During inference, the model supports missing modalities by marginalizing over available experts.

    This flexible design allows for robust unsupervised representation learning, missing data imputation, and multimodal generation tasks
    in diverse domains including medical imaging, sensor fusion, and more.

    Args:
        image_input_channels (int, optional): Number of input channels in the image modality (e.g., 1 for grayscale). Default is 1.
        signal_input_dim (int, optional): Length of the 1D input signal. Default is 60000.
        latent_dim (int, optional): Dimensionality of the shared latent space. Default is 256.

    Forward Inputs:
        image (Tensor, optional): Image tensor of shape (batch_size, image_input_channels, H, W)
        signal (Tensor, optional): 1D signal tensor of shape (batch_size, 1, signal_input_dim)

    Forward Outputs:
        image_recon (Tensor): Reconstructed image tensor (batch_size, image_input_channels, H, W)
        signal_recon (Tensor): Reconstructed signal tensor (batch_size, 1, signal_input_dim)
        mean (Tensor): Mean vector of the fused latent distribution (batch_size, latent_dim)
        log_var (Tensor): Log-variance vector of the fused latent distribution (batch_size, latent_dim)

    Example:
        model = SignalImageVAE(image_input_channels=1, signal_input_dim=60000, latent_dim=128)
        image_recon, signal_recon, mean, log_var = model(image=image_data, signal=signal_data)
    """

    def __init__(self, image_input_channels=1, signal_input_dim=60000, latent_dim=256):
        super().__init__()
        self.image_encoder = ImageVAEEncoder(image_input_channels, latent_dim)
        self.signal_encoder = SignalVAEEncoder(signal_input_dim, latent_dim)
        self.image_decoder = ImageVAEDecoder(latent_dim, image_input_channels)
        self.signal_decoder = SignalVAEDecoder(latent_dim, signal_input_dim)
        self.experts = ProductOfExperts()
        self.n_latents = latent_dim

    @staticmethod
    def prior_expert(size, use_cuda=False):
        """
        Creates a universal prior expert as a spherical Gaussian N(0, 1) with specified size.

        Args:
            size (tuple): Desired shape, typically (1, batch_size, latent_dim).
            use_cuda (bool): Whether to move tensors to CUDA.

        Returns:
            mean (Tensor): Zero-mean tensor.
            log_var (Tensor): Zero log-variance tensor.
        """
        mean = torch.zeros(size)
        log_var = torch.zeros(size)
        if use_cuda:
            mean = mean.cuda()
            log_var = log_var.cuda()
        return mean, log_var

    def reparametrize(self, mean, log_var):
        """
        Applies the reparameterization trick to sample from a Gaussian distribution parameterized by mean and log_var.

        This allows backpropagation through stochastic nodes by expressing the random variable as a deterministic
        function of mean, log_var, and noise. During training, returns a random sample from N(mean, sigma^2).
        During evaluation, returns mean (the mean).

        Args:
            mean (Tensor): Mean of the Gaussian, shape (batch_size, latent_dim).
            log_var (Tensor): Log-variance of the Gaussian, shape (batch_size, latent_dim).

        Returns:
            z (Tensor): Sampled latent vector, shape (batch_size, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean

    def forward(self, image=None, signal=None):
        mean, log_var = self.infer(image, signal)
        z = self.reparametrize(mean, log_var)
        image_recon = self.image_decoder(z)
        signal_recon = self.signal_decoder(z)
        return image_recon, signal_recon, mean, log_var

    def infer(self, image=None, signal=None):
        """
        Computes the parameters of the fused latent distribution using the available modalities and the universal prior expert.

        Encodes the input image and/or signal into their respective latent Gaussians, concatenates them (along with the prior expert),
        and applies the Product of Experts to produce a single joint posterior for the latent variables.
        This function supports missing modalities by only using the provided inputs.

        Args:
            image (Tensor, optional): Image tensor, shape (batch_size, channels, H, W).
            signal (Tensor, optional): 1D signal tensor, shape (batch_size, 1, signal_length).

        Returns:
            mean (Tensor): Mean vector of the fused latent distribution, shape (batch_size, latent_dim).
            log_var (Tensor): Log-variance vector of the fused latent distribution, shape (batch_size, latent_dim).
        """
        batch_size = image.size(0) if image is not None else signal.size(0)
        use_cuda = next(self.parameters()).is_cuda
        mean, log_var = self.prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
        if image is not None:
            img_mu, img_log_var = self.image_encoder(image)
            mean = torch.cat((mean, img_mu.unsqueeze(0)), dim=0)
            log_var = torch.cat((log_var, img_log_var.unsqueeze(0)), dim=0)
        if signal is not None:
            signal_mu, signal_log_var = self.signal_encoder(signal)
            mean = torch.cat((mean, signal_mu.unsqueeze(0)), dim=0)
            log_var = torch.cat((log_var, signal_log_var.unsqueeze(0)), dim=0)
        mean, log_var = self.experts(mean, log_var)
        return mean, log_var
