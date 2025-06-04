# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================

"""
multimodal_encoder.py

Contains multimodal encoder models, such as BimodalVAE, for joint representation learning.
"""

import torch
import torch.nn as nn
from kale.embed.image_cnn import ImageVAEEncoder
from kale.embed.signal_cnn import SignalVAEEncoder
from kale.predict.decode import ImageVAEDecoder
from kale.predict.decode import SignalVAEDecoder
from kale.embed.feature_fusion import ProductOfExperts

class BimodalVAE(nn.Module):
    """
    BimodalVAE performs joint variational autoencoding for two input modalities (such as images and 1D signals).

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
        mu (Tensor): Mean vector of the fused latent distribution (batch_size, latent_dim)
        logvar (Tensor): Log-variance vector of the fused latent distribution (batch_size, latent_dim)

    Example:
        model = MultimodalVAE(image_input_channels=1, signal_input_dim=60000, latent_dim=128)
        image_recon, signal_recon, mu, logvar = model(image=image_data, signal=signal_data)
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
            mu (Tensor): Zero-mean tensor.
            logvar (Tensor): Zero log-variance tensor.
        """
        mu = torch.zeros(size)
        logvar = torch.zeros(size)
        if use_cuda:
            mu = mu.cuda()
            logvar = logvar.cuda()
        return mu, logvar

    def reparametrize(self, mu, logvar):
        """
        Applies the reparameterization trick to sample from a Gaussian distribution parameterized by mu and logvar.

        This allows backpropagation through stochastic nodes by expressing the random variable as a deterministic
        function of mu, logvar, and noise. During training, returns a random sample from N(mu, sigma^2).
        During evaluation, returns mu (the mean).

        Args:
            mu (Tensor): Mean of the Gaussian, shape (batch_size, latent_dim).
            logvar (Tensor): Log-variance of the Gaussian, shape (batch_size, latent_dim).

        Returns:
            z (Tensor): Sampled latent vector, shape (batch_size, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, image=None, signal=None):
        mu, logvar = self.infer(image, signal)
        z = self.reparametrize(mu, logvar)
        image_recon = self.image_decoder(z)
        signal_recon = self.signal_decoder(z)
        return image_recon, signal_recon, mu, logvar

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
            mu (Tensor): Mean vector of the fused latent distribution, shape (batch_size, latent_dim).
            logvar (Tensor): Log-variance vector of the fused latent distribution, shape (batch_size, latent_dim).
        """
        batch_size = image.size(0) if image is not None else signal.size(0)
        use_cuda = next(self.parameters()).is_cuda
        mu, logvar = self.prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)
        if signal is not None:
            signal_mu, signal_logvar = self.signal_encoder(signal)
            mu = torch.cat((mu, signal_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, signal_logvar.unsqueeze(0)), dim=0)
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar

