# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
This module provides CNN-based encoders for transforming 1D signals into latent representations, primarily for use in variational autoencoders (VAE) and related deep learning applications.
"""

import torch.nn as nn

from kale.embed.base_cnn import BaseCNN


class SignalVAEEncoder(BaseCNN):
    """
    SignalVAEEncoder encodes 1D signals into a latent representation suitable for variational autoencoders (VAE).

    This encoder uses a series of 1D convolutional layers to extract hierarchical temporal features from generic 1D signals,
    followed by fully connected layers that output the mean and log-variance vectors for the latent Gaussian distribution.
    This structure is commonly used for unsupervised or multimodal learning on time-series or sequential data.

    This class now inherits from BaseCNN to leverage shared CNN utilities for improved code reusability and FAIR compliance.

    Args:
        input_dim (int, optional): Length of the input 1D signal (number of time points). Default is 60000.
        latent_dim (int, optional): Dimensionality of the latent space representation. Default is 256.

    Forward Input:
        x (Tensor): Input signal tensor of shape (batch_size, 1, input_dim).

    Forward Output:
        mean (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        log_var (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        >>> encoder = SignalVAEEncoder(input_dim=60000, latent_dim=128)
        >>> mean, log_var = encoder(signals)
    """

    def __init__(self, input_dim=60000, latent_dim=256):
        super().__init__()

        self.conv_layers, _ = self._create_sequential_conv_blocks(
            in_channels=1,
            out_channels_size_list=[16, 32, 64],
            kernel_sizes=3,
            conv_type="1d",
            strides=2,
            conv_padding=1,
            use_batch_norm=False,
            bias=True,
        )

        self.conv1 = self.conv_layers[0]
        self.conv2 = self.conv_layers[1]
        self.conv3 = self.conv_layers[2]

        self.fc_mu = nn.Linear(64 * (input_dim // 8), latent_dim)
        self.fc_log_var = nn.Linear(64 * (input_dim // 8), latent_dim)

    def forward(self, x):
        """
        Forward pass through the SignalVAEEncoder.

        Args:
            x (torch.Tensor): Input 1D signal tensor of shape (batch_size, 1, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mean (torch.Tensor): Mean vector of the latent Gaussian distribution,
                    shape (batch_size, latent_dim).
                - log_var (torch.Tensor): Log-variance vector of the latent Gaussian,
                    shape (batch_size, latent_dim).
        """
        # Apply convolutions with ReLU activation (iterate for maintainability)
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = self._apply_activation(conv(x), "relu")

        x = self._flatten_features(x)

        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

    def output_size(self) -> int:
        """
        Return the output feature dimension before the latent projection.

        Returns:
            int: Number of flattened features after convolutions (64 * flattened_dim).
        """
        return self.fc_mu.in_features

    def __repr__(self) -> str:
        """Return a string representation of the SignalVAEEncoder."""
        return (
            f"{self.__class__.__name__}("
            f"latent_dim={self.fc_mu.out_features}, "
            f"input_features={self.fc_mu.in_features})"
        )
