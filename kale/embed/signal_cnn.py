# =============================================================================
# Author: Mohammod Suvon, m.suvon@sheffield.ac.uk
# =============================================================================
"""
This module provides CNN-based encoders for transforming 1D signals into latent representations, primarily for use in variational autoencoders (VAE) and related deep learning applications.
"""

import torch.nn as nn

class SignalVAEEncoder(nn.Module):
    """
    SignalVAEEncoder encodes 1D signals into a latent representation suitable for variational autoencoders (VAE).

    This encoder uses a series of 1D convolutional layers to extract hierarchical temporal features from generic 1D signals,
    followed by fully connected layers that output the mean and log-variance vectors for the latent Gaussian distribution.
    This structure is commonly used for unsupervised or multimodal learning on time-series or sequential data.

    Args:
        input_dim (int, optional): Length of the input 1D signal (number of time points). Default is 60000.
        latent_dim (int, optional): Dimensionality of the latent space representation. Default is 256.

    Forward Input:
        x (Tensor): Input signal tensor of shape (batch_size, 1, input_dim).

    Forward Output:
        mu (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        logvar (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        encoder = SignalVAEEncoder(input_dim=60000, latent_dim=128)
        mu, logvar = encoder(signals)
    """
    def __init__(self, input_dim=60000, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * (input_dim // 8), latent_dim)
        self.fc_logvar = nn.Linear(64 * (input_dim // 8), latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
