"""
Convolutional Neural Network (CNN) architectures for embedding and feature extraction.

This module provides a collection of CNN-based models for various tasks including:
- Drug-target interaction prediction (CNNEncoder)
- Protein sequence feature extraction (ProteinCNN)
- CNN-Transformer hybrid architectures (CNNTransformer, ContextCNNGeneric)

All CNN implementations inherit from BaseCNN (from kale.embed.base_cnn), which provides
reusable utilities for creating convolutional blocks, applying activations, weight
initialization, pooling operations, and embedding layer creation.

Classes:
    CNNEncoder: 1D CNN encoder for sequence data (DeepDTA architecture).
    ProteinCNN: 1D CNN for protein sequence feature extraction.
    ContextCNNGeneric: Template for CNN + sequence-to-sequence contextualizer.
    CNNTransformer: CNN backbone followed by Transformer-Encoder.

Note:
    Import BaseCNN directly from kale.embed.base_cnn for base utilities.

Example:
    >>> from kale.embed.base_cnn import BaseCNN
    >>> from kale.embed.cnn import CNNEncoder, ProteinCNN
    >>> # Create a custom CNN using BaseCNN utilities
    >>> class MyCNN(BaseCNN):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
    ...             in_channels=3, out_channels_size_list=[32, 64], kernel_sizes=3, conv_type='2d'
    ...         )
    >>> # Use existing implementations
    >>> encoder = CNNEncoder(num_embeddings=64, embedding_dim=128, sequence_length=85,
    ...                       num_kernels=32, kernel_length=8)
"""
from typing import Any, Tuple, Union

import torch
import torch.nn as nn

from kale.embed.attention import PositionalEncoding
from kale.embed.base_cnn import BaseCNN
from kale.prepdata.tensor_reshape import seq_to_spatial, spatial_to_seq


class CNNEncoder(BaseCNN):
    r"""
    The DeepDTA's CNN encoder module, which comprises three 1D-convolutional layers and one max-pooling layer.
    The module is applied to encoding drug/target sequence information, and the input should be transformed information
    with integer/label encoding. The original paper is `"DeepDTA: deep drug–target binding affinity prediction"
    <https://academic.oup.com/bioinformatics/article/34/17/i821/5093245>`_.

    This class now inherits from BaseCNN to leverage shared CNN utilities.

    Args:
        num_embeddings (int): Number of embedding labels/categories, depends on the types of encoding sequence.
        embedding_dim (int): Dimension of embedding labels/categories.
        sequence_length (int): Max length of the input sequence.
        num_kernels (int): Number of kernels (filters).
        kernel_length (int): Length of kernel (filter).
    """

    def __init__(self, num_embeddings, embedding_dim, sequence_length, num_kernels, kernel_length):
        super(CNNEncoder, self).__init__()
        self.embedding = self._create_embedding_layer(num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim)

        conv_layers, _ = self._create_progressive_conv_blocks(
            in_channels=sequence_length,
            base_channels=num_kernels,
            num_layers=3,
            multipliers=[1, 2, 3],  # DeepDTA pattern: 1x, 2x, 3x base channels
            kernel_sizes=kernel_length,
            conv_type="1d",
            conv_padding=0,  # No padding for DeepDTA architecture
            use_batch_norm=False,
            bias=True,
        )

        self.conv_layers = nn.ModuleList(conv_layers)

        # Maintain backward compatibility: expose individual layer attributes
        self.conv1 = self.conv_layers[0]
        self.conv2 = self.conv_layers[1]
        self.conv3 = self.conv_layers[2]
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self._out_features = num_kernels * 3

    def forward(self, x):
        """
        Forward pass through the CNNEncoder.

        Args:
            x (torch.Tensor): Input tensor containing embedded sequence data of shape
                (batch_size, sequence_length).

        Returns:
            torch.Tensor: Encoded feature vector of shape (batch_size, num_kernels * 3).
        """
        x = self.embedding(x)
        for conv in self.conv_layers:
            x = self._apply_activation(conv(x), "relu")
        x = self.global_max_pool(x)
        x = x.squeeze(2)
        return x

    def output_size(self) -> int:
        """
        Return the output feature dimension of the encoder.

        Returns:
            int: Number of output features (num_kernels * 3).
        """
        return self._out_features

    def __repr__(self) -> str:
        """Return a string representation of the CNNEncoder."""
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding.embedding_dim}, "
            f"num_embeddings={self.embedding.num_embeddings}, "
            f"output_features={self._out_features})"
        )


class ProteinCNN(BaseCNN):
    """
    A protein feature extractor using Convolutional Neural Networks (CNNs).

    This class extracts features from protein sequences using a series of 1D convolutional layers.
    The input protein sequence is first embedded and then passed through multiple convolutional
    and batch normalization layers to produce a fixed-size feature vector.

    This class now inherits from BaseCNN to leverage shared CNN utilities.

    Args:
        embedding_dim (int): Dimensionality of the embedding space for protein sequences.
        num_filters (list of int): A list specifying the number of filters for each convolutional layer.
        kernel_size (list of int): A list specifying the kernel size for each convolutional layer.
        padding (bool, optional): Whether to apply padding to the embedding layer. Defaults to True.
            Note: This controls the `padding_idx` parameter of the embedding layer, not the convolutional
            layer padding (which is controlled by the `conv_padding` argument in BaseCNN utilities).
    """

    def __init__(self, embedding_dim, num_filters, kernel_size, padding: bool = True):
        super(ProteinCNN, self).__init__()
        padding_idx = 0 if padding else None
        self.embedding = self._create_embedding_layer(
            num_embeddings=26, embedding_dim=embedding_dim, padding_idx=padding_idx
        )

        conv_layers, batch_norms = self._create_sequential_conv_blocks(
            in_channels=embedding_dim,
            out_channels_size_list=num_filters,
            kernel_sizes=kernel_size,
            conv_type="1d",
            conv_padding=0,
            use_batch_norm=True,
            bias=True,
        )

        self.conv_layers = nn.ModuleList(conv_layers)
        self.batch_norms = nn.ModuleList(batch_norms)

        # Maintain backward compatibility: expose individual layer attributes
        self.conv1 = self.conv_layers[0]
        self.bn1 = self.batch_norms[0]
        self.conv2 = self.conv_layers[1]
        self.bn2 = self.batch_norms[1]
        self.conv3 = self.conv_layers[2]
        self.bn3 = self.batch_norms[2]
        self._out_features = num_filters[-1]

    def forward(self, v):
        """
        Forward pass through the ProteinCNN.

        Args:
            v (torch.Tensor): Input tensor containing protein sequence indices of shape
                (batch_size, sequence_length).

        Returns:
            torch.Tensor: Extracted protein features of shape
                (batch_size, sequence_length, num_filters[-1]).
        """
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            v = bn(self._apply_activation(conv(v), "relu"))
        v = v.view(v.size(0), v.size(2), -1)
        return v

    def output_size(self) -> int:
        """
        Return the output feature dimension of the protein CNN.

        Returns:
            int: Number of output features (last filter size).
        """
        return self._out_features

    def __repr__(self) -> str:
        """Return a string representation of the ProteinCNN."""
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding.embedding_dim}, "
            f"num_embeddings={self.embedding.num_embeddings}, "
            f"output_features={self._out_features})"
        )


class ContextCNNGeneric(nn.Module):
    """
    A template to construct a feature extractor consisting of a CNN followed by a sequence-to-sequence contextualizer
    like a Transformer-Encoder. Before inputting the CNN output tensor to the contextualizer, the tensor's spatial
    dimensions are unrolled into a sequence.

    Args:
        cnn (nn.Module): Any convolutional neural network that takes in batches of images of
                        shape (batch_size, channels, height, width) and outputs tensor representations of
                        shape (batch_size, out_channels, out_height, out_width).
        cnn_output_shape (tuple): A tuple of shape (batch_size, num_channels, height, width) describing
                        the output shape of the given CNN (required).
        contextualizer (nn.Module, optional): A sequence-to-sequence model that takes inputs of
                        shape (num_timesteps, batch_size, num_features) and uses attention to contextualize
                        the sequence and returns a sequence of the exact same shape.
                        This will mainly be a Transformer-Encoder (required).
        output_type (string): One of 'sequence' or 'spatial'. If spatial, then the final output of the model,
                        which is a sequence, will be reshaped to resemble the image-batch shape of the output of the CNN.
                        If sequence then the output sequence is returned as is (required).

    Examples:
        >>> cnn = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3),
        >>>                     nn.Conv2d(32, 64, kernel_size=3),
        >>>                     nn.MaxPool2d(2))
        >>> cnn_output_shape = (-1, 64, 8, 8)
        >>> contextualizer = nn.TransformerEncoderLayer(...)
        >>> output_type = 'spatial'
        >>>
        >>> attention_cnn = ContextCNNGeneric(cnn, cnn_output_shape, contextualizer, output_type)
        >>> output = attention_cnn(torch.randn((32,3,16,16)))
        >>>
        >>> output.size() == cnn_output_shape # True
    """

    def __init__(
        self,
        cnn: nn.Module,
        cnn_output_shape: Tuple[int, int, int, int],
        contextualizer: Union[nn.Module, Any],
        output_type: str,
    ):
        super(ContextCNNGeneric, self).__init__()
        assert output_type in ["spatial", "sequence"], (
            "parameter 'output_type' must be one of ('spatial', 'sequence')" + f" but is {output_type}"
        )

        self.cnn = cnn
        self.cnn_output_shape = cnn_output_shape
        self.contextualizer = contextualizer
        self.output_type = output_type

    def forward(self, x: torch.Tensor):
        """
        Pass the input through the cnn and then the contextualizer.

        Args:
            x: input image batch exactly as for CNNs (required).
        """
        cnn_rep = self.cnn(x)
        seq_rep = spatial_to_seq(cnn_rep)
        seq_rep = self.contextualizer(seq_rep)

        output = seq_rep
        if self.output_type == "spatial":
            desired_height = self.cnn_output_shape[2]
            desired_width = self.cnn_output_shape[3]
            output = seq_to_spatial(output, desired_height, desired_width)

        return output


class CNNTransformer(ContextCNNGeneric):
    """
    A feature extractor consisting of a given CNN backbone followed by a standard Transformer-Encoder.
    See documentation of "ContextCNNGeneric" for more information.

    Args:
        cnn (nn.Module): Any convolutional neural network that takes in batches of images of
                        shape (batch_size, channels, height, width) and outputs tensor representations of
                        shape (batch_size, out_channels, out_height, out_width) (required).
        cnn_output_shape (tuple): A tuple of shape (batch_size, num_channels, height, width) describing the
                        output shape of the given CNN (required).
        num_layers (int): Number of attention layers in the Transformer-Encoder (required).
        num_heads (int): Number of attention heads in each transformer block (required).
        dim_feedforward (int): Number of neurons in the intermediate dense layer of each transformer feedforward block (required).
        dropout (float): Dropout rate of the transformer layers (required).
        output_type (string): One of 'sequence' or 'spatial'. If Spatial then the final output of the model,
                        which is the sequence output of the Transformer-Encoder, will be reshaped to resemble the
                        image-batch shape of the output of the CNN (required).
        positional_encoder (nn.Module): None or a nn.Module that expects inputs of
                        shape (sequence_length, batch_size, embedding_dim) and returns the same input after adding
                        some positional information to the embeddings. If `None`, then the default and fixed sin-cos
                        positional encodings of base transformers are applied (optional).

    Examples:
        See pykale/examples/cifar_cnntransformer/model.py
    """

    def __init__(
        self,
        cnn: nn.Module,
        cnn_output_shape: Tuple[int, int, int, int],
        num_layers: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        output_type: str,
        positional_encoder: nn.Module = None,
    ):
        num_channels = cnn_output_shape[1]
        height = cnn_output_shape[2]
        width = cnn_output_shape[3]

        encoder_layer = nn.TransformerEncoderLayer(num_channels, num_heads, dim_feedforward, dropout, batch_first=True)
        encoder_normalizer = nn.LayerNorm(num_channels)
        encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_normalizer)

        if positional_encoder is None:
            positional_encoder = PositionalEncoding(d_model=num_channels, max_len=height * width)
        else:
            # allows for passing the identity block to skip this step
            # or chosing a different encoding
            positional_encoder = positional_encoder

        transformer_input_dropout = nn.Dropout(dropout)
        contextualizer = nn.Sequential(positional_encoder, transformer_input_dropout, encoder)

        super(CNNTransformer, self).__init__(cnn, cnn_output_shape, contextualizer, output_type)

        # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
