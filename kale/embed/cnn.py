from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from kale.embed.attention import PositionalEncoding
from kale.prepdata.tensor_reshape import seq_to_spatial, spatial_to_seq


class BaseCNN(nn.Module):
    """
    Base class for CNN architectures providing common functionality and utilities.

    This class provides shared methods for creating convolutional blocks, applying
    activations, initializing weights, and performing common tensor operations.
    All CNN models should inherit from this base class to promote code reusability
    and maintain consistency across different architectures.

    The base class is designed to be flexible and accommodate both 1D and 2D
    convolutional networks, different activation functions, and various output formats.

    Example:
        >>> class MyCNN(BaseCNN):
        >>>     def __init__(self, input_channels, output_channels):
        >>>         super().__init__()
        >>>         self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
        >>>             in_channels=input_channels,
        >>>             out_channels_list=[32, 64],
        >>>             kernel_sizes=[3, 3],
        >>>             conv_type='2d'
        >>>         )
        >>>
        >>>     def forward(self, x):
        >>>         for conv, bn in zip(self.conv_layers, self.batch_norms):
        >>>             x = self._apply_activation(bn(conv(x)), "relu")
        >>>         return self._flatten_features(x)

    """

    def __init__(self):
        """Initialize the BaseCNN module."""
        super(BaseCNN, self).__init__()

    def _apply_activation(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        """
        Apply the specified activation function to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            activation (str): Activation function to apply. Supported values:
                'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'.

        Returns:
            torch.Tensor: Activated tensor.

        Raises:
            ValueError: If the activation function is not supported.

        Examples:
            >>> x = torch.randn(2, 3, 4, 4)
            >>> activated = self._apply_activation(x, "relu")
        """
        activation = activation.lower()
        if activation == "relu":
            return F.relu(x)
        elif activation == "tanh":
            return torch.tanh(x)
        elif activation == "sigmoid":
            return torch.sigmoid(x)
        elif activation == "leaky_relu":
            return F.leaky_relu(x)
        elif activation == "elu":
            return F.elu(x)
        else:
            raise ValueError(
                f"Unsupported activation function: {activation}. "
                f"Supported: 'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'."
            )

    def _create_sequential_conv_blocks(
        self,
        in_channels: int,
        out_channels_list: List[int],
        kernel_sizes: Union[int, List[int]],
        conv_type: str = "2d",
        strides: Union[int, List[int]] = 1,
        paddings: Union[int, List[int], str] = "same",
        use_batch_norm: bool = True,
        bias: bool = False,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Create a sequence of convolutional layers with optional batch normalization.

        This method generates convolution and batch normalization layers based on the
        specified parameters. It supports both 1D and 2D convolutions.

        Args:
            in_channels (int): Number of input channels for the first convolutional layer.
            out_channels_list (List[int]): List of output channels for each convolutional layer.
            kernel_sizes (Union[int, List[int]]): Kernel size(s) for the convolutional layers.
                If an integer, the same kernel size is used for all layers.
            conv_type (str, optional): Type of convolution ('1d' or '2d'). Defaults to '2d'.
            strides (Union[int, List[int]], optional): Stride(s) for the convolutional layers.
                If an integer, the same stride is used for all layers. Defaults to 1.
            paddings (Union[int, List[int], str], optional): Padding for the convolutional layers.
                Can be an integer, a list of integers, or 'same' for automatic padding.
                Defaults to 'same'.
            use_batch_norm (bool, optional): Whether to include batch normalization layers.
                Defaults to True.
            bias (bool, optional): Whether to include bias in convolutional layers.
                Typically set to False when using batch normalization. Defaults to False.

        Returns:
            Tuple[nn.ModuleList, nn.ModuleList]: A tuple containing:
                - conv_layers: ModuleList of convolutional layers
                - batch_norms: ModuleList of batch normalization layers (or Identity layers if disabled)

        Raises:
            ValueError: If conv_type is not '1d' or '2d'.

        Examples:
            >>> conv_layers, batch_norms = self._create_sequential_conv_blocks(
            ...     in_channels=3, out_channels_list=[32, 64, 128],
            ...     kernel_sizes=[3, 3, 3], conv_type='2d'
            ... )
        """
        if conv_type not in ["1d", "2d"]:
            raise ValueError(f"conv_type must be '1d' or '2d', got '{conv_type}'")

        conv_class = nn.Conv1d if conv_type == "1d" else nn.Conv2d
        bn_class = nn.BatchNorm1d if conv_type == "1d" else nn.BatchNorm2d

        num_layers = len(out_channels_list)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        if isinstance(strides, int):
            strides = [strides] * num_layers
        if isinstance(paddings, int):
            paddings = [paddings] * num_layers
        elif paddings == "same":
            paddings = [(k - 1) // 2 for k in kernel_sizes]

        conv_layers = nn.ModuleList()
        batch_norms = nn.ModuleList()

        current_in_channels = in_channels
        for out_channels, kernel_size, stride, padding in zip(out_channels_list, kernel_sizes, strides, paddings):
            conv = conv_class(
                in_channels=current_in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
            conv_layers.append(conv)

            if use_batch_norm:
                batch_norms.append(bn_class(out_channels))
            else:
                batch_norms.append(nn.Identity())

            current_in_channels = out_channels

        return conv_layers, batch_norms

    def _create_doubling_conv_blocks(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        kernel_sizes: Union[int, List[int]],
        conv_type: str = "2d",
        strides: Union[int, List[int]] = 1,
        paddings: Union[int, List[int], str] = "same",
        use_batch_norm: bool = True,
        bias: bool = False,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Create convolutional blocks where the number of output channels doubles at each layer.

        This is a common pattern in CNNs where feature maps progressively increase in depth
        while spatial dimensions are reduced through pooling.

        Args:
            in_channels (int): Number of input channels for the first convolutional layer.
            base_channels (int): Number of output channels for the first layer.
                Each subsequent layer will have 2x the previous layer's channels.
            num_layers (int): Number of convolutional layers to create.
            kernel_sizes (Union[int, List[int]]): Kernel size(s) for the convolutional layers.
            conv_type (str, optional): Type of convolution ('1d' or '2d'). Defaults to '2d'.
            strides (Union[int, List[int]], optional): Stride(s) for the convolutional layers.
                Defaults to 1.
            paddings (Union[int, List[int], str], optional): Padding for the convolutional layers.
                Defaults to 'same'.
            use_batch_norm (bool, optional): Whether to include batch normalization layers.
                Defaults to True.
            bias (bool, optional): Whether to include bias in convolutional layers.
                Defaults to False.

        Returns:
            Tuple[nn.ModuleList, nn.ModuleList]: A tuple containing:
                - conv_layers: ModuleList of convolutional layers
                - batch_norms: ModuleList of batch normalization layers

        Examples:
            >>> # Creates layers with [64, 128, 256] output channels
            >>> conv_layers, batch_norms = self._create_doubling_conv_blocks(
            ...     in_channels=3, base_channels=64, num_layers=3,
            ...     kernel_sizes=3, conv_type='2d'
            ... )
        """
        out_channels_list = [base_channels * (2**i) for i in range(num_layers)]

        return self._create_sequential_conv_blocks(
            in_channels=in_channels,
            out_channels_list=out_channels_list,
            kernel_sizes=kernel_sizes,
            conv_type=conv_type,
            strides=strides,
            paddings=paddings,
            use_batch_norm=use_batch_norm,
            bias=bias,
        )

    def _initialize_weights(self, method: str = "kaiming") -> None:
        """
        Initialize the weights of the network using the specified method.

        Args:
            method (str, optional): Weight initialization method. Supported values:
                'kaiming' (He initialization), 'xavier', 'normal', 'uniform'.
                Defaults to 'kaiming'.

        Raises:
            ValueError: If the initialization method is not supported.

        Examples:
            >>> self._initialize_weights(method='kaiming')
        """
        method = method.lower()
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                if method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                elif method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif method == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif method == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                else:
                    raise ValueError(
                        f"Unsupported initialization method: {method}. "
                        f"Supported: 'kaiming', 'xavier', 'normal', 'uniform'."
                    )

                # Initialize bias if present
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # Standard batch norm initialization
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _flatten_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten the input tensor to shape (batch_size, features).

        This is commonly used before fully connected layers to convert
        spatial feature maps to a 1D feature vector per sample.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, *spatial_dims).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, channels * spatial_dims).

        Examples:
            >>> x = torch.randn(32, 64, 7, 7)  # (batch, channels, height, width)
            >>> x_flat = self._flatten_features(x)
            >>> x_flat.shape  # torch.Size([32, 3136])
        """
        return x.view(x.size(0), -1)

    def _apply_pooling(
        self,
        x: torch.Tensor,
        pool_type: str = "max",
        pool_size: int = 2,
        adaptive_output_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply pooling operation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.
            pool_type (str, optional): Type of pooling ('max', 'avg', 'adaptive_max', 'adaptive_avg').
                Defaults to 'max'.
            pool_size (int, optional): Size of the pooling window for standard pooling.
                Defaults to 2.
            adaptive_output_size (int, optional): Output size for adaptive pooling.
                Required when using 'adaptive_max' or 'adaptive_avg'.

        Returns:
            torch.Tensor: Pooled tensor.

        Raises:
            ValueError: If pool_type is not supported or adaptive_output_size is missing
                for adaptive pooling.

        Examples:
            >>> x = torch.randn(32, 64, 28, 28)
            >>> x_pooled = self._apply_pooling(x, pool_type='max', pool_size=2)
            >>> x_pooled.shape  # torch.Size([32, 64, 14, 14])
        """
        pool_type = pool_type.lower()
        ndim = x.ndim - 2  # Subtract batch and channel dimensions

        if pool_type == "max":
            if ndim == 1:
                return F.max_pool1d(x, kernel_size=pool_size)
            elif ndim == 2:
                return F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            if ndim == 1:
                return F.avg_pool1d(x, kernel_size=pool_size)
            elif ndim == 2:
                return F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "adaptive_max":
            if adaptive_output_size is None:
                raise ValueError("adaptive_output_size must be specified for adaptive pooling")
            if ndim == 1:
                return F.adaptive_max_pool1d(x, output_size=adaptive_output_size)
            elif ndim == 2:
                return F.adaptive_max_pool2d(x, output_size=adaptive_output_size)
        elif pool_type == "adaptive_avg":
            if adaptive_output_size is None:
                raise ValueError("adaptive_output_size must be specified for adaptive pooling")
            if ndim == 1:
                return F.adaptive_avg_pool1d(x, output_size=adaptive_output_size)
            elif ndim == 2:
                return F.adaptive_avg_pool2d(x, output_size=adaptive_output_size)
        else:
            raise ValueError(
                f"Unsupported pool_type: {pool_type}. " f"Supported: 'max', 'avg', 'adaptive_max', 'adaptive_avg'."
            )

    def _create_embedding_layer(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ) -> nn.Embedding:
        """
        Create an embedding layer for sequence or categorical data.

        Args:
            num_embeddings (int): Size of the embedding dictionary (vocabulary size).
            embedding_dim (int): Dimensionality of the embedding vectors.
            padding_idx (int, optional): Index of the padding token. If specified,
                embeddings at this index are initialized to zeros and not updated during training.

        Returns:
            nn.Embedding: Embedding layer.

        Examples:
            >>> embedding = self._create_embedding_layer(num_embeddings=1000, embedding_dim=128)
        """
        if padding_idx is not None:
            return nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        return nn.Embedding(num_embeddings, embedding_dim)

    def _get_conv_output_size(
        self, input_size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
    ) -> int:
        """
        Calculate the output size of a convolutional layer given input parameters.

        Args:
            input_size (int): Size of the input dimension (height or width).
            kernel_size (int): Size of the convolving kernel.
            stride (int, optional): Stride of the convolution. Defaults to 1.
            padding (int, optional): Zero-padding added to both sides. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.

        Returns:
            int: Output size of the convolutional layer.

        Examples:
            >>> output_size = self._get_conv_output_size(input_size=28, kernel_size=3, padding=1)
            >>> output_size  # 28
        """
        return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class CNNEncoder(nn.Module):
    r"""
    The DeepDTA's CNN encoder module, which comprises three 1D-convolutional layers and one max-pooling layer.
    The module is applied to encoding drug/target sequence information, and the input should be transformed information
    with integer/label encoding. The original paper is `"DeepDTA: deep drug–target binding affinity prediction"
    <https://academic.oup.com/bioinformatics/article/34/17/i821/5093245>`_.

    Args:
        num_embeddings (int): Number of embedding labels/categories, depends on the types of encoding sequence.
        embedding_dim (int): Dimension of embedding labels/categories.
        sequence_length (int): Max length of the input sequence.
        num_kernels (int): Number of kernels (filters).
        kernel_length (int): Length of kernel (filter).
    """

    def __init__(self, num_embeddings, embedding_dim, sequence_length, num_kernels, kernel_length):
        super(CNNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings + 1, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=sequence_length, out_channels=num_kernels, kernel_size=kernel_length)
        self.conv2 = nn.Conv1d(in_channels=num_kernels, out_channels=num_kernels * 2, kernel_size=kernel_length)
        self.conv3 = nn.Conv1d(in_channels=num_kernels * 2, out_channels=num_kernels * 3, kernel_size=kernel_length)
        self.global_max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x)
        x = x.squeeze(2)
        return x


class ProteinCNN(nn.Module):
    """
    A protein feature extractor using Convolutional Neural Networks (CNNs).

    This class extracts features from protein sequences using a series of 1D convolutional layers.
    The input protein sequence is first embedded and then passed through multiple convolutional
    and batch normalization layers to produce a fixed-size feature vector.

    Args:
        embedding_dim (int): Dimensionality of the embedding space for protein sequences.
        num_filters (list of int): A list specifying the number of filters for each convolutional layer.
        kernel_size (list of int): A list specifying the kernel size for each convolutional layer.
        padding (bool): Whether to apply padding to the embedding layer.
    """

    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        # self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


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

        encoder_layer = nn.TransformerEncoderLayer(num_channels, num_heads, dim_feedforward, dropout)
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
