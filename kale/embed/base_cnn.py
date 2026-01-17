"""
Base CNN class providing common utilities for convolutional neural networks.

This module contains the BaseCNN class which serves as a foundation for all CNN
architectures in PyKale. It provides reusable utilities for:
- Creating convolutional blocks with batch normalization
- Applying activation functions consistently
- Weight initialization
- Pooling operations
- Embedding layer creation
- Tensor operations and shape calculations

The BaseCNN class was extracted from kale.embed.cnn to improve module cohesion
and maintainability while keeping all concrete CNN implementations together.

Classes:
    BaseCNN: Base class providing common CNN utilities and patterns.

Example:
    >>> from kale.embed.base_cnn import BaseCNN
    >>> class MyCNN(BaseCNN):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
    ...             in_channels=3, out_channels_list=[32, 64], kernel_sizes=3, conv_type='2d'
    ...         )
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def _validate_conv_block_inputs(self, in_channels: int, out_channels_list: List[int], conv_type: str) -> None:
        """Validate inputs for convolutional block creation."""
        if conv_type not in ["1d", "2d"]:
            raise ValueError(f"conv_type must be '1d' or '2d', got '{conv_type}'")
        if not out_channels_list:
            raise ValueError("out_channels_list cannot be empty")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

    def _normalize_conv_parameters(
        self,
        kernel_sizes: Union[int, List[int]],
        strides: Union[int, List[int]],
        paddings: Union[int, List[int], str],
        num_layers: int,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Normalize convolutional parameters to lists."""
        kernel_sizes_list = [kernel_sizes] * num_layers if isinstance(kernel_sizes, int) else kernel_sizes
        strides_list = [strides] * num_layers if isinstance(strides, int) else strides

        if isinstance(paddings, int):
            paddings_list: List[int] = [paddings] * num_layers
        elif paddings == "same":
            paddings_list = [(k - 1) // 2 for k in kernel_sizes_list]
        else:
            paddings_list = paddings  # type: ignore[assignment]

        return kernel_sizes_list, strides_list, paddings_list

    def _validate_parameter_lengths(
        self,
        kernel_sizes_list: List[int],
        strides_list: List[int],
        paddings_list: List[int],
        num_layers: int,
    ) -> None:
        """Validate that all parameter lists match the expected length."""
        if len(kernel_sizes_list) != num_layers:
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes_list)}) must match out_channels_list length ({num_layers})"
            )
        if len(strides_list) != num_layers:
            raise ValueError(f"strides length ({len(strides_list)}) must match out_channels_list length ({num_layers})")
        if len(paddings_list) != num_layers:
            raise ValueError(
                f"paddings length ({len(paddings_list)}) must match out_channels_list length ({num_layers})"
            )

    def _build_conv_and_bn_layers(
        self,
        in_channels: int,
        out_channels_list: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        conv_class: type,
        bn_class: type,
        use_batch_norm: bool,
        bias: bool,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """Build convolutional and batch normalization layer modules."""
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
        # Validate inputs
        self._validate_conv_block_inputs(in_channels, out_channels_list, conv_type)

        # Select appropriate layer classes
        conv_class = nn.Conv1d if conv_type == "1d" else nn.Conv2d
        bn_class = nn.BatchNorm1d if conv_type == "1d" else nn.BatchNorm2d

        # Normalize parameters to lists
        num_layers = len(out_channels_list)
        kernel_sizes_list, strides_list, paddings_list = self._normalize_conv_parameters(
            kernel_sizes, strides, paddings, num_layers
        )

        # Validate parameter list lengths
        self._validate_parameter_lengths(kernel_sizes_list, strides_list, paddings_list, num_layers)

        # Build layers
        return self._build_conv_and_bn_layers(
            in_channels,
            out_channels_list,
            kernel_sizes_list,
            strides_list,
            paddings_list,
            conv_class,
            bn_class,
            use_batch_norm,
            bias,
        )

    def _create_doubling_conv_blocks(
        self,
        input_channels: int,
        base_channels: int,
        num_layers: int,
        first_kernel_size: int = 5,
        subsequent_kernel_size: int = 3,
        first_padding: int = 2,
        subsequent_padding: int = 1,
        use_batch_norm: bool = True,
        bias: bool = False,
    ) -> Tuple[nn.ModuleList, Optional[nn.ModuleList], nn.ModuleList]:
        """
        Create convolutional blocks with doubling channel pattern for architectures like LeNet.

        This helper creates layers where each subsequent layer doubles the number of channels
        (base_channels → 2*base_channels → 4*base_channels, etc.) along with corresponding
        batch normalization and adaptive average pooling layers.

        Args:
            input_channels (int): Number of input channels for the first layer
            base_channels (int): Base number of output channels (will be doubled for each layer)
            num_layers (int): Total number of convolutional layers to create
            first_kernel_size (int): Kernel size for the first layer (default: 5)
            subsequent_kernel_size (int): Kernel size for subsequent layers (default: 3)
            first_padding (int): Padding for the first layer (default: 2)
            subsequent_padding (int): Padding for subsequent layers (default: 1)
            use_batch_norm (bool): Whether to create batch normalization layers (default: True)
            bias (bool): Whether to include bias in convolution (default: False)

        Returns:
            Tuple[nn.ModuleList, Optional[nn.ModuleList], nn.ModuleList]: Tuple of
                (conv_layers, batch_norms, global_pools)
        """
        conv_layers = nn.ModuleList()
        batch_norms: Optional[nn.ModuleList] = nn.ModuleList() if use_batch_norm else None
        global_pools = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                out_channels = base_channels
                in_ch = input_channels
                kernel_size = first_kernel_size
                padding = first_padding
            else:
                out_channels = (2**i) * base_channels
                in_ch = (2 ** (i - 1)) * base_channels
                kernel_size = subsequent_kernel_size
                padding = subsequent_padding

            conv_layers.append(nn.Conv2d(in_ch, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))

            if use_batch_norm and batch_norms is not None:
                batch_norms.append(nn.BatchNorm2d(out_channels))

            global_pools.append(nn.AdaptiveAvgPool2d(1))

        return conv_layers, batch_norms, global_pools

    def _create_progressive_conv_blocks(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        multipliers: List[int],
        kernel_sizes: Union[int, List[int]],
        conv_type: str = "2d",
        strides: Union[int, List[int]] = 1,
        paddings: Union[int, List[int], str] = "same",
        use_batch_norm: bool = True,
        bias: bool = False,
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        """
        Create convolutional blocks with custom channel multipliers (e.g., [1, 2, 3] for DeepDTA pattern).

        This method allows for flexible channel progression patterns beyond simple doubling.
        Useful for architectures that require specific channel scaling patterns.

        Args:
            in_channels (int): Number of input channels for the first convolutional layer.
            base_channels (int): Base number of output channels.
            num_layers (int): Number of convolutional layers to create.
            multipliers (List[int]): Channel multipliers for each layer (e.g., [1, 2, 3]).
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

        Raises:
            ValueError: If length of multipliers doesn't match num_layers.

        Examples:
            >>> # Creates layers with [32, 64, 96] output channels (DeepDTA pattern)
            >>> conv_layers, batch_norms = self._create_progressive_conv_blocks(
            ...     in_channels=85, base_channels=32, num_layers=3,
            ...     multipliers=[1, 2, 3], kernel_sizes=8, conv_type='1d'
            ... )
        """
        if len(multipliers) != num_layers:
            raise ValueError(f"Length of multipliers ({len(multipliers)}) must match num_layers ({num_layers})")

        out_channels_list = [base_channels * mult for mult in multipliers]

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

    def _init_conv_or_linear_weights(self, module: nn.Module, method: str) -> None:
        """
        Initialize weights for convolutional or linear layers.

        Args:
            module: The layer module to initialize
            method: Initialization method ('kaiming', 'xavier', 'normal', 'uniform')

        Raises:
            ValueError: If method is not supported
        """
        if method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        elif method == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif method == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif method == "uniform":
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)
        else:
            raise ValueError(
                f"Unsupported initialization method: {method}. " f"Supported: 'kaiming', 'xavier', 'normal', 'uniform'."
            )

        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    def _init_batch_norm_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for batch normalization layers.

        Args:
            module: The batch normalization module to initialize
        """
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)

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
                self._init_conv_or_linear_weights(m, method)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._init_batch_norm_weights(m)

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

    def _apply_standard_pooling(self, x: torch.Tensor, pool_type: str, pool_size: int) -> torch.Tensor:
        """
        Apply standard (non-adaptive) pooling operation.

        Args:
            x: Input tensor
            pool_type: Type of pooling ('max' or 'avg')
            pool_size: Size of the pooling window

        Returns:
            Pooled tensor
        """
        ndim = x.ndim - 2
        if pool_type == "max":
            return F.max_pool1d(x, kernel_size=pool_size) if ndim == 1 else F.max_pool2d(x, kernel_size=pool_size)
        else:  # avg
            return F.avg_pool1d(x, kernel_size=pool_size) if ndim == 1 else F.avg_pool2d(x, kernel_size=pool_size)

    def _apply_adaptive_pooling(self, x: torch.Tensor, pool_type: str, output_size: int) -> torch.Tensor:
        """
        Apply adaptive pooling operation.

        Args:
            x: Input tensor
            pool_type: Type of pooling ('adaptive_max' or 'adaptive_avg')
            output_size: Desired output size

        Returns:
            Pooled tensor
        """
        ndim = x.ndim - 2
        if pool_type == "adaptive_max":
            return (
                F.adaptive_max_pool1d(x, output_size=output_size)
                if ndim == 1
                else F.adaptive_max_pool2d(x, output_size=output_size)
            )
        else:  # adaptive_avg
            return (
                F.adaptive_avg_pool1d(x, output_size=output_size)
                if ndim == 1
                else F.adaptive_avg_pool2d(x, output_size=output_size)
            )

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

        if pool_type in ["max", "avg"]:
            return self._apply_standard_pooling(x, pool_type, pool_size)
        elif pool_type in ["adaptive_max", "adaptive_avg"]:
            if adaptive_output_size is None:
                raise ValueError("adaptive_output_size must be specified for adaptive pooling")
            return self._apply_adaptive_pooling(x, pool_type, adaptive_output_size)
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
