"""CNNs for extracting features from small images of size 32x32 (e.g. MNIST) and regular images of size 224x224 (e.g.
ImageNet). The code is based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/modules.py,
 which is for domain adaptation.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from kale.embed.cnn import BaseCNN


class Flatten(nn.Module):
    """Flatten layer
    This module is to replace the last fc layer of the pre-trained model with a flatten layer. It flattens the input
    tensor to a 2D vector, which is (B, N). B is the batch size and N is the product of all dimensions except
    the batch size.

    Examples:
        >>> x = torch.randn(8, 3, 224, 224)
        >>> x = Flatten()(x)
        >>> print(x.shape)
        (8, 150528)
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Identity layer
    This module is to replace any unwanted layers in a pre-defined model with an identity layer.
    It returns the input tensor as the output.

    Examples:
        >>> x = torch.randn(8, 3, 224, 224)
        >>> x = Identity()(x)
        >>> print(x.shape)
        (8, 3, 224, 224)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# From FeatureExtractorDigits in adalib
class SmallCNNFeature(BaseCNN):
    """
    A feature extractor for small 32x32 images (e.g. CIFAR, MNIST) that outputs a feature vector of length 128.

    This network uses three convolutional layers with batch normalization and pooling to extract
    hierarchical features from small images. The architecture is specifically designed for 32x32
    input images and produces a fixed-size 128-dimensional feature vector.

    Args:
        num_channels (int): The number of input channels (default=3).
        kernel_size (int): The size of the convolution kernel (default=5).

    Example:
        >>> # Create a feature extractor for RGB images
        >>> feature_network = SmallCNNFeature(num_channels=3, kernel_size=5)
        >>> images = torch.randn(8, 3, 32, 32)  # Batch of 8 RGB 32x32 images
        >>> features = feature_network(images)
        >>> print(features.shape)  # torch.Size([8, 128])
        >>> print(feature_network.output_size())  # 128
    """

    def __init__(self, num_channels: int = 3, kernel_size: int = 5):
        super(SmallCNNFeature, self).__init__()

        # Use padding=0 (no padding) to match original behavior that reduces spatial dimensions
        self.conv_layers, self.batch_norms = self._create_sequential_conv_blocks(
            in_channels=num_channels,
            out_channels_list=[64, 64, 128],
            kernel_sizes=[kernel_size, kernel_size, kernel_size],
            conv_type="2d",
            paddings=[0, 0, 0],  # No padding to reduce spatial dimensions like original
            use_batch_norm=True,
        )

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self._out_features = 128

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            input_ (torch.Tensor): Input image tensor of shape (batch_size, num_channels, 32, 32).

        Returns:
            torch.Tensor: Flattened feature vector of shape (batch_size, 128).
        """
        assert self.batch_norms is not None, "batch_norms should be initialized"
        x = self.batch_norms[0](self.conv_layers[0](input_))
        x = self._apply_activation(self.pool1(x), "relu")

        x = self.batch_norms[1](self.conv_layers[1](x))
        x = self._apply_activation(self.pool2(x), "relu")

        x = self._apply_activation(self.batch_norms[2](self.conv_layers[2](x)), "sigmoid")
        x = self._flatten_features(x)
        return x

    def output_size(self) -> int:
        """
        Get the size of the output feature vector.

        Args:
            None

        Returns:
            int: The dimensionality of the output features (128).
        """
        return self._out_features

    def __repr__(self) -> str:
        """
        Return a string representation of the SmallCNNFeature model.

        Returns:
            str: String describing the model's configuration.
        """
        return (
            f"{self.__class__.__name__}("
            f"num_channels={self.conv_layers[0].in_channels}, "
            f"kernel_size={self.conv_layers[0].kernel_size[0]}, "
            f"output_features={self._out_features})"
        )


class SimpleCNNBuilder(BaseCNN):
    """A builder for simple CNNs to experiment with different basic architectures.

    This class now inherits from BaseCNN to leverage shared utilities and follow FAIR principles.
    The dynamic layer construction allows experimentation with various CNN architectures by
    specifying layer configurations as a list.

    Args:
        num_channels (int, optional): the number of input channels. Defaults to 3.
        conv_layers_spec (list): a list for each convolutional layer given as [num_channels, kernel_size].
            For example, [[16, 3], [16, 1]] represents 2 layers with 16 filters and kernel sizes of 3 and 1 respectively.
        activation_fun (str): a string specifying the activation function to use. one of ('relu', 'elu', 'leaky_relu').
            Defaults to "relu".
        use_batchnorm (boolean): a boolean flag indicating whether to use batch normalization. Defaults to True.
        pool_locations (tuple): the index after which pooling layers should be placed in the convolutional layer list.
            Defaults to (0,3). (0,3) means placing 2 pooling layers after the first and fourth convolutional layer.

    Example:
        >>> # Build a CNN with custom layer specifications
        >>> model = SimpleCNNBuilder(
        ...     conv_layers_spec=[[16, 3], [32, 3], [64, 3]],
        ...     activation_fun='relu',
        ...     use_batchnorm=True,
        ...     num_channels=3
        ... )
        >>> images = torch.randn(8, 3, 32, 32)
        >>> output = model(images)
    """

    def __init__(
        self, conv_layers_spec, activation_fun="relu", use_batchnorm=True, pool_locations=(0, 3), num_channels=3
    ):
        super(SimpleCNNBuilder, self).__init__()
        self.conv_layers_spec = conv_layers_spec
        self.activation_fun = activation_fun
        self.use_batchnorm = use_batchnorm
        self.pool_locations = pool_locations
        self.num_input_channels = num_channels

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        in_channels = num_channels

        for layer_num, (num_kernels, kernel_size) in enumerate(conv_layers_spec):
            conv = nn.Conv2d(in_channels, num_kernels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.conv_layers.append(conv)

            if use_batchnorm:
                self.batch_norms.append(nn.BatchNorm2d(num_kernels))
            else:
                self.batch_norms.append(nn.Identity())

            in_channels = num_kernels

        self._out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dynamically constructed CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through all layers.
        """
        for layer_num, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x = conv(x)
            x = bn(x)
            x = self._apply_activation(x, self.activation_fun)

            if layer_num in self.pool_locations:
                x = F.max_pool2d(x, kernel_size=2)

        return x

    def output_size(self) -> int:
        """
        Return the number of output channels.

        Returns:
            int: Number of output channels from the final convolutional layer.
        """
        return self._out_channels

    def __repr__(self) -> str:
        """Return a string representation of the SimpleCNNBuilder."""
        return (
            f"{self.__class__.__name__}("
            f"num_layers={len(self.conv_layers)}, "
            f"output_channels={self._out_channels}, "
            f"activation={self.activation_fun}, "
            f"use_batchnorm={self.use_batchnorm})"
        )


class _Bottleneck(nn.Module):
    """Simple bottleneck as domain specific feature extractor, used in multi-source domain adaptation method MFSAN only.
    Compared to the torchvision implementation, it accepts both 1D and 2D input, and the value of expansion is
    flexible and an average pooling layer is added.

    The code is based on:
        https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L86,
        https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN/MFSAN_2src/resnet.py#L94, and
        https://github.com/easezyc/deep-transfer-learning/blob/master/MUDA/MFSAN/MFSAN_3src/resnet.py#L93
    """

    def __init__(self, inplanes: int, planes: int, stride: int = 1, expansion: int = 1, input_dimension=2):
        super(_Bottleneck, self).__init__()
        self.input_dimension = input_dimension
        if input_dimension == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
            avgpool = nn.AdaptiveAvgPool1d(1)
        self.expansion = expansion
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = bn(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.avgpool = avgpool

    def forward(self, x):
        if self.input_dimension == 1 and len(x.shape) == 2:
            x = x.view(x.shape + (1,))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        return out


class ResNet18Feature(nn.Module):
    """
    Modified ResNet18 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        weights (models.ResNet18_Weights or string): The pretrained weights to use. See
         https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.ResNet18_Weights
        for more details. By default, ResNet18_Weights.DEFAULT will be used.
    """

    def __init__(self, weights=models.ResNet18_Weights.DEFAULT):
        super(ResNet18Feature, self).__init__()
        self.model = models.resnet18(weights=weights)
        self._out_features = self.model.fc.in_features
        self.model.fc = Flatten()

    def forward(self, x):
        """
        Forward pass through ResNet18 feature extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 512).
        """
        return self.model(x)

    def output_size(self):
        """
        Return the output feature dimension.

        Returns:
            int: Number of output features (512 for ResNet18).
        """
        return self._out_features


class ResNet34Feature(nn.Module):
    """
    Modified ResNet34 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        weights (models.ResNet34_Weights or string): The pretrained weights to use. See
         https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html#torchvision.models.ResNet34_Weights
        for more details. By default, ResNet34_Weights.DEFAULT will be used.
    """

    def __init__(self, weights=models.ResNet34_Weights.DEFAULT):
        super(ResNet34Feature, self).__init__()
        self.model = models.resnet34(weights=weights)
        self._out_features = self.model.fc.in_features
        self.model.fc = Flatten()

    def forward(self, x):
        """
        Forward pass through ResNet34 feature extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 512).
        """
        return self.model(x)

    def output_size(self):
        """
        Return the output feature dimension.

        Returns:
            int: Number of output features (512 for ResNet34).
        """
        return self._out_features


class ResNet50Feature(nn.Module):
    """
    Modified ResNet50 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        weights (models.ResNet50_Weights or string): The pretrained weights to use. See
         https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
        for more details. By default, ResNet50_Weights.DEFAULT will be used.
    """

    def __init__(self, weights=models.ResNet50_Weights.DEFAULT):
        super(ResNet50Feature, self).__init__()
        self.model = models.resnet50(weights=weights)
        self._out_features = self.model.fc.in_features
        self.model.fc = Flatten()

    def forward(self, x):
        """
        Forward pass through ResNet50 feature extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 2048).
        """
        return self.model(x)

    def output_size(self):
        """
        Return the output feature dimension.

        Returns:
            int: Number of output features (2048 for ResNet50).
        """
        return self._out_features


class ResNet101Feature(nn.Module):
    """
    Modified ResNet101 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        weights (models.ResNet101_Weights or string): The pretrained weights to use. See
         https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet101.html#torchvision.models.ResNet101_Weights
        for more details. By default, ResNet101_Weights.DEFAULT will be used.
    """

    def __init__(self, weights=models.ResNet101_Weights.DEFAULT):
        super(ResNet101Feature, self).__init__()
        self.model = models.resnet101(weights=weights)
        self._out_features = self.model.fc.in_features
        self.model.fc = Flatten()

    def forward(self, x):
        """
        Forward pass through ResNet101 feature extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 2048).
        """
        return self.model(x)

    def output_size(self):
        """
        Return the output feature dimension.

        Returns:
            int: Number of output features (2048 for ResNet101).
        """
        return self._out_features


class ResNet152Feature(nn.Module):
    """
    Modified ResNet152 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        weights (models.ResNet152_Weights or string): The pretrained weights to use. See
         https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet152.html#torchvision.models.ResNet152_Weights
        for more details. By default, ResNet152_Weights.DEFAULT will be used.
    """

    def __init__(self, weights=models.ResNet152_Weights.DEFAULT):
        super(ResNet152Feature, self).__init__()
        self.model = models.resnet152(weights=weights)
        self._out_features = self.model.fc.in_features
        self.model.fc = Flatten()

    def forward(self, x):
        """
        Forward pass through ResNet152 feature extractor.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, 2048).
        """
        return self.model(x)

    def output_size(self):
        """
        Return the output feature dimension.

        Returns:
            int: Number of output features (2048 for ResNet152).
        """
        return self._out_features


class LeNet(BaseCNN):
    """
    LeNet is a customizable Convolutional Neural Network (CNN) model based on the LeNet architecture,
    designed for feature extraction from image and audio modalities.

    LeNet supports several layers of 2D convolution, followed by batch normalization, max pooling,
    and adaptive average pooling, with a configurable number of channels. The depth of the network
    (number of convolutional blocks) is adjustable with the 'additional_layers' parameter.

    An optional linear layer can be added at the end for further transformation of the output,
    which could be useful for various tasks such as classification or regression. The
    'output_each_layer' option allows for returning the output of each layer instead of just
    the final output, which can be beneficial for certain tasks or for analyzing the intermediate
    representations learned by the network.

    By default, the output tensor is squeezed before being returned, removing dimensions of size one,
    but this can be configured with the 'squeeze_output' parameter.

    Args:
        input_channels (int): Input channel number.
        output_channels (int): Output channel number for the first block.
        additional_layers (int): Number of additional blocks for LeNet.
        output_each_layer (bool, optional): Whether to return the output of all layers.
            Defaults to False.
        linear (Optional[Tuple[int, int]], optional): Tuple of (input_dim, output_dim) for optional
            linear layer post-processing. Defaults to None.
        squeeze_output (bool, optional): Whether to squeeze output before returning.
            Defaults to True.

    Example:
        >>> # Create a LeNet model for single-channel 32x32 images
        >>> model = LeNet(
        ...     input_channels=1,
        ...     output_channels=4,
        ...     additional_layers=2,
        ...     output_each_layer=False,
        ...     squeeze_output=True
        ... )
        >>> images = torch.randn(2, 1, 32, 32)
        >>> output = model(images)
        >>> print(output.shape)  # torch.Size([2, 16, 4, 4])

    Note:
        Adapted code from https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        additional_layers: int,
        output_each_layer: bool = False,
        linear: Optional[Tuple[int, int]] = None,
        squeeze_output: bool = True,
    ):
        """
        Initialize the LeNet model.

        Args:
            input_channels (int): Input channel number.
            output_channels (int): Output channel number for the first block.
            additional_layers (int): Number of additional blocks.
            output_each_layer (bool): Whether to return outputs from all layers (default=False).
            linear (Optional[Tuple[int, int]]): Tuple of (input_dim, output_dim) for optional
                linear layer (default=None).
            squeeze_output (bool): Whether to squeeze output dimensions (default=True).

        Returns:
            None
        """
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.squeeze_output = squeeze_output

        num_layers = 1 + additional_layers
        self.conv_layers, self.batch_norms, self.global_pools = self._create_doubling_conv_blocks(
            input_channels=input_channels,
            base_channels=output_channels,
            num_layers=num_layers,
            first_kernel_size=5,
            subsequent_kernel_size=3,
            first_padding=2,
            subsequent_padding=1,
            use_batch_norm=True,
            bias=False,
        )

        self.linear = None
        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the LeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: If output_each_layer=True, returns a list
                of tensors from each layer. Otherwise, returns the final output tensor.
                Output shape depends on squeeze_output setting.
        """
        assert self.batch_norms is not None, "batch_norms should be initialized"
        intermediate_outputs = []
        output = x
        for i in range(len(self.conv_layers)):
            output = self._apply_activation(self.batch_norms[i](self.conv_layers[i](output)), "relu")
            output = F.max_pool2d(output, 2)
            global_pool = self.global_pools[i](output).view(output.size(0), -1)
            intermediate_outputs.append(global_pool)

        if self.linear is not None:
            output = self.linear(output)
        intermediate_outputs.append(output)

        if self.output_each_layer:
            if self.squeeze_output:
                return [t.squeeze() for t in intermediate_outputs]
            return intermediate_outputs

        if self.squeeze_output:
            return output.squeeze()
        return output

    def output_size(self) -> int:
        """
        Return the output feature dimension of the LeNet model.

        Returns:
            int: Number of output channels in the final convolutional layer.
                This is (2^(num_layers-1)) * base_channels.
        """
        num_layers = len(self.conv_layers)
        if num_layers > 0:
            # Last layer has channels: (2^(num_layers-1)) * base_channels
            return self.conv_layers[-1].out_channels
        return 0

    def __repr__(self) -> str:
        """Return a string representation of the LeNet model."""
        num_layers = len(self.conv_layers)
        has_linear = self.linear is not None
        return (
            f"{self.__class__.__name__}("
            f"num_layers={num_layers}, "
            f"output_channels={self.output_size()}, "
            f"output_each_layer={self.output_each_layer}, "
            f"has_linear={has_linear}, "
            f"squeeze_output={self.squeeze_output})"
        )


class ImageVAEEncoder(BaseCNN):
    """
    ImageVAEEncoder encodes 2D image data into a latent representation for use in a Variational Autoencoder (VAE).

    This encoder consists of a stack of convolutional layers followed by fully connected
    layers to produce the mean and log-variance of the latent Gaussian distribution.
    It is suitable for compressing image modalities (such as chest X-rays) into a
    lower-dimensional latent space, facilitating downstream tasks like reconstruction,
    multimodal learning, or generative modelling.

    Args:
        input_channels (int, optional): Number of input channels in the image
            (e.g., 1 for grayscale, 3 for RGB). Default is 1.
        latent_dim (int, optional): Dimensionality of the latent space representation.
            Default is 256.

    Forward Input:
        x (Tensor): Input image tensor of shape (batch_size, input_channels, 224, 224).

    Forward Output:
        mean (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        log_var (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        >>> encoder = ImageVAEEncoder(input_channels=1, latent_dim=128)
        >>> images = torch.randn(2, 1, 224, 224)  # Batch of 2 grayscale 224x224 images
        >>> mean, log_var = encoder(images)
        >>> print(mean.shape, log_var.shape)  # torch.Size([2, 128]) torch.Size([2, 128])

    Note:
        This implementation assumes the input images are 224 x 224 pixels.
        If you use images of a different size, you must modify the architecture
        (e.g., adjust the linear layer input).
    """

    def __init__(self, input_channels: int = 1, latent_dim: int = 256):
        """
        Initialize the ImageVAEEncoder model.

        Args:
            input_channels (int): Number of input channels (default=1).
            latent_dim (int): Dimensionality of the latent space (default=256).

        Returns:
            None
        """
        super().__init__()

        self.conv_layers, _ = self._create_sequential_conv_blocks(
            in_channels=input_channels,
            out_channels_list=[16, 32, 64],
            kernel_sizes=[3, 3, 3],
            conv_type="2d",
            strides=[2, 2, 2],
            paddings=[1, 1, 1],
            use_batch_norm=False,
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 28 * 28, latent_dim)
        self.fc_log_var = nn.Linear(64 * 28 * 28, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for 224 x 224 images.

        Args:
            x (Tensor): Input image tensor, shape (batch_size, input_channels, 224, 224)

        Returns:
            mean (Tensor): Latent mean, shape (batch_size, latent_dim)
            log_var (Tensor): Latent log-variance, shape (batch_size, latent_dim)
        """
        for conv in self.conv_layers:
            x = self._apply_activation(conv(x), "relu")

        x = self.flatten(x)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

    def output_size(self) -> int:
        """
        Return the dimensionality of the latent space.

        Returns:
            int: Latent dimension (size of mean and log_var outputs).
        """
        return self.fc_mu.out_features

    def __repr__(self) -> str:
        """Return a string representation of the ImageVAEEncoder."""
        input_channels = self.conv_layers[0].in_channels
        latent_dim = self.fc_mu.out_features
        return f"{self.__class__.__name__}(" f"input_channels={input_channels}, " f"latent_dim={latent_dim})"
