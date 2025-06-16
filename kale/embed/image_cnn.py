"""CNNs for extracting features from small images of size 32x32 (e.g. MNIST) and regular images of size 224x224 (e.g.
ImageNet). The code is based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/modules.py,
 which is for domain adaptation.
"""
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


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
class SmallCNNFeature(nn.Module):
    """
    A feature extractor for small 32x32 images (e.g. CIFAR, MNIST) that outputs a feature vector of length 128.

    Args:
        num_channels (int): the number of input channels (default=3).
        kernel_size (int): the size of the convolution kernel (default=5).

    Examples::
        >>> feature_network = SmallCNNFeature(num_channels)
    """

    def __init__(self, num_channels=3, kernel_size=5):
        super(SmallCNNFeature, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64 * 2, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.sigmoid = nn.Sigmoid()
        self._out_features = 128

    def forward(self, input_):
        x = self.bn1(self.conv1(input_))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = self.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class SimpleCNNBuilder(nn.Module):
    """A builder for simple CNNs to experiment with different basic architectures.

    Args:
        num_channels (int, optional): the number of input channels. Defaults to 3.
        conv_layers_spec (list): a list for each convolutional layer given as [num_channels, kernel_size].
            For example, [[16, 3], [16, 1]] represents 2 layers with 16 filters and kernel sizes of 3 and 1 respectively.
        activation_fun (str): a string specifying the activation function to use. one of ('relu', 'elu', 'leaky_relu').
            Defaults to "relu".
        use_batchnorm (boolean): a boolean flag indicating whether to use batch normalization. Defaults to True.
        pool_locations (tuple): the index after which pooling layers should be placed in the convolutional layer list.
            Defaults to (0,3). (0,3) means placing 2 pooling layers after the first and fourth convolutional layer.
        num_channels (int): the number of input channels. Defaults to 3.
    """

    activations = {"relu": nn.ReLU(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU()}

    def __init__(
        self, conv_layers_spec, activation_fun="relu", use_batchnorm=True, pool_locations=(0, 3), num_channels=3
    ):
        super(SimpleCNNBuilder, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = num_channels
        activation_fun = self.activations[activation_fun]

        # Repetitively adds a convolution, batch-norm, activation Function, and max-pooling layer.
        for layer_num, (num_kernels, kernel_size) in enumerate(conv_layers_spec):
            conv = nn.Conv2d(in_channels, num_kernels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.layers.append(conv)

            if use_batchnorm:
                self.layers.append(nn.BatchNorm2d(num_kernels))

            self.layers.append(activation_fun)

            if layer_num in pool_locations:
                self.layers.append(nn.MaxPool2d(kernel_size=2))

            in_channels = num_kernels

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x


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
        return self.model(x)

    def output_size(self):
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
        return self.model(x)

    def output_size(self):
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
        return self.model(x)

    def output_size(self):
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
        return self.model(x)

    def output_size(self):
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
        return self.model(x)

    def output_size(self):
        return self._out_features


class LeNet(nn.Module):
    """LeNet is a customizable Convolutional Neural Network (CNN) model based on the LeNet architecture, designed for
    feature extraction from image and audio modalities.
       LeNet supports several layers of 2D convolution, followed by batch normalization, max pooling, and adaptive
       average pooling, with a configurable number of channels.
       The depth of the network (number of convolutional blocks) is adjustable with the 'additional_layers' parameter.
       An optional linear layer can be added at the end for further transformation of the output, which could be useful
       for various tasks such as classification or regression. The 'output_each_layer' option allows for returning the
       output of each layer instead of just the final output, which can be beneficial for certain tasks or for analyzing
       the intermediate representations learned by the network.
       By default, the output tensor is squeezed before being returned, removing dimensions of size one, but this can be
       configured with the 'squeeze_output' parameter.

    Args:
        input_channels (int): Input channel number.
        output_channels (int): Output channel number for block.
        additional_layers (int): Number of additional blocks for LeNet.
        output_each_layer (bool, optional): Whether to return the output of all layers. Defaults to False.
        linear (tuple, optional): Tuple of (input_dim, output_dim) for optional linear layer post-processing. Defaults to None.
        squeeze_output (bool, optional): Whether to squeeze output before returning. Defaults to True.

    Note:
        Adapted code from https://github.com/slyviacassell/_MFAS/blob/master/models/central/avmnist.py.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        additional_layers,
        output_each_layer=False,
        linear=None,
        squeeze_output=True,
    ):
        super(LeNet, self).__init__()
        self.output_each_layer = output_each_layer
        self.conv_layers = [nn.Conv2d(input_channels, output_channels, kernel_size=5, padding=2, bias=False)]
        self.batch_norms = [nn.BatchNorm2d(output_channels)]
        self.global_pools = [nn.AdaptiveAvgPool2d(1)]

        for i in range(additional_layers):
            self.conv_layers.append(
                nn.Conv2d(
                    (2**i) * output_channels, (2 ** (i + 1)) * output_channels, kernel_size=3, padding=1, bias=False
                )
            )
            self.batch_norms.append(nn.BatchNorm2d(output_channels * (2 ** (i + 1))))
            self.global_pools.append(nn.AdaptiveAvgPool2d(1))

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.batch_norms = nn.ModuleList(self.batch_norms)
        self.global_pools = nn.ModuleList(self.global_pools)
        self.squeeze_output = squeeze_output
        self.linear = None

        if linear is not None:
            self.linear = nn.Linear(linear[0], linear[1])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        intermediate_outputs = []
        output = x
        for i in range(len(self.conv_layers)):
            output = F.relu(self.batch_norms[i](self.conv_layers[i](output)))
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


class ImageVAEEncoder(nn.Module):
    """
    ImageVAEEncoder encodes 2D image data into a latent representation for use in a Variational Autoencoder (VAE).

    Note:
        This implementation assumes the input images are 224 x 224 pixels.
        If you use images of a different size, you must modify the architecture (e.g., adjust the linear layer input).

    This encoder consists of a stack of convolutional layers followed by fully connected layers to produce the
    mean and log-variance of the latent Gaussian distribution. It is suitable for compressing image modalities
    (such as chest X-rays) into a lower-dimensional latent space, facilitating downstream tasks like reconstruction,
    multimodal learning, or generative modelling.

    Args:
        input_channels (int, optional): Number of input channels in the image (e.g., 1 for grayscale, 3 for RGB). Default is 1.
        latent_dim (int, optional): Dimensionality of the latent space representation. Default is 256.

    Forward Input:
        x (Tensor): Input image tensor of shape (batch_size, input_channels, 224, 224).

    Forward Output:
        mean (Tensor): Mean vector of the latent Gaussian distribution, shape (batch_size, latent_dim).
        log_var (Tensor): Log-variance vector of the latent Gaussian, shape (batch_size, latent_dim).

    Example:
        encoder = ImageVAEEncoder(input_channels=1, latent_dim=128)
        mean, log_var = encoder(images)
    """

    def __init__(self, input_channels=1, latent_dim=256):
        super().__init__()
        # Convolutional layers for 224x224 input
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 28 * 28, latent_dim)
        self.fc_log_var = nn.Linear(64 * 28 * 28, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for 224 x 224 images.

        Args:
            x (Tensor): Input image tensor, shape (batch_size, input_channels, 224, 224)

        Returns:
            mean (Tensor): Latent mean, shape (batch_size, latent_dim)
            log_var (Tensor): Latent log-variance, shape (batch_size, latent_dim)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mean = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mean, log_var
