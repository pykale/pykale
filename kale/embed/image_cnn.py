"""CNNs for extracting features from small images of size 32x32 (e.g. MNIST) and regular images of size 224x224 (e.g.
ImageNet). The code is based on https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/modules.py,
 which is for domain adaptation.
"""

import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


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
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Note:
        Code adapted by pytorch-ada from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self, pretrained=True):
        super(ResNet18Feature, self).__init__()
        model_resnet18 = models.resnet18(pretrained)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self._out_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class ResNet34Feature(nn.Module):
    """
    Modified ResNet34 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Note:
        Code adapted by pytorch-ada from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self, pretrained=True):
        super(ResNet34Feature, self).__init__()
        model_resnet34 = models.resnet34(pretrained)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self._out_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class ResNet50Feature(nn.Module):
    """
    Modified ResNet50 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Note:
        Code adapted by pytorch-ada from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self, pretrained=True):
        super(ResNet50Feature, self).__init__()
        model_resnet50 = models.resnet50(pretrained)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self._out_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class ResNet101Feature(nn.Module):
    """
    Modified ResNet101 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Note:
        Code adapted by pytorch-ada from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self, pretrained=True):
        super(ResNet101Feature, self).__init__()
        model_resnet101 = models.resnet101(pretrained)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self._out_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class ResNet152Feature(nn.Module):
    """
    Modified ResNet152 (without the last layer) feature extractor for regular 224x224 images.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Note:
        Code adapted by pytorch-ada from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self, pretrained=True):
        super(ResNet152Feature, self).__init__()
        model_resnet152 = models.resnet152(pretrained)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self._out_features = model_resnet152.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


class LeNet(nn.Module):
    """LeNet is a customizable Convolutional Neural Network (CNN) model based on the LeNet architecture, designed for feature extraction from image and audio modalities.
       LeNet supports several layers of 2D convolution, followed by batch normalization, max pooling, and adaptive average pooling, with a configurable number of channels. The depth of the network (number of convolutional blocks) is adjustable with the 'additional_layers' parameter.
       An optional linear layer can be added at the end for further transformation of the output, which could be useful for various tasks such as classification or regression. The 'output_each_layer' option allows for returning the output of each layer instead of just the final output, which can be beneficial for certain tasks or for analyzing the intermediate representations learned by the network.
       By default, the output tensor is squeezed before being returned, removing dimensions of size one, but this can be configured with the 'squeeze_output' parameter.

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
                    (2 ** i) * output_channels, (2 ** (i + 1)) * output_channels, kernel_size=3, padding=1, bias=False
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
