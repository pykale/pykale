"""
CNNs for extracting features from small images of size 32x32 (e.g. MNIST) and regular images of size 224x224 (e.g. ImageNet). The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/modules.py, which is for domain adaptation.
"""

import torch.nn as nn
from torchvision import models


# From FeatureExtractorDigits in adalib
class SmallCNNFeature(nn.Module):
    """
    A feature extractor for small 32x32 images (e.g. CIFAR, MNIST) that outputs a feature vector of length 128.

    Args:
        num_channels: the number of input channels (default=3).
        ckernel_size: the size of the convolution kernel (default=5).

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

    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = self.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features


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
