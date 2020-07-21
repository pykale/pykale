import numpy as np
import torch.nn as nn
import torch
from torchvision import models

class FeatureExtractFF(nn.Module):
    def __init__(
        self, input_dim, hidden_sizes=(15,), activation_fn=nn.ReLU, **activation_args
    ):
        super(FeatureExtractFF, self).__init__()
        self._in = input_dim
        self._hidden_sizes = hidden_sizes
        self._activation_fn = activation_fn
        self._activation_args = activation_args

        self.feature = nn.Sequential()
        hin = self._in
        for i, h in enumerate(self._hidden_sizes):
            self.feature.add_module(f"f_fc{i}", nn.Linear(hin, h))
            self.feature.add_module(
                f"f_{activation_fn.__name__}{i}", activation_fn(**activation_args)
            )
            hin = h

        self._out_features = hin

    def forward(self, input_data):
        return self.feature(input_data)

    def extra_repr(self):
        return f"FC: {self.hidden_sizes}x{self._activation_fn.__name__}"

    def hidden_layer(self, index=0):
        return self.feature[index * 2]

    def output_size(self):
        return self._out_features


class FeatureExtractorDigits(nn.Module):
    """
    Feature extractor for MNIST-like data
    """

    def __init__(self, num_channels=3, kernel_size=5):
        super(FeatureExtractorDigits, self).__init__()
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


class AlexNetFeature(nn.Module):
    """
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(AlexNetFeature, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in xrange(6):
            self.classifier.add_module(
                "classifier" + str(i), model_alexnet.classifier[i]
            )
        self._out_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_size(self):
        return self._out_features


class ResNet18Feature(nn.Module):
    """
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(ResNet18Feature, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
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
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(ResNet34Feature, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
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
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(ResNet50Feature, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
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
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(ResNet101Feature, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
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
    PyTorch model convnet without the last layer
    adapted from https://github.com/thuml/Xlearn/blob/master/pytorch/src/network.py
    """

    def __init__(self):
        super(ResNet152Feature, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
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
