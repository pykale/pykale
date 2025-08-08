# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""
Define MC3_18, R3D_18, R2plus1D_18 on Action Recognition from https://arxiv.org/abs/1711.11248
Created by Xianyuan Liu from modifying https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py
"""

import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


class Conv3DSimple(nn.Conv3d):
    """3D convolutions for R3D (3x3x3 kernel)"""

    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv2Plus1D(nn.Sequential):
    """(2+1)D convolutions for R2plus1D (1x3x3 kernel + 3x1x1 kernel)"""

    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes, out_planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
    """3D convolutions without temporal dimension for MCx (1x3x3 kernel)"""

    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class BasicBlock(nn.Module):
    """
    Basic ResNet building block. Each block consists of two convolutional layers with a ReLU activation function
    after each layer and residual connections. In `forward`, we check if SELayers are used, which are
    channel-wise (SELayerC) and temporal-wise (SELayerT).
    """

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        # Check if SELayer is used.
        if "SELayerC" in dir(self):  # check channel-wise
            out = self.SELayerC(out)
        if "SELayerCoC" in dir(self):
            out = self.SELayerCoC(out)
        if "SELayerMC" in dir(self):
            out = self.SELayerMC(out)
        if "SELayerMAC" in dir(self):
            out = self.SELayerMAC(out)

        if "SELayerT" in dir(self):  # check temporal-wise
            out = self.SELayerT(out)

        if "SELayerCTc" in dir(self):  # check channel-temporal-wise
            out = self.SELayerCTc(out)
        if "SELayerCTt" in dir(self):
            out = self.SELayerCTt(out)

        if "SELayerTCt" in dir(self):  # check temporal-channel-wise
            out = self.SELayerTCt(out)
        if "SELayerTCc" in dir(self):
            out = self.SELayerTCc(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    BottleNeck building block. Default: No use. Each block consists of two 1*n*n and one n*n*n convolutional layers
    with a ReLU activation function after each layer and residual connections.
    """

    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride), nn.BatchNorm3d(planes), nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem. The first layer normally. (64 3x7x7 kernels)"""

    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class BasicFLowStem(nn.Sequential):
    """The default stem for optical flow."""

    def __init__(self):
        super(BasicFLowStem, self).__init__(
            nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem(nn.Sequential):
    """
    R(2+1)D stem is different than the default one as it uses separated 3D convolution.
    (45 1x7x7 kernels + 64 3x1x1 kernel)
    """

    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class R2Plus1dFlowStem(nn.Sequential):
    """R(2+1)D stem for optical flow."""

    def __init__(self):
        super(R2Plus1dFlowStem, self).__init__(
            nn.Conv3d(2, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, stem, num_classes=400, zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def replace_fc(self, num_classes, block=BasicBlock):
        """Update the output size with num_classes according to the specific setting."""

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.stem(x)  # [b, 64, 16, 112, 112]

        x = self.layer1(x)  # [b, 64, 16, 112, 112]
        x = self.layer2(x)  # [b, 128, 8, 56, 56]
        x = self.layer3(x)  # [b, 256, 4, 28, 28]
        x = self.layer4(x)  # [b, 512, 2, 14, 14]

        x = self.avgpool(x)  # [b, 512, 1, 1, 1]
        # Flatten the layer to fc
        x = x.flatten(1)  # [b, 512]
        # x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def _video_resnet_flow(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop("stem.0.weight")
        model.load_state_dict(state_dict, strict=False)
    return model


def r3d_18_rgb(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model for RGB as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet(
        "r3d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def r3d_18_flow(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model for optical flow."""

    return _video_resnet_flow(
        "r3d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicFLowStem,
        **kwargs,
    )


def mc3_18_rgb(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network for RGB as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """

    return _video_resnet(
        "mc3_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def mc3_18_flow(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network for optical flow."""

    return _video_resnet_flow(
        "mc3_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicFLowStem,
        **kwargs,
    )


def r2plus1d_18_rgb(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network for RGB as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """

    return _video_resnet(
        "r2plus1d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs,
    )


def r2plus1d_18_flow(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network for optical flow."""

    return _video_resnet_flow(
        "r2plus1d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dFlowStem,
        **kwargs,
    )


def r3d(rgb=False, flow=False, pretrained=False, progress=True):
    """Get R3D_18 models."""
    r3d_rgb = r3d_flow = None
    if rgb:
        r3d_rgb = r3d_18_rgb(pretrained=pretrained, progress=progress)
    if flow:
        r3d_flow = r3d_18_flow(pretrained=pretrained, progress=progress)
    models = {"rgb": r3d_rgb, "flow": r3d_flow}
    return models


def mc3(rgb=False, flow=False, pretrained=False, progress=True):
    """Get MC3_18 models."""
    mc3_rgb = mc3_flow = None
    if rgb:
        mc3_rgb = mc3_18_rgb(pretrained=pretrained, progress=progress)
    if flow:
        mc3_flow = mc3_18_flow(pretrained=pretrained, progress=progress)
    models = {"rgb": mc3_rgb, "flow": mc3_flow}
    return models


def r2plus1d(rgb=False, flow=False, pretrained=False, progress=True):
    """Get R2PLUS1D_18 models."""
    r2plus1d_rgb = r2plus1d_flow = None
    if rgb:
        r2plus1d_rgb = r2plus1d_18_rgb(pretrained=pretrained, progress=progress)
    if flow:
        r2plus1d_flow = r2plus1d_18_flow(pretrained=pretrained, progress=progress)
    models = {"rgb": r2plus1d_rgb, "flow": r2plus1d_flow}
    return models
