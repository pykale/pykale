# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""
Define Inflated 3D ConvNets(I3D) on Action Recognition from https://ieeexplore.ieee.org/document/8099985
Created by Xianyuan Liu from modifying https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py and
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

model_urls = {
    "rgb_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
    "flow_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_imagenet.pt",
    "rgb_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_charades.pt",
    "flow_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_charades.pt",
}


class MaxPool3dSamePadding(nn.MaxPool3d):
    """
    Construct 3d max pool with same padding. PyTorch does not provide same padding.
    Same padding means the output size matches input size for stride=1.
    """

    def compute_pad(self, dim, s):
        """Get the zero padding number."""

        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        """Compute 'same' padding. Add zero to the back position first."""

        (batch, channel, time, height, width) = x.size()
        pad_t = self.compute_pad(0, time)
        pad_h = self.compute_pad(1, height)
        pad_w = self.compute_pad(2, width)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        pad = (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name="unit_3d",
    ):
        """Initializes Unit3D module."""

        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            # we always want padding to be 0 here. We will dynamically pad based on input size in
            # forward function
            bias=self._use_bias,
        )

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        """Get the zero padding number."""

        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        """
        Connects the module to inputs. Dynamically pad based on input size in forward function.
        Args:
            x: Inputs to the Unit3D component.

        Returns:
            Outputs from the module.
        """

        # compute 'same' padding
        (batch, channel, time, height, width) = x.size()
        pad_t = self.compute_pad(0, time)
        pad_h = self.compute_pad(1, height)
        pad_w = self.compute_pad(2, width)

        pad_t_front = pad_t // 2
        pad_t_back = pad_t - pad_t_front
        pad_h_front = pad_h // 2
        pad_h_back = pad_h - pad_h_front
        pad_w_front = pad_w // 2
        pad_w_back = pad_w - pad_w_front

        pad = (pad_w_front, pad_w_back, pad_h_front, pad_h_back, pad_t_front, pad_t_back)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """
    Construct Inception module. Concatenation after four branches (1x1x1 conv; 1x1x1 + 3x3x3 convs; 1x1x1 + 3x3x3
    convs; 3x3x3 max-pool + 1x1x1 conv). In `forward`, we check if SELayers are used, which are
    channel-wise (SELayerC), temporal-wise (SELayerT), channel-temporal-wise (SELayerTC & SELayerCT).
    """

    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_0/Conv3d_0a_1x1",
        )
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_1/Conv3d_0a_1x1",
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_1/Conv3d_0b_3x3",
        )
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_2/Conv3d_0a_1x1",
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            name=name + "/Branch_2/Conv3d_0b_3x3",
        )
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name + "/Branch_3/Conv3d_0b_1x1",
        )
        self.name = name

    def _forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))

        output = [b0, b1, b2, b3]
        return output

    def forward(self, x):
        outputs = self._forward(x)
        out = torch.cat(outputs, dim=1)

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

        return out


class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        "Conv3d_1a_7x7",
        "MaxPool3d_2a_3x3",
        "Conv3d_2b_1x1",
        "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
        "Mixed_3b",
        "Mixed_3c",
        "MaxPool3d_4a_3x3",
        "Mixed_4b",
        "Mixed_4c",
        "Mixed_4d",
        "Mixed_4e",
        "Mixed_4f",
        "MaxPool3d_5a_2x2",
        "Mixed_5b",
        "Mixed_5c",
        "Logits",
        "Predictions",
    )

    def __init__(
        self,
        num_classes=400,
        spatial_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
    ):
        """
        Initializes I3D model instance.

        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset). Use `replace_logits` to update num_classes.
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.

        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError("Unknown final endpoint %s" % self._final_endpoint)

        """Construct I3D architecture."""
        self.end_points = {}
        end_point = "Conv3d_1a_7x7"
        self.end_points[end_point] = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name + end_point,
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_2a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2b_1x1"
        self.end_points[end_point] = Unit3D(
            in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Conv3d_2c_3x3"
        self.end_points[end_point] = Unit3D(
            in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name=name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_3a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3b"
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_3c"
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_4a_3x3"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4b"
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4c"
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4d"
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4e"
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = InceptionModule(
            112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "MaxPool3d_5a_2x2"
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5b"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = InceptionModule(
            256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Logits"
        # self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        # self.avg_pool_flow = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=400,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

        self.build()

    def replace_logits(self, num_classes):
        """Update the output size with num_classes according to the specific setting."""

        self._num_classes = num_classes
        self.logits = Unit3D(
            in_channels=384 + 384 + 128 + 128,
            output_channels=self._num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name="logits",
        )

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        """The output is the result of the final average pooling layer with 1024 dimensions."""

        # x = self._modules["Conv3d_1a_7x7"](x)  # out: [2, 64, 1, 112, 112]
        # x = self._modules["MaxPool3d_2a_3x3"](x)  # [2, 64, 1, 56, 56]
        # x = self._modules["Conv3d_2b_1x1"](x)  # [2, 64, 1, 56, 56]
        # x = self._modules["Conv3d_2c_3x3"](x)  # [2, 192, 1, 56, 56]
        # x = self._modules["MaxPool3d_3a_3x3"](x)  # [2, 192, 1, 28, 28]
        # x = self._modules["Mixed_3b"](x)  # [2, 256, 1, 28, 28]
        # x = self._modules["Mixed_3c"](x)  # [2, 480, 1, 28, 28]
        # x = self._modules["MaxPool3d_4a_3x3"](x)  # [2, 480, 1, 14, 14]
        # x = self._modules["Mixed_4b"](x)  # [2, 512, 1, 14, 14]
        # x = self._modules["Mixed_4c"](x)  # [2, 512, 1, 14, 14]
        # x = self._modules["Mixed_4d"](x)  # [2, 512, 1, 14, 14]
        # x = self._modules["Mixed_4e"](x)  # [2, 528, 1, 14, 14]
        # x = self._modules["Mixed_4f"](x)  # [2, 832, 1, 14, 14]
        # x = self._modules["MaxPool3d_5a_2x2"](x)  # [2, 832, 1, 7, 7]
        # x = self._modules["Mixed_5b"](x)  # [2, 832, 1, 7, 7]
        # x = self._modules["Mixed_5c"](x)  # [2, 1024, 1, 7, 7]

        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.avg_pool(x)
        # logits = self.logits(self.dropout(x))
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        # x is batch X time X classes, which is what we want to work with
        return x

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def i3d(name, num_channels, num_classes, pretrained=False, progress=True):
    """Get InceptionI3d module w/o pretrained model."""
    model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)

        # delete the last layer's parameter and only load the parameters before the last due to different class number.
        # uncomment and change the output size of I3D when using the default classifier in I3D.

        # state_dict.pop("logits.conv3d.weight")
        # state_dict.pop("logits.conv3d.bias")
        # model.load_state_dict(state_dict, strict=False)

        model.load_state_dict(state_dict)
    return model


def i3d_joint(rgb_pt, flow_pt, num_classes, pretrained=False, progress=True):
    """Get I3D models for different inputs.

    Args:
        rgb_pt (string, optional): the name of pre-trained model for RGB input.
        flow_pt (string, optional): the name of pre-trained model for flow input.
        num_classes (int): the class number of dataset.
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        models (dictionary): A dictionary contains RGB and flow models.
    """
    i3d_rgb = i3d_flow = None
    if rgb_pt is not None:
        i3d_rgb = i3d(name=rgb_pt, num_channels=3, num_classes=num_classes, pretrained=pretrained, progress=progress)
    if flow_pt is not None:
        i3d_flow = i3d(name=flow_pt, num_channels=2, num_classes=num_classes, pretrained=pretrained, progress=progress)
    models = {"rgb": i3d_rgb, "flow": i3d_flow}
    return models
