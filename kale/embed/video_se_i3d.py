"""
Define Inflated 3D ConvNets(I3D) on Action Recognition from https://ieeexplore.ieee.org/document/8099985
Created by Xianyuan Liu from modifying https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py and
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from kale.embed.video_i3d import InceptionI3d

__all__ = ['se_i3d_joint', 'InceptionI3d']

model_urls = {
    "rgb_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
    "flow_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_imagenet.pt",
    "rgb_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_charades.pt",
    "flow_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_charades.pt",
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SEInceptionI3D(nn.Module):

    def __init__(self, num_channels):
        super(SEInceptionI3D, self).__init__()
        model = InceptionI3d(in_channels=num_channels)
        model.Mixed_3b.add_module("SELayer", SELayer(256))
        model.Mixed_3c.add_module("SELayer", SELayer(480))
        model.Mixed_4b.add_module("SELayer", SELayer(512))
        model.Mixed_4c.add_module("SELayer", SELayer(512))
        model.Mixed_4d.add_module("SELayer", SELayer(512))
        model.Mixed_4e.add_module("SELayer", SELayer(528))
        model.Mixed_4f.add_module("SELayer", SELayer(832))
        model.Mixed_5b.add_module("SELayer", SELayer(832))
        model.Mixed_5c.add_module("SELayer", SELayer(1024))

        self.model = model

    def forward(self, x):
        return self.model(x)


def se_inception_i3d(name, num_channels, pretrained=False, progress=True):
    """Get InceptionI3d module w/o pretrained model."""
    model = SEInceptionI3D(num_channels)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
        # Create new OrderedDict that add `model.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "model.{}".format(k)
            new_state_dict[name] = v
        # Load params
        model.load_state_dict(new_state_dict, strict=False)
    return model


def se_i3d_joint(rgb_pt, flow_pt, pretrained=False, progress=True):
    """Get I3D models."""
    i3d_rgb = i3d_flow = None
    if rgb_pt is not None and flow_pt is None:
        i3d_rgb = se_inception_i3d(name=rgb_pt, num_channels=3, pretrained=pretrained, progress=progress)
    elif rgb_pt is None and flow_pt is not None:
        i3d_flow = se_inception_i3d(name=flow_pt, num_channels=2, pretrained=pretrained, progress=progress)
    elif rgb_pt is not None and flow_pt is not None:
        i3d_rgb = se_inception_i3d(name=rgb_pt, num_channels=3, pretrained=pretrained, progress=progress)
        i3d_flow = se_inception_i3d(name=flow_pt, num_channels=2, pretrained=pretrained, progress=progress)
    models = {'rgb': i3d_rgb, 'flow': i3d_flow}
    return models
