# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Add SELayers to I3D"""

import torch.nn as nn
from torch.hub import load_state_dict_from_url

from kale.embed.video_i3d import InceptionI3d
from kale.embed.video_selayer import get_selayer, SELayerC, SELayerT

model_urls = {
    "rgb_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
    "flow_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_imagenet.pt",
    "rgb_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_charades.pt",
    "flow_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_charades.pt",
}


class SEInceptionI3DRGB(nn.Module):
    """Add the several SELayers to I3D for RGB input.
    Args:
        num_channels (int): the channel number of the input.
        num_classes (int): the class number of dataset.
        attention (string): the name of the SELayer.
            (Options: ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerMAC", "SELayerCT" and "SELayerTC"])

    Returns:
        model (VideoResNet): I3D model with SELayers.
    """

    def __init__(self, num_channels, num_classes, attention):
        super(SEInceptionI3DRGB, self).__init__()
        model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)
        temporal_length = 16

        # Add channel-wise SELayer
        if attention in ["SELayerC", "SELayerCoC", "SELayerMC", "SELayerMAC"]:
            se_layer = get_selayer(attention)
            model.Mixed_3b.add_module(attention, se_layer(256))
            model.Mixed_3c.add_module(attention, se_layer(480))
            model.Mixed_4b.add_module(attention, se_layer(512))
            model.Mixed_4c.add_module(attention, se_layer(512))
            model.Mixed_4d.add_module(attention, se_layer(512))
            model.Mixed_4e.add_module(attention, se_layer(528))
            model.Mixed_4f.add_module(attention, se_layer(832))
            model.Mixed_5b.add_module(attention, se_layer(832))
            model.Mixed_5c.add_module(attention, se_layer(1024))

        # Add temporal-wise SELayer
        elif attention == "SELayerT":
            se_layer = get_selayer(attention)
            model.Mixed_3b.add_module(attention, se_layer(temporal_length // 2))
            model.Mixed_3c.add_module(attention, se_layer(temporal_length // 2))
            model.Mixed_4b.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_4c.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_4d.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_4e.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_4f.add_module(attention, se_layer(temporal_length // 4))
            # model.Mixed_5b.add_module(attention, SELayerT(temporal_length//8))
            # model.Mixed_5c.add_module(attention, SELayerT(temporal_length//8))

        # Add channel-temporal-wise SELayer
        elif attention == "SELayerCT":
            model.Mixed_3b.add_module(attention + "c", SELayerC(256))
            model.Mixed_3c.add_module(attention + "c", SELayerC(480))
            model.Mixed_4b.add_module(attention + "c", SELayerC(512))
            model.Mixed_4c.add_module(attention + "c", SELayerC(512))
            model.Mixed_4d.add_module(attention + "c", SELayerC(512))
            model.Mixed_4e.add_module(attention + "c", SELayerC(528))
            model.Mixed_4f.add_module(attention + "c", SELayerC(832))
            model.Mixed_5b.add_module(attention + "c", SELayerC(832))
            model.Mixed_5c.add_module(attention + "c", SELayerC(1024))

            model.Mixed_3b.add_module(attention + "t", SELayerT(temporal_length // 2))
            model.Mixed_3c.add_module(attention + "t", SELayerT(temporal_length // 2))
            model.Mixed_4b.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4c.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4d.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4e.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4f.add_module(attention + "t", SELayerT(temporal_length // 4))
            # model.Mixed_5b.add_module(attention + "t", SELayerT(temporal_length // 8))
            # model.Mixed_5c.add_module(attention + "t", SELayerT(temporal_length // 8))

        # Add temporal-channel-wise SELayer
        elif attention == "SELayerTC":
            model.Mixed_3b.add_module(attention + "t", SELayerT(temporal_length // 2))
            model.Mixed_3c.add_module(attention + "t", SELayerT(temporal_length // 2))
            model.Mixed_4b.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4c.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4d.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4e.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4f.add_module(attention + "t", SELayerT(temporal_length // 4))
            # model.Mixed_5b.add_module(attention + "t", SELayerT(temporal_length // 8))
            # model.Mixed_5c.add_module(attention + "t", SELayerT(temporal_length // 8))

            model.Mixed_3b.add_module(attention + "c", SELayerC(256))
            model.Mixed_3c.add_module(attention + "c", SELayerC(480))
            model.Mixed_4b.add_module(attention + "c", SELayerC(512))
            model.Mixed_4c.add_module(attention + "c", SELayerC(512))
            model.Mixed_4d.add_module(attention + "c", SELayerC(512))
            model.Mixed_4e.add_module(attention + "c", SELayerC(528))
            model.Mixed_4f.add_module(attention + "c", SELayerC(832))
            model.Mixed_5b.add_module(attention + "c", SELayerC(832))
            model.Mixed_5c.add_module(attention + "c", SELayerC(1024))

        else:
            raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))

        self.model = model

    def forward(self, x):
        return self.model(x)


class SEInceptionI3DFlow(nn.Module):
    """Add the several SELayers to I3D for optical flow input."""

    def __init__(self, num_channels, num_classes, attention):
        super(SEInceptionI3DFlow, self).__init__()
        model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)
        temporal_length = 16

        # Add channel-wise SELayer
        if attention in ["SELayerC", "SELayerCoC", "SELayerMC", "SELayerMAC"]:
            se_layer = get_selayer(attention)
            model.Mixed_3b.add_module(attention, se_layer(256))
            model.Mixed_3c.add_module(attention, se_layer(480))
            model.Mixed_4b.add_module(attention, se_layer(512))
            model.Mixed_4c.add_module(attention, se_layer(512))
            model.Mixed_4d.add_module(attention, se_layer(512))
            model.Mixed_4e.add_module(attention, se_layer(528))
            model.Mixed_4f.add_module(attention, se_layer(832))
            model.Mixed_5b.add_module(attention, se_layer(832))
            model.Mixed_5c.add_module(attention, se_layer(1024))

        # Add temporal-wise SELayer
        elif attention == "SELayerT":
            se_layer = get_selayer(attention)
            model.Mixed_3b.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_3c.add_module(attention, se_layer(temporal_length // 4))
            model.Mixed_4b.add_module(attention, se_layer(temporal_length // 8))
            model.Mixed_4c.add_module(attention, se_layer(temporal_length // 8))
            model.Mixed_4d.add_module(attention, se_layer(temporal_length // 8))
            model.Mixed_4e.add_module(attention, se_layer(temporal_length // 8))
            model.Mixed_4f.add_module(attention, se_layer(temporal_length // 8))

        # Add channel-temporal-wise SELayer
        elif attention == "SELayerCT":
            model.Mixed_3b.add_module(attention + "c", SELayerC(256))
            model.Mixed_3c.add_module(attention + "c", SELayerC(480))
            model.Mixed_4b.add_module(attention + "c", SELayerC(512))
            model.Mixed_4c.add_module(attention + "c", SELayerC(512))
            model.Mixed_4d.add_module(attention + "c", SELayerC(512))
            model.Mixed_4e.add_module(attention + "c", SELayerC(528))
            model.Mixed_4f.add_module(attention + "c", SELayerC(832))
            model.Mixed_5b.add_module(attention + "c", SELayerC(832))
            model.Mixed_5c.add_module(attention + "c", SELayerC(1024))

            model.Mixed_3b.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_3c.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4b.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4c.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4d.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4e.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4f.add_module(attention + "t", SELayerT(temporal_length // 8))

        # Add temporal-channel-wise SELayer
        elif attention == "SELayerTC":
            model.Mixed_3b.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_3c.add_module(attention + "t", SELayerT(temporal_length // 4))
            model.Mixed_4b.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4c.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4d.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4e.add_module(attention + "t", SELayerT(temporal_length // 8))
            model.Mixed_4f.add_module(attention + "t", SELayerT(temporal_length // 8))

            model.Mixed_3b.add_module(attention + "c", SELayerC(256))
            model.Mixed_3c.add_module(attention + "c", SELayerC(480))
            model.Mixed_4b.add_module(attention + "c", SELayerC(512))
            model.Mixed_4c.add_module(attention + "c", SELayerC(512))
            model.Mixed_4d.add_module(attention + "c", SELayerC(512))
            model.Mixed_4e.add_module(attention + "c", SELayerC(528))
            model.Mixed_4f.add_module(attention + "c", SELayerC(832))
            model.Mixed_5b.add_module(attention + "c", SELayerC(832))
            model.Mixed_5c.add_module(attention + "c", SELayerC(1024))

        else:
            raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))

        self.model = model

    def forward(self, x):
        return self.model(x)


def se_inception_i3d(name, num_channels, num_classes, attention, pretrained=False, progress=True, rgb=True):
    """Get InceptionI3d module w/o SELayer and pretrained model."""
    if rgb:
        model = SEInceptionI3DRGB(num_channels, num_classes, attention)
    else:
        model = SEInceptionI3DFlow(num_channels, num_classes, attention)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)

        # delete the last layer's parameter and only load the parameters before the last due to different class number.
        # uncomment and change the output size of I3D when using the default classifier in I3D.

        # state_dict.pop("logits.conv3d.weight")
        # state_dict.pop("logits.conv3d.bias")
        # model.load_state_dict(state_dict, strict=False)

        # Create new OrderedDict that add `model.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = "model.{}".format(k)
            new_state_dict[name] = v

        # Load params except SELayer
        model.load_state_dict(new_state_dict, strict=False)
    return model


def se_i3d_joint(rgb_pt, flow_pt, num_classes, attention, pretrained=False, progress=True):
    """Get I3D models with SELayers for different inputs.

    Args:
        rgb_pt (string, optional): the name of pre-trained model for RGB input.
        flow_pt (string, optional): the name of pre-trained model for optical flow input.
        num_classes (int): the class number of dataset.
        attention (string, optional): the name of the SELayer.
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        models (dictionary): A dictionary contains models for RGB and optical flow.
    """

    i3d_rgb = i3d_flow = None
    if rgb_pt is not None:
        i3d_rgb = se_inception_i3d(
            name=rgb_pt,
            num_channels=3,
            num_classes=num_classes,
            attention=attention,
            pretrained=pretrained,
            progress=progress,
            rgb=True,
        )
    if flow_pt is not None:
        i3d_flow = se_inception_i3d(
            name=flow_pt,
            num_channels=2,
            num_classes=num_classes,
            attention=attention,
            pretrained=pretrained,
            progress=progress,
            rgb=False,
        )
    models = {"rgb": i3d_rgb, "flow": i3d_flow}
    return models
