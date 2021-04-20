# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Add SELayers to I3D"""

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from kale.embed.video_i3d import InceptionI3d
from kale.embed.video_selayer import SELayerC, SELayerCoC, SELayerMAC, SELayerMC, SELayerT

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
        if attention == "SELayerC":
            model.Mixed_3b.add_module("SELayerC", SELayerC(256))
            model.Mixed_3c.add_module("SELayerC", SELayerC(480))
            model.Mixed_4b.add_module("SELayerC", SELayerC(512))
            model.Mixed_4c.add_module("SELayerC", SELayerC(512))
            model.Mixed_4d.add_module("SELayerC", SELayerC(512))
            model.Mixed_4e.add_module("SELayerC", SELayerC(528))
            model.Mixed_4f.add_module("SELayerC", SELayerC(832))
            model.Mixed_5b.add_module("SELayerC", SELayerC(832))
            model.Mixed_5c.add_module("SELayerC", SELayerC(1024))

        # Add temporal-wise SELayer
        elif attention == "SELayerT":
            model.Mixed_3b.add_module("SELayerT", SELayerT(temporal_length // 2))
            model.Mixed_3c.add_module("SELayerT", SELayerT(temporal_length // 2))
            model.Mixed_4b.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_4c.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_4d.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_4e.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_4f.add_module("SELayerT", SELayerT(temporal_length // 4))
            # model.Mixed_5b.add_module("SELayerT", SELayerT(temporal_length//8))
            # model.Mixed_5c.add_module("SELayerT", SELayerT(temporal_length//8))

        # Add convolution-based channel-wise SELayer
        elif attention == "SELayerCoC":
            model.Mixed_3b.add_module("SELayerCoC", SELayerCoC(256))
            model.Mixed_3c.add_module("SELayerCoC", SELayerCoC(480))
            model.Mixed_4b.add_module("SELayerCoC", SELayerCoC(512))
            model.Mixed_4c.add_module("SELayerCoC", SELayerCoC(512))
            model.Mixed_4d.add_module("SELayerCoC", SELayerCoC(512))
            model.Mixed_4e.add_module("SELayerCoC", SELayerCoC(528))
            model.Mixed_4f.add_module("SELayerCoC", SELayerCoC(832))
            model.Mixed_5b.add_module("SELayerCoC", SELayerCoC(832))
            model.Mixed_5c.add_module("SELayerCoC", SELayerCoC(1024))

        # Add max-pooling-based channel-wise SELayer
        elif attention == "SELayerMC":
            model.Mixed_3b.add_module("SELayerMC", SELayerMC(256))
            model.Mixed_3c.add_module("SELayerMC", SELayerMC(480))
            model.Mixed_4b.add_module("SELayerMC", SELayerMC(512))
            model.Mixed_4c.add_module("SELayerMC", SELayerMC(512))
            model.Mixed_4d.add_module("SELayerMC", SELayerMC(512))
            model.Mixed_4e.add_module("SELayerMC", SELayerMC(528))
            model.Mixed_4f.add_module("SELayerMC", SELayerMC(832))
            model.Mixed_5b.add_module("SELayerMC", SELayerMC(832))
            model.Mixed_5c.add_module("SELayerMC", SELayerMC(1024))

        # Add multi-pooling-based channel-wise SELayer
        elif attention == "SELayerMAC":
            model.Mixed_3b.add_module("SELayerMAC", SELayerMAC(256))
            model.Mixed_3c.add_module("SELayerMAC", SELayerMAC(480))
            model.Mixed_4b.add_module("SELayerMAC", SELayerMAC(512))
            model.Mixed_4c.add_module("SELayerMAC", SELayerMAC(512))
            model.Mixed_4d.add_module("SELayerMAC", SELayerMAC(512))
            model.Mixed_4e.add_module("SELayerMAC", SELayerMAC(528))
            model.Mixed_4f.add_module("SELayerMAC", SELayerMAC(832))
            model.Mixed_5b.add_module("SELayerMAC", SELayerMAC(832))
            model.Mixed_5c.add_module("SELayerMAC", SELayerMAC(1024))

        # Add channel-temporal-wise SELayer
        elif attention == "SELayerCT":
            model.Mixed_3b.add_module("SELayerCTc", SELayerC(256))
            model.Mixed_3c.add_module("SELayerCTc", SELayerC(480))
            model.Mixed_4b.add_module("SELayerCTc", SELayerC(512))
            model.Mixed_4c.add_module("SELayerCTc", SELayerC(512))
            model.Mixed_4d.add_module("SELayerCTc", SELayerC(512))
            model.Mixed_4e.add_module("SELayerCTc", SELayerC(528))
            model.Mixed_4f.add_module("SELayerCTc", SELayerC(832))
            model.Mixed_5b.add_module("SELayerCTc", SELayerC(832))
            model.Mixed_5c.add_module("SELayerCTc", SELayerC(1024))

            model.Mixed_3b.add_module("SELayerCTt", SELayerT(temporal_length // 2))
            model.Mixed_3c.add_module("SELayerCTt", SELayerT(temporal_length // 2))
            model.Mixed_4b.add_module("SELayerCTt", SELayerT(temporal_length // 4))
            model.Mixed_4c.add_module("SELayerCTt", SELayerT(temporal_length // 4))
            model.Mixed_4d.add_module("SELayerCTt", SELayerT(temporal_length // 4))
            model.Mixed_4e.add_module("SELayerCTt", SELayerT(temporal_length // 4))
            model.Mixed_4f.add_module("SELayerCTt", SELayerT(temporal_length // 4))
            # model.Mixed_5b.add_module("SELayerCTt", SELayerT(temporal_length // 8))
            # model.Mixed_5c.add_module("SELayerCTt", SELayerT(temporal_length // 8))

        # Add temporal-channel-wise SELayer
        elif attention == "SELayerTC":
            model.Mixed_3b.add_module("SELayerTCt", SELayerT(temporal_length // 2))
            model.Mixed_3c.add_module("SELayerTCt", SELayerT(temporal_length // 2))
            model.Mixed_4b.add_module("SELayerTCt", SELayerT(temporal_length // 4))
            model.Mixed_4c.add_module("SELayerTCt", SELayerT(temporal_length // 4))
            model.Mixed_4d.add_module("SELayerTCt", SELayerT(temporal_length // 4))
            model.Mixed_4e.add_module("SELayerTCt", SELayerT(temporal_length // 4))
            model.Mixed_4f.add_module("SELayerTCt", SELayerT(temporal_length // 4))
            model.Mixed_5b.add_module("SELayerTCt", SELayerT(temporal_length // 8))
            model.Mixed_5c.add_module("SELayerTCt", SELayerT(temporal_length // 8))

            model.Mixed_3b.add_module("SELayerTCc", SELayerC(256))
            model.Mixed_3c.add_module("SELayerTCc", SELayerC(480))
            model.Mixed_4b.add_module("SELayerTCc", SELayerC(512))
            model.Mixed_4c.add_module("SELayerTCc", SELayerC(512))
            model.Mixed_4d.add_module("SELayerTCc", SELayerC(512))
            model.Mixed_4e.add_module("SELayerTCc", SELayerC(528))
            model.Mixed_4f.add_module("SELayerTCc", SELayerC(832))
            model.Mixed_5b.add_module("SELayerTCc", SELayerC(832))
            model.Mixed_5c.add_module("SELayerTCc", SELayerC(1024))

        self.model = model

    def forward(self, x):
        return self.model(x)


class SEInceptionI3DFlow(nn.Module):
    """
    Add the several SELayers to I3D for optical flow input.
    """

    def __init__(self, num_channels, num_classes, attention):
        super(SEInceptionI3DFlow, self).__init__()
        model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)
        temporal_length = 16

        # Add channel-wise SELayer
        if attention == "SELayerC":
            model.Mixed_3b.add_module("SELayerC", SELayerC(256))
            model.Mixed_3c.add_module("SELayerC", SELayerC(480))
            model.Mixed_4b.add_module("SELayerC", SELayerC(512))
            model.Mixed_4c.add_module("SELayerC", SELayerC(512))
            model.Mixed_4d.add_module("SELayerC", SELayerC(512))
            model.Mixed_4e.add_module("SELayerC", SELayerC(528))
            model.Mixed_4f.add_module("SELayerC", SELayerC(832))
            model.Mixed_5b.add_module("SELayerC", SELayerC(832))
            model.Mixed_5c.add_module("SELayerC", SELayerC(1024))

        # Add temporal-wise SELayer
        elif attention == "SELayerT":
            model.Mixed_3b.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_3c.add_module("SELayerT", SELayerT(temporal_length // 4))
            model.Mixed_4b.add_module("SELayerT", SELayerT(temporal_length // 8))
            model.Mixed_4c.add_module("SELayerT", SELayerT(temporal_length // 8))
            model.Mixed_4d.add_module("SELayerT", SELayerT(temporal_length // 8))
            model.Mixed_4e.add_module("SELayerT", SELayerT(temporal_length // 8))
            model.Mixed_4f.add_module("SELayerT", SELayerT(temporal_length // 8))

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

        # delete logits.conv3d parameters due to different class number.
        # state_dict.pop("logits.conv3d.weight")
        # state_dict.pop("logits.conv3d.bias")

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
        rgb_pt (string): the name of pre-trained model for RGB input.
        flow_pt (string): the name of pre-trained model for optical flow input.
        num_classes (int): the class number of dataset.
        attention (string): the name of the SELayer.
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
