"""
Define Inflated 3D ConvNets(I3D) on Action Recognition from https://ieeexplore.ieee.org/document/8099985
Created by Xianyuan Liu from modifying https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py and
https://github.com/deepmind/kinetics-i3d/blob/master/i3d.py
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from kale.embed.video_i3d import InceptionI3d

__all__ = ['se_i3d_joint', 'SEInceptionI3DRGB', 'SEInceptionI3DFlow']

model_urls = {
    "rgb_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_imagenet.pt",
    "flow_imagenet": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_imagenet.pt",
    "rgb_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/rgb_charades.pt",
    "flow_charades": "https://github.com/XianyuanLiu/pytorch-i3d/raw/master/models/flow_charades.pt",
}


class SELayerC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
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
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerT(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayerT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()
        output = x.transpose(1, 2).contiguous()
        y = self.avg_pool(output).view(b, t)
        y = self.fc(y).view(b, t, 1, 1, 1)
        y = y.transpose(1, 2).contiguous()
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerCoC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerCoC, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=channel // reduction)
        # self.conv3 = nn.Conv2d(
        #     in_channels=channel // reduction,
        #     out_channels=channel // reduction,
        #     kernel_size=3,
        #     padding=1,
        #     groups=channel // reduction,
        #     bias=False)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv3d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm3d(num_features=channel)

    def forward(self, x):
        b, c, t, _, _ = x.size()  # n, c, t, h, w
        y = self.conv1(x)  # n, c/r, t, h, w
        y = self.bn1(y)  # n, c/r, t, h, w
        y = self.avg_pool(y)  # n, c/r, 1, 1, 1
        y = self.conv2(y)  # n, c, 1, 1, 1
        y = self.bn2(y)  # n, c, 1, 1, 1
        y = self.sigmoid(y)  # n, c, 1, 1, 1
        # out = x * y.expand_as(x)  # n, c, t, h, w
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerMC, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SELayerMAC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayerMAC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 2),
            bias=False
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y = torch.cat((y_avg, y_max), dim=2).squeeze().unsqueeze(dim=1)
        y = self.conv(y).squeeze()
        y = self.fc(y).view(b, c, 1, 1, 1)
        # out = x * y.expand_as(x)
        y = y - 0.5
        out = x + x * y.expand_as(x)
        return out


class SEInceptionI3DRGB(nn.Module):

    def __init__(self, num_channels, num_classes, attention):
        super(SEInceptionI3DRGB, self).__init__()
        model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)
        n = 16
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

        elif attention == "SELayerT":
            model.Mixed_3b.add_module("SELayerT", SELayerT(n//2))
            model.Mixed_3c.add_module("SELayerT", SELayerT(n//2))
            model.Mixed_4b.add_module("SELayerT", SELayerT(n//4))
            model.Mixed_4c.add_module("SELayerT", SELayerT(n//4))
            model.Mixed_4d.add_module("SELayerT", SELayerT(n//4))
            model.Mixed_4e.add_module("SELayerT", SELayerT(n//4))
            model.Mixed_4f.add_module("SELayerT", SELayerT(n//4))
            model.Mixed_5b.add_module("SELayerT", SELayerT(n//8))
            model.Mixed_5c.add_module("SELayerT", SELayerT(n//8))

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

        elif attention == "SELayerCT":
            model.Mixed_3b.add_module("SELayerC", SELayerC(256))
            model.Mixed_3c.add_module("SELayerC", SELayerC(480))
            model.Mixed_4b.add_module("SELayerC", SELayerC(512))
            model.Mixed_4c.add_module("SELayerC", SELayerC(512))
            model.Mixed_4d.add_module("SELayerC", SELayerC(512))
            model.Mixed_4e.add_module("SELayerC", SELayerC(528))
            model.Mixed_4f.add_module("SELayerC", SELayerC(832))
            model.Mixed_5b.add_module("SELayerC", SELayerC(832))
            model.Mixed_5c.add_module("SELayerC", SELayerC(1024))

            model.Mixed_3b.add_module("SELayerT", SELayerT(n // 2))
            model.Mixed_3c.add_module("SELayerT", SELayerT(n // 2))
            model.Mixed_4b.add_module("SELayerT", SELayerT(n // 4))
            model.Mixed_4c.add_module("SELayerT", SELayerT(n // 4))
            model.Mixed_4d.add_module("SELayerT", SELayerT(n // 4))
            model.Mixed_4e.add_module("SELayerT", SELayerT(n // 4))
            model.Mixed_4f.add_module("SELayerT", SELayerT(n // 4))
            model.Mixed_5b.add_module("SELayerT", SELayerT(n // 8))
            model.Mixed_5c.add_module("SELayerT", SELayerT(n // 8))

        self.model = model

    def forward(self, x):
        return self.model(x)


class SEInceptionI3DFlow(nn.Module):

    def __init__(self, num_channels, num_classes, attention):
        super(SEInceptionI3DFlow, self).__init__()
        model = InceptionI3d(in_channels=num_channels, num_classes=num_classes)
        n = 16
        # if attention == "SELayerC":
        #     model.Mixed_3b.add_module("SELayerC", SELayerC(256))
        #     model.Mixed_3c.add_module("SELayerC", SELayerC(480))
        #     model.Mixed_4b.add_module("SELayerC", SELayerC(512))
        #     model.Mixed_4c.add_module("SELayerC", SELayerC(512))
        #     model.Mixed_4d.add_module("SELayerC", SELayerC(512))
        #     model.Mixed_4e.add_module("SELayerC", SELayerC(528))
        #     model.Mixed_4f.add_module("SELayerC", SELayerC(832))
        #     model.Mixed_5b.add_module("SELayerC", SELayerC(832))
        #     model.Mixed_5c.add_module("SELayerC", SELayerC(1024))
        #
        # elif attention == "SELayerT":
        #     model.Mixed_3b.add_module("SELayerT", SELayerT(n // 4))
        #     model.Mixed_3c.add_module("SELayerT", SELayerT(n // 4))
        #     model.Mixed_4b.add_module("SELayerT", SELayerT(n // 8))
        #     model.Mixed_4c.add_module("SELayerT", SELayerT(n // 8))
        #     model.Mixed_4d.add_module("SELayerT", SELayerT(n // 8))
        #     model.Mixed_4e.add_module("SELayerT", SELayerT(n // 8))
        #     model.Mixed_4f.add_module("SELayerT", SELayerT(n // 8))

        self.model = model

    def forward(self, x):
        return self.model(x)


def se_inception_i3d(name, num_channels, num_classes, attention, pretrained=False, progress=True, rgb=True):
    """Get InceptionI3d module w/o pretrained model."""
    if rgb:
        model = SEInceptionI3DRGB(num_channels, num_classes, attention)
    else:
        model = SEInceptionI3DFlow(num_channels, num_classes, attention)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[name], progress=progress)
        # delete logits.conv3d parameters due to different class number.
        state_dict.pop("logits.conv3d.weight")
        state_dict.pop("logits.conv3d.bias")
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
    """Get I3D models."""
    i3d_rgb = i3d_flow = None
    if rgb_pt is not None and flow_pt is None:
        i3d_rgb = se_inception_i3d(
            name=rgb_pt, num_channels=3, num_classes=num_classes, attention=attention, pretrained=pretrained, progress=progress, rgb=True
        )
    elif rgb_pt is None and flow_pt is not None:
        i3d_flow = se_inception_i3d(
            name=flow_pt, num_channels=2, num_classes=num_classes, attention=attention, pretrained=pretrained, progress=progress, rgb=False
        )
    elif rgb_pt is not None and flow_pt is not None:
        i3d_rgb = se_inception_i3d(
            name=rgb_pt, num_channels=3, num_classes=num_classes, attention=attention, pretrained=pretrained, progress=progress, rgb=True
        )
        i3d_flow = se_inception_i3d(
            name=flow_pt, num_channels=2, num_classes=num_classes, attention=attention, pretrained=pretrained, progress=progress, rgb=False
        )
    models = {'rgb': i3d_rgb, 'flow': i3d_flow}
    return models
