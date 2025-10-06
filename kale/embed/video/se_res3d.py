# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@outlook.com
# =============================================================================

"""Add SELayers to MC3_18, R3D_18, R2plus1D_18"""

from torch.hub import load_state_dict_from_url

from kale.embed.video.res3d import (
    BasicBlock,
    BasicFLowStem,
    BasicStem,
    Conv2Plus1D,
    Conv3DNoTemporal,
    Conv3DSimple,
    R2Plus1dFlowStem,
    R2Plus1dStem,
    VideoResNet,
)
from kale.embed.video.selayer import get_selayer

model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


def _se_video_resnet_rgb(arch, attention, pretrained=False, progress=True, **kwargs):
    """Add the several SELayers to MC3_18, R3D_18, R2plus1D_18 for RGB input.

    Args:
        arch (string): the name of basic architecture. (Options: ["r3d_18", "mc3_18" and "r2plus1d_18"])
        attention (string): the name of the SELayer.
            (Options: ["SELayerC", "SELayerT", "SELayerMC", "SELayerMAC", "SELayerCT", and "SELayerTC"])
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        model (VideoResNet): 3D convolution-based model with SELayers.
    """
    model = VideoResNet(**kwargs)
    temporal_length = 16

    # Add channel-wise SELayer
    if attention in ["SELayerC", "SELayerMC", "SELayerMAC"]:
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(64))
        model.layer1._modules["1"].add_module(attention, se_layer(64))
        model.layer2._modules["0"].add_module(attention, se_layer(128))
        model.layer2._modules["1"].add_module(attention, se_layer(128))
        model.layer3._modules["0"].add_module(attention, se_layer(256))
        model.layer3._modules["1"].add_module(attention, se_layer(256))
        model.layer4._modules["0"].add_module(attention, se_layer(512))
        model.layer4._modules["1"].add_module(attention, se_layer(512))

    # Add temporal-wise SELayer
    elif attention == "SELayerT":
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(temporal_length))
        model.layer1._modules["1"].add_module(attention, se_layer(temporal_length))
        model.layer2._modules["0"].add_module(attention, se_layer(temporal_length // 2))
        model.layer2._modules["1"].add_module(attention, se_layer(temporal_length // 2))
        model.layer3._modules["0"].add_module(attention, se_layer(temporal_length // 4))
        model.layer3._modules["1"].add_module(attention, se_layer(temporal_length // 4))

    # Add channel-temporal-wise SELayer
    elif attention == "SELayerCT":
        se_layer = get_selayer(attention)
        layer_schedule = [
            ("layer1", "0", 64, temporal_length),
            ("layer1", "1", 64, temporal_length),
            ("layer2", "0", 128, temporal_length // 2),
            ("layer2", "1", 128, temporal_length // 2),
            ("layer3", "0", 256, temporal_length // 4),
            ("layer3", "1", 256, temporal_length // 4),
            ("layer4", "0", 512, max(1, temporal_length // 8)),
            ("layer4", "1", 512, max(1, temporal_length // 8)),
        ]
        for layer_name, block_idx, channels, temporal in layer_schedule:
            getattr(model, layer_name)._modules[block_idx].add_module(attention, se_layer(channels, temporal))

    # Add temporal-channel-wise SELayer
    elif attention == "SELayerTC":
        se_layer = get_selayer(attention)
        layer_schedule = [
            ("layer1", "0", 64, temporal_length),
            ("layer1", "1", 64, temporal_length),
            ("layer2", "0", 128, temporal_length // 2),
            ("layer2", "1", 128, temporal_length // 2),
            ("layer3", "0", 256, temporal_length // 4),
            ("layer3", "1", 256, temporal_length // 4),
            ("layer4", "0", 512, max(1, temporal_length // 8)),
            ("layer4", "1", 512, max(1, temporal_length // 8)),
        ]
        for layer_name, block_idx, channels, temporal in layer_schedule:
            getattr(model, layer_name)._modules[block_idx].add_module(attention, se_layer(channels, temporal))

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def _se_video_resnet_flow(arch, attention, pretrained=False, progress=True, **kwargs):
    """Add the several SELayers to MC3_18, R3D_18, R2plus1D_18 for optical flow input."""
    model = VideoResNet(**kwargs)
    temporal_length = 16

    # Add channel-wise SELayer
    if attention in ["SELayerC", "SELayerMC", "SELayerMAC"]:
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(64))
        model.layer1._modules["1"].add_module(attention, se_layer(64))
        model.layer2._modules["0"].add_module(attention, se_layer(128))
        model.layer2._modules["1"].add_module(attention, se_layer(128))
        model.layer3._modules["0"].add_module(attention, se_layer(256))
        model.layer3._modules["1"].add_module(attention, se_layer(256))
        model.layer4._modules["0"].add_module(attention, se_layer(512))
        model.layer4._modules["1"].add_module(attention, se_layer(512))

    # Add temporal-wise SELayer
    elif attention == "SELayerT":
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(temporal_length // 2))
        model.layer1._modules["1"].add_module(attention, se_layer(temporal_length // 2))
        model.layer2._modules["0"].add_module(attention, se_layer(temporal_length // 4))
        model.layer2._modules["1"].add_module(attention, se_layer(temporal_length // 4))

    # Add channel-temporal-wise SELayer
    elif attention == "SELayerCT":
        se_layer = get_selayer(attention)
        layer_schedule = [
            ("layer1", "0", 64, temporal_length // 2),
            ("layer1", "1", 64, temporal_length // 2),
            ("layer2", "0", 128, temporal_length // 4),
            ("layer2", "1", 128, temporal_length // 4),
            ("layer3", "0", 256, max(1, temporal_length // 8)),
            ("layer3", "1", 256, max(1, temporal_length // 8)),
            ("layer4", "0", 512, max(1, temporal_length // 16)),
            ("layer4", "1", 512, max(1, temporal_length // 16)),
        ]
        for layer_name, block_idx, channels, temporal in layer_schedule:
            getattr(model, layer_name)._modules[block_idx].add_module(attention, se_layer(channels, temporal))

    # Add temporal-channel-wise SELayer
    elif attention == "SELayerTC":
        se_layer = get_selayer(attention)
        layer_schedule = [
            ("layer1", "0", 64, temporal_length),
            ("layer1", "1", 64, temporal_length),
            ("layer2", "0", 128, temporal_length // 2),
            ("layer2", "1", 128, temporal_length // 2),
            ("layer3", "0", 256, temporal_length // 4),
            ("layer3", "1", 256, temporal_length // 4),
            ("layer4", "0", 512, max(1, temporal_length // 8)),
            ("layer4", "1", 512, max(1, temporal_length // 8)),
        ]
        for layer_name, block_idx, channels, temporal in layer_schedule:
            getattr(model, layer_name)._modules[block_idx].add_module(attention, se_layer(channels, temporal))

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop("stem.0.weight")
        model.load_state_dict(state_dict, strict=False)
    return model


def se_r3d_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "r3d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def se_r3d_18_flow(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_flow(
        "r3d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicFLowStem,
        **kwargs,
    )


def se_mc3_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "mc3_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def se_mc3_18_flow(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_flow(
        "mc3_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicFLowStem,
        **kwargs,
    )


def se_r2plus1d_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "r2plus1d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs,
    )


def se_r2plus1d_18_flow(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_flow(
        "r2plus1d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dFlowStem,
        **kwargs,
    )


def se_r3d(attention, rgb=False, flow=False, pretrained=False, progress=True):
    """Get R3D_18 models with SELayers for different inputs.

    Args:
        attention (string): the name of the SELayer.
        rgb (bool): choose if RGB model is needed. (Default: False)
        flow (bool): choose if optical flow model is needed. (Default: False)
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        models (dictionary): A dictionary contains models for RGB and optical flow.
    """

    r3d_rgb = r3d_flow = None
    if rgb:
        r3d_rgb = se_r3d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    if flow:
        r3d_flow = se_r3d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {"rgb": r3d_rgb, "flow": r3d_flow}
    return models


def se_mc3(attention, rgb=False, flow=False, pretrained=False, progress=True):
    """Get MC3_18 models with SELayers for different inputs."""
    mc3_rgb = mc3_flow = None
    if rgb:
        mc3_rgb = se_mc3_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    if flow:
        mc3_flow = se_mc3_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {"rgb": mc3_rgb, "flow": mc3_flow}
    return models


def se_r2plus1d(attention, rgb=False, flow=False, pretrained=False, progress=True):
    """Get R2+1D_18 models with SELayers for different inputs."""
    r2plus1d_rgb = r2plus1d_flow = None
    if rgb:
        r2plus1d_rgb = se_r2plus1d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    if flow:
        r2plus1d_flow = se_r2plus1d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {"rgb": r2plus1d_rgb, "flow": r2plus1d_flow}
    return models
