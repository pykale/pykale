import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from kale.embed.video_res3d import \
    Conv3DSimple, Conv2Plus1D, Conv3DNoTemporal, \
    BasicBlock, BasicStem, BasicFLowStem, R2Plus1dStem, R2Plus1dFlowStem, VideoResNet
from kale.embed.video_se_cnn import SELayerC, SELayerT, SELayerCoC, SELayerMC, SELayerMAC

model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


def _se_video_resnet(arch, attention, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if attention == "SELayerC":
        model.layer1.modules["0"].add_module("SELayerC", SELayerC(64))
        model.layer1.modules["1"].add_module("SELayerC", SELayerC(64))
        model.layer2.modules["0"].add_module("SELayerC", SELayerC(128))
        model.layer2.modules["1"].add_module("SELayerC", SELayerC(128))
        model.layer3.modules["0"].add_module("SELayerC", SELayerC(256))
        model.layer3.modules["1"].add_module("SELayerC", SELayerC(256))
        model.layer4.modules["0"].add_module("SELayerC", SELayerC(512))
        model.layer4.modules["1"].add_module("SELayerC", SELayerC(512))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def _se_video_resnet_flow(arch, attention, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    # if attention == "SELayerC":
    #     model.layer1.modules["0"].add_module("SELayerC", SELayerC(64))
    #     model.layer1.modules["1"].add_module("SELayerC", SELayerC(64))
    #     model.layer2.modules["0"].add_module("SELayerC", SELayerC(128))
    #     model.layer2.modules["1"].add_module("SELayerC", SELayerC(128))
    #     model.layer3.modules["0"].add_module("SELayerC", SELayerC(256))
    #     model.layer3.modules["1"].add_module("SELayerC", SELayerC(256))
    #     model.layer4.modules["0"].add_module("SELayerC", SELayerC(512))
    #     model.layer4.modules["1"].add_module("SELayerC", SELayerC(512))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop("stem.0.weight")
        model.load_state_dict(state_dict, strict=False)
    return model


def se_r3d_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet(
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
    return _se_video_resnet(
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
    return _se_video_resnet(
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
    return _se_video_resnet(
        'r2plus1d_18',
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
    return _se_video_resnet(
        'r2plus1d_18',
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
    """Get R3D_18 models."""
    r3d_rgb = r3d_flow = None
    if rgb and not flow:
        r3d_rgb = se_r3d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    elif not rgb and flow:
        r3d_flow = se_r3d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    elif rgb and flow:
        r3d_rgb = se_r3d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
        r3d_flow = se_r3d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {'rgb': r3d_rgb, 'flow': r3d_flow}
    return models


def se_mc3(attention, rgb=False, flow=False, pretrained=False, progress=True):
    """Get R3D_18 models."""
    mc3_rgb = mc3_flow = None
    if rgb and not flow:
        mc3_rgb = se_mc3_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    elif not rgb and flow:
        mc3_flow = se_mc3_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    elif rgb and flow:
        mc3_rgb = se_mc3_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
        mc3_flow = se_mc3_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {'rgb': mc3_rgb, 'flow': mc3_flow}
    return models


def se_r2plus1d(attention, rgb=False, flow=False, pretrained=False, progress=True):
    """Get R3D_18 models."""
    r2plus1d_rgb = r2plus1d_flow = None
    if rgb and not flow:
        r2plus1d_rgb = se_r2plus1d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
    elif not rgb and flow:
        r2plus1d_flow = se_r2plus1d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    elif rgb and flow:
        r2plus1d_rgb = se_r2plus1d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
        r2plus1d_flow = se_r2plus1d_18_flow(attention=attention, pretrained=pretrained, progress=progress)
    models = {'rgb': r2plus1d_rgb, 'flow': r2plus1d_flow}
    return models
