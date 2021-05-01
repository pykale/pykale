"""Reference: https://github.com/pytorch/vision/blob/master/test/test_models.py#L257"""

import pytest
import torch

from kale.embed.video_i3d import InceptionI3d
from kale.embed.video_res3d import BasicBlock, BasicStem, Bottleneck, Conv3DSimple, VideoResNet
from kale.embed.video_se_i3d import SEInceptionI3DFlow, SEInceptionI3DRGB
from kale.embed.video_se_res3d import se_r3d_18_flow, se_r3d_18_rgb
from kale.utils.seed import set_seed

set_seed(36)
# The default input shape is batch_size * num_channel * frame_per_segment * height * weight.
# In our experiment, the height and weight are both 224. To avoid to allocate too many memory,
# height and weight are set as 112. It won't influence the model test.
# Differences between inputs of RGB and flow is channel number and frame_per_segment.
INPUT_BATCH_RGB = torch.randn(2, 3, 16, 112, 112)
INPUT_BATCH_FLOW = torch.randn(2, 2, 8, 112, 112)
SE_LAYERS = ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerMAC", "SELayerCT", "SELayerTC"]


def test_i3d_shapes():
    # test InceptionI3D
    i3d = InceptionI3d()
    i3d.eval()
    output_batch = i3d(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 1024, 1)

    # test InceptionI3D.extract_features
    output_batch = i3d.extract_features(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 1024, 1, 1, 1)


def test_videoresnet_basicblock_shapes():
    # test VideoResNet with BasicBlock
    resnet = VideoResNet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem,)
    resnet.eval()
    output_batch = resnet(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 512)


def test_videoresenet_bottleneck_shapes():
    # test VideoResNet with Bottleneck
    resnet_bottleneck = VideoResNet(
        block=Bottleneck, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2], stem=BasicStem,
    )
    resnet_bottleneck.eval()
    output_batch = resnet_bottleneck(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 2048)


@pytest.mark.parametrize("se_layers", SE_LAYERS)
def test_i3d_selayer_shapes(se_layers):
    # test I3D with SELayers
    se_rgb = SEInceptionI3DRGB(3, 8, se_layers)
    se_flow = SEInceptionI3DFlow(2, 8, se_layers)
    se_rgb.eval()
    se_flow.eval()

    output_batch = se_rgb(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 1024, 1)
    output_batch = se_flow(INPUT_BATCH_FLOW)
    assert output_batch.size() == (2, 1024, 1)


@pytest.mark.parametrize("se_layers", SE_LAYERS)
def test_r3d_selayer_shapes(se_layers):
    # test R3D with SELayers
    se_rgb = se_r3d_18_rgb(attention=se_layers, pretrained=False)
    se_flow = se_r3d_18_flow(attention=se_layers, pretrained=False)
    se_rgb.eval()
    se_flow.eval()

    output_batch = se_rgb(INPUT_BATCH_RGB)
    assert output_batch.size() == (2, 512)
    output_batch = se_flow(INPUT_BATCH_FLOW)
    assert output_batch.size() == (2, 512)
