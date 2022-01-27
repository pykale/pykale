import pytest
import torch

from kale.embed.video_feature_extractor import get_extractor_feat, get_extractor_video

# ATTENTION = ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerCT", "SELayerTC", "SELayerMAC", "None"]

NUM_CLASSES = {"verb": 6, "noun": 7}


@pytest.mark.parametrize("model_name", ["I3D", "R3D_18", "R2PLUS1D_18", "MC3_18"])
@pytest.mark.parametrize("image_modality", ["rgb", "flow", "joint"])
@pytest.mark.parametrize("attention", ["SELayerC", "SELayerT", "SELayerCT", "SELayerTC", "None"])
def test_get_video_feat_extractor(model_name, image_modality, attention):
    feature_network, class_feature_dim, domain_feature_dim = get_extractor_video(
        model_name, image_modality, attention, NUM_CLASSES
    )

    assert isinstance(feature_network, dict)

    if image_modality == "joint":
        assert isinstance(feature_network["rgb"], torch.nn.Module)
        assert isinstance(feature_network["flow"], torch.nn.Module)
    elif image_modality == "rgb":
        assert isinstance(feature_network["rgb"], torch.nn.Module)
        assert feature_network["flow"] is None
    else:
        assert feature_network["rgb"] is None
        assert isinstance(feature_network["flow"], torch.nn.Module)

    if model_name == "I3D":
        assert domain_feature_dim == 1024
        assert class_feature_dim == 2048 if image_modality == "joint" else class_feature_dim == 1024

    else:
        assert domain_feature_dim == 512
        assert class_feature_dim == 1024 if image_modality == "joint" else class_feature_dim == 512


@pytest.mark.parametrize("image_modality", ["rgb", "flow", "audio", "joint", "all"])
def test_get_vector_feat_extractor(image_modality):
    feature_network, class_feature_dim, domain_feature_dim = get_extractor_feat("test", image_modality, 32, 5, 2)

    assert isinstance(feature_network, dict)

    if image_modality == "rgb":
        assert isinstance(feature_network["rgb"], torch.nn.Module)
        assert feature_network["flow"] is None
        assert feature_network["audio"] is None
    elif image_modality == "flow":
        assert isinstance(feature_network["flow"], torch.nn.Module)
        assert feature_network["rgb"] is None
        assert feature_network["audio"] is None
    elif image_modality == "audio":
        assert isinstance(feature_network["audio"], torch.nn.Module)
        assert feature_network["flow"] is None
        assert feature_network["rgb"] is None
    elif image_modality == "joint":
        assert isinstance(feature_network["rgb"], torch.nn.Module)
        assert isinstance(feature_network["flow"], torch.nn.Module)
        assert feature_network["audio"] is None
    elif image_modality == "all":
        assert isinstance(feature_network["rgb"], torch.nn.Module)
        assert isinstance(feature_network["flow"], torch.nn.Module)
        assert isinstance(feature_network["audio"], torch.nn.Module)

    assert domain_feature_dim == 10
    if image_modality in ["rgb", "flow", "audio"]:
        assert class_feature_dim == 10
    elif image_modality == "joint":
        assert class_feature_dim == 20
    elif image_modality == "all":
        assert class_feature_dim == 30
