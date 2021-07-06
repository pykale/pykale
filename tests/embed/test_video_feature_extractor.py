import pytest
import torch

from kale.embed.video_feature_extractor import get_video_feat_extractor

MODEL_NAME = ["I3D", "R3D_18", "R2PLUS1D_18", "MC3_18"]
IMAGE_MODALITY = ["rgb", "flow", "joint"]
# ATTENTION = ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerCT", "SELayerTC", "SELayerMAC", "None"]
ATTENTION = ["SELayerC", "SELayerT", "SELayerCT", "SELayerTC", "None"]
# NUM_CLASSES = [6, 7, 8]
NUM_CLASSES = [6]


@pytest.mark.parametrize("model_name", MODEL_NAME)
@pytest.mark.parametrize("image_modality", IMAGE_MODALITY)
@pytest.mark.parametrize("attention", ATTENTION)
@pytest.mark.parametrize("num_classes", NUM_CLASSES)
def test_get_video_feat_extractor(model_name, image_modality, attention, num_classes):
    feature_network, class_feature_dim, domain_feature_dim = get_video_feat_extractor(
        model_name, image_modality, attention, num_classes
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
