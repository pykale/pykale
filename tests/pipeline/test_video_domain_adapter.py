import os
from pathlib import Path

import pytest
from yacs.config import CfgNode as CN

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.predict.class_domain_nets import ClassNetVideo
from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed
from tests.helpers.boring_model import VideoBoringModel, VideoVectorBoringModel
from tests.helpers.pipe_test_helper import DASetupHelper, ModelTestHelper

SOURCES = [
    "ADL;adl_P_11_train.pkl;adl_P_11_test.pkl",
    "EPIC100;EPIC_100_uda_source_test_timestamps.pkl;EPIC_100_uda_source_train.pkl",
]
TARGETS = [
    "ADL;adl_P_11_train.pkl;adl_P_11_test.pkl",
    "EPIC100;EPIC_100_uda_source_test_timestamps.pkl;EPIC_100_uda_source_train.pkl",
]
IMAGE_MODALITY = ["rgb", "flow", "joint"]
# IMAGE_MODALITY = ["rgb"]
IMAGE_MODALITY_FEAT = ["rgb", "flow", "audio", "all"]
# CLASS_TYPE = ["verb", "verb+noun"]
CLASS_TYPE = ["verb+noun"]
DA_METHODS = ["DANN", "CDAN", "CDAN-E", "WDGRL", "DAN", "JAN", "Source"]
# DA_METHODS = ["JAN"]
WEIGHT_TYPE = "natural"
DATASIZE_TYPE = "max"
VALID_RATIO = 0.1
seed = 36
set_seed(seed)

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DAN = CN()
    cfg.DATASET.ROOT = root_dir + "/" + download_path + "/video_test_data/"
    cfg.DATASET.FRAMES_PER_SEGMENT = 16
    cfg.DATASET.WEIGHT_TYPE = WEIGHT_TYPE
    cfg.DATASET.SIZE_TYPE = DATASIZE_TYPE
    cfg.DAN.USERANDOM = False
    yield cfg


@pytest.fixture(scope="module")
def testing_training_cfg():
    config_params = {
        "train_params": {
            "adapt_lambda": True,
            "adapt_lr": True,
            "lambda_init": 1,
            "nb_adapt_epochs": 2,
            "nb_init_epochs": 1,
            "init_lr": 0.001,
            "batch_size": 2,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


@pytest.mark.parametrize("image_modality", IMAGE_MODALITY)
@pytest.mark.parametrize("da_method", DA_METHODS)
def test_video_domain_adapter(image_modality, da_method, testing_cfg, testing_training_cfg):
    source_name, source_trainlist, source_testlist = SOURCES[0].split(";")
    target_name, target_trainlist, target_testlist = TARGETS[0].split(";")

    # get cfg parameters
    cfg = testing_cfg
    cfg.DATASET.SOURCE = source_name
    cfg.DATASET.SRC_TRAINLIST = source_trainlist
    cfg.DATASET.SRC_TESTLIST = source_testlist
    cfg.DATASET.TARGET = target_name
    cfg.DATASET.TGT_TRAINLIST = target_trainlist
    cfg.DATASET.TGT_TESTLIST = target_testlist
    cfg.DATASET.IMAGE_MODALITY = image_modality
    cfg.DATASET.INPUT_TYPE = "image"
    cfg.DATASET.CLASS_TYPE = "verb"
    cfg.DATASET.NUM_SEGMENTS = 1
    cfg.DATASET.FRAMES_PER_SEGMENT = 16

    # download example data
    download_file_by_url(
        url=url,
        output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()),
        output_file_name="video_test_data.zip",
        file_format="zip",
    )

    # build dataset
    source, target, dict_num_classes = VideoDataset.get_source_target(
        VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    )

    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        random_state=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
    )

    # setup feature extractor
    if cfg.DATASET.IMAGE_MODALITY in ["rgb", "flow"]:
        class_feature_dim = 10
        domain_feature_dim = class_feature_dim
        if cfg.DATASET.IMAGE_MODALITY == "rgb":
            feature_network = {"rgb": VideoBoringModel(3, 10), "flow": None, "audio": None}
        else:
            feature_network = {"rgb": None, "flow": VideoBoringModel(2, 10), "audio": None}
    else:
        class_feature_dim = 20
        domain_feature_dim = int(class_feature_dim / 2)
        feature_network = {"rgb": VideoBoringModel(3, 10), "flow": VideoBoringModel(2, 10), "audio": None}

    # setup classifier
    class_type = cfg.DATASET.CLASS_TYPE
    classifier_network = ClassNetVideo(
        input_size=class_feature_dim, dict_n_class=dict_num_classes, class_type=class_type
    )
    train_params = testing_training_cfg["train_params"]

    # setup domain adapter
    model = DASetupHelper.setup_da(
        da_method,
        dataset,
        feature_network,
        classifier_network,
        class_type,
        train_params,
        domain_feature_dim,
        dict_num_classes,
        cfg,
    )

    ModelTestHelper.test_model(model, train_params)


@pytest.mark.parametrize("image_modality_feat", IMAGE_MODALITY_FEAT)
@pytest.mark.parametrize("da_method", DA_METHODS)
@pytest.mark.parametrize("class_type", CLASS_TYPE)
def test_video_domain_adapter_feature_vector(
    image_modality_feat, da_method, class_type, testing_cfg, testing_training_cfg
):
    source_name, source_trainlist, source_testlist = SOURCES[1].split(";")
    target_name, target_trainlist, target_testlist = TARGETS[1].split(";")

    # get cfg parameters
    cfg = testing_cfg
    cfg.DATASET.SOURCE = source_name
    cfg.DATASET.SRC_TRAINLIST = source_trainlist
    cfg.DATASET.SRC_TESTLIST = source_testlist
    cfg.DATASET.TARGET = target_name
    cfg.DATASET.TGT_TRAINLIST = target_trainlist
    cfg.DATASET.TGT_TESTLIST = target_testlist
    cfg.DATASET.IMAGE_MODALITY = image_modality_feat
    cfg.DATASET.INPUT_TYPE = "feature"
    cfg.DATASET.CLASS_TYPE = class_type
    cfg.DATASET.NUM_SEGMENTS = 8
    cfg.DATASET.FRAMES_PER_SEGMENT = 1

    # download example data
    download_file_by_url(
        url=url,
        output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()),
        output_file_name="video_test_data.zip",
        file_format="zip",
    )

    # build dataset
    source, target, dict_num_classes = VideoDataset.get_source_target(
        VideoDataset(cfg.DATASET.SOURCE.upper()), VideoDataset(cfg.DATASET.TARGET.upper()), seed, cfg
    )

    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        random_state=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
    )

    # setup feature extractor
    feat_rgb = feat_flow = feat_audio = None
    if cfg.DATASET.IMAGE_MODALITY in ["rgb", "all"]:
        feat_rgb = VideoVectorBoringModel(1024, 10)
    if cfg.DATASET.IMAGE_MODALITY in ["flow", "all"]:
        feat_flow = VideoVectorBoringModel(1024, 10)
    if cfg.DATASET.IMAGE_MODALITY in ["audio", "all"]:
        feat_audio = VideoVectorBoringModel(1024, 10)

    domain_feature_dim = int(10 * cfg.DATASET.NUM_SEGMENTS)
    if cfg.DATASET.IMAGE_MODALITY in ["rgb", "flow", "audio"]:
        class_feature_dim = domain_feature_dim
    else:
        class_feature_dim = int(domain_feature_dim * 3)

    feature_network = {"rgb": feat_rgb, "flow": feat_flow, "audio": feat_audio}

    # setup classifier
    class_type = cfg.DATASET.CLASS_TYPE
    classifier_network = ClassNetVideo(
        input_size=class_feature_dim, dict_n_class=dict_num_classes, class_type=class_type
    )
    train_params = testing_training_cfg["train_params"]

    # setup domain adapter
    model = DASetupHelper.setup_da(
        da_method,
        dataset,
        feature_network,
        classifier_network,
        class_type,
        train_params,
        domain_feature_dim,
        dict_num_classes,
        cfg,
    )

    ModelTestHelper.test_model(model, train_params)
