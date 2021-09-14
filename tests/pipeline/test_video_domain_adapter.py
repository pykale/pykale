import os
from pathlib import Path

import pytest
from yacs.config import CfgNode as CN

from kale.loaddata.video_access import VideoDataset
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.pipeline import domain_adapter, video_domain_adapter
from kale.predict.class_domain_nets import ClassNetVideo, DomainNetVideo
from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed
from tests.helpers.boring_model import VideoBoringModel
from tests.helpers.pipe_test_helper import ModelTestHelper

SOURCES = [
    "ADL;7;adl_P_11_train.pkl;adl_P_11_test.pkl",
]
TARGETS = [
    "ADL;7;adl_P_11_train.pkl;adl_P_11_test.pkl",
]
ALL = SOURCES + TARGETS
IMAGE_MODALITY = ["rgb", "flow", "joint"]
DA_METHODS = ["DANN", "CDAN", "CDAN-E", "WDGRL", "DAN", "JAN", "Source"]
WEIGHT_TYPE = "natural"
DATASIZE_TYPE = "max"
# INPUT_TYPE = ["image", "feature"]
INPUT_TYPE = ["image"]
CLASS_TYPE_IMAGE = ["verb"]
# CLASS_TYPE_FEAT = ["noun", "verb+noun"]
# VERB_CLASS = True
# NOUN_CLASS = False
TRAIN_BATCH_SIZE = 2
VAL_RATIO = 0.1
seed = 36
set_seed(seed)

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DAN = CN()
    cfg.SOLVER = CN()
    cfg.SOLVER.WORKERS = 0
    cfg.DATASET.ROOT = root_dir + "/" + download_path + "/video_test_data/"
    cfg.DATASET.FRAMES_PER_SEGMENT = 4
    cfg.DATASET.NUM_SEGMENTS = 1
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
            "batch_size": TRAIN_BATCH_SIZE,
            "optimizer": {"type": "SGD", "optim_params": {"momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}},
        }
    }
    yield config_params


@pytest.mark.parametrize("source_cfg", SOURCES)
@pytest.mark.parametrize("target_cfg", TARGETS)
@pytest.mark.parametrize("image_modality", IMAGE_MODALITY)
@pytest.mark.parametrize("da_method", DA_METHODS)
@pytest.mark.parametrize("input_type", INPUT_TYPE)
@pytest.mark.parametrize("class_type", CLASS_TYPE_IMAGE)
def test_video_domain_adapter(source_cfg, target_cfg, image_modality, da_method, testing_cfg, testing_training_cfg, input_type, class_type):
    source_name, source_n_class, source_trainlist, source_testlist = source_cfg.split(";")
    target_name, target_n_class, target_trainlist, target_testlist = target_cfg.split(";")

    # get cfg parameters
    cfg = testing_cfg
    cfg.DATASET.SOURCE = source_name
    cfg.DATASET.SRC_TRAINLIST = source_trainlist
    cfg.DATASET.SRC_TESTLIST = source_testlist
    cfg.DATASET.TARGET = target_name
    cfg.DATASET.TGT_TRAINLIST = target_trainlist
    cfg.DATASET.TGT_TESTLIST = target_testlist
    cfg.DATASET.IMAGE_MODALITY = image_modality
    cfg.DATASET.WEIGHT_TYPE = WEIGHT_TYPE
    cfg.DATASET.SIZE_TYPE = DATASIZE_TYPE
    cfg.DATASET.INPUT_TYPE = input_type
    cfg.DATASET.CLASS_TYPE = class_type
    # cfg.DATASET.VERB_CLASS = VERB_CLASS
    # cfg.DATASET.NOUN_CLASS = NOUN_CLASS
    cfg.SOLVER.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    cfg.DAN.USERANDOM = False

    # download example data
    download_file_by_url(
        url=url,
        output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()),
        output_file_name="video_test_data.zip",
        file_format="zip",
    )

    # build dataset
    source, target, dict_num_classes = VideoDataset.get_source_target(
        VideoDataset(source_name), VideoDataset(target_name), seed, cfg
    )

    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        random_state=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
        num_workers=cfg.SOLVER.WORKERS,
    )

    # setup feature extractor
    if cfg.DATASET.IMAGE_MODALITY in ["rgb", "flow"]:
        class_feature_dim = 16
        domain_feature_dim = class_feature_dim
        if cfg.DATASET.IMAGE_MODALITY == "rgb":
            feature_network = {"rgb": VideoBoringModel(3), "flow": None, "audio": None}
        else:
            feature_network = {"rgb": None, "flow": VideoBoringModel(2), "audio": None}
    else:
        class_feature_dim = 32
        domain_feature_dim = int(class_feature_dim / 2)
        feature_network = {"rgb": VideoBoringModel(3), "flow": VideoBoringModel(2), "audio": None}

    # setup classifier
    classifier_network = ClassNetVideo(
        input_size=class_feature_dim, dict_n_class=dict_num_classes, class_type=class_type.lower()
    )
    train_params = testing_training_cfg["train_params"]
    method_params = {}
    method = domain_adapter.Method(da_method)

    # setup DA method
    if method.is_mmd_method():
        model = video_domain_adapter.create_mmd_based_video(
            method=method,
            dataset=dataset,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            input_type=input_type,
            class_type=class_type,
            **method_params,
            **train_params,
        )
    else:
        critic_input_size = domain_feature_dim
        # setup critic network
        if method.is_cdan_method():
            if cfg.DAN.USERANDOM:
                critic_input_size = 1024
            else:
                critic_input_size = domain_feature_dim * dict_num_classes["verb"]
        critic_network = DomainNetVideo(input_size=critic_input_size)

        if da_method == "CDAN":
            method_params["use_random"] = cfg.DAN.USERANDOM

        model = video_domain_adapter.create_dann_like_video(
            method=method,
            dataset=dataset,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            feature_extractor=feature_network,
            task_classifier=classifier_network,
            critic=critic_network,
            input_type=input_type,
            class_type=class_type,
            **method_params,
            **train_params,
        )

    ModelTestHelper.test_model(model, train_params)
