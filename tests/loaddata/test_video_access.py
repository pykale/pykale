import os
from pathlib import Path

import torch

import pytest
from kale.loaddata.video_access import get_image_modality, VideoDataset, VideoDatasetAccess
from kale.utils.download import download_compressed_file_by_url
from kale.utils.seed import set_seed
from yacs.config import CfgNode as CN

SOURCES = [
    "EPIC;8;epic_D1_train.pkl;epic_D1_test.pkl",
    "ADL;7;adl_P_04_train.pkl;adl_P_04_test.pkl",
    "GTEA;6;gtea_train.pkl;gtea_test.pkl",
    "KITCHEN;6;kitchen_train.pkl;kitchen_test.pkl",
]
TARGETS = [
    "EPIC;8;epic_D1_train.pkl;epic_D1_test.pkl",
    "ADL;7;adl_P_04_train.pkl;adl_P_04_test.pkl",
    "GTEA;6;gtea_train.pkl;gtea_test.pkl",
    "KITCHEN;6;kitchen_train.pkl;kitchen_test.pkl",
]
ALL = SOURCES + TARGETS
IMAGE_MODALITY = ["rgb", "flow", "joint"]
VAL_RATIO = [0.1]
seed = 36
set_seed(seed)

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/raw/main/video_data/video_test_data.zip"


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CN()
    cfg.DATASET = CN()
    cfg.DATASET.ROOT = root_dir + "/" + download_path + "/video_test_data/"
    cfg.DATASET.IMAGE_MODALITY = "joint"
    cfg.DATASET.FRAMES_PER_SEGMENT = 16
    yield cfg


@pytest.mark.parametrize("image_modality", IMAGE_MODALITY)
def test_get_image_modality(image_modality):
    rgb, flow = get_image_modality(image_modality)

    assert isinstance(rgb, bool)
    assert isinstance(flow, bool)


@pytest.mark.parametrize("source_cfg", SOURCES)
@pytest.mark.parametrize("target_cfg", TARGETS)
@pytest.mark.parametrize("val_ratio", VAL_RATIO)
def test_get_source_target(source_cfg, target_cfg, val_ratio, testing_cfg):
    source_name, source_n_class, source_trainlist, source_testlist = source_cfg.split(";")
    target_name, target_n_class, target_trainlist, target_testlist = target_cfg.split(";")
    n_class = eval(min(source_n_class, target_n_class))

    # get cfg parameters
    cfg = testing_cfg
    cfg.DATASET.SOURCE = source_name
    cfg.DATASET.SRC_TRAINLIST = source_trainlist
    cfg.DATASET.SRC_TESTLIST = source_testlist
    cfg.DATASET.TARGET = target_name
    cfg.DATASET.TAR_TRAINLIST = target_trainlist
    cfg.DATASET.TAR_TESTLIST = target_testlist

    download_compressed_file_by_url(
        url=url, output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()), output_file_name="video_test_data.zip"
    )

    # test get_source_target
    source, target, num_classes = VideoDataset.get_source_target(
        VideoDataset(source_name), VideoDataset(target_name), seed, cfg
    )

    assert num_classes == n_class
    assert isinstance(source, dict)
    assert isinstance(target, dict)
    assert isinstance(source["rgb"], VideoDatasetAccess)
    assert isinstance(target["rgb"], VideoDatasetAccess)
    assert isinstance(source["flow"], VideoDatasetAccess)
    assert isinstance(target["flow"], VideoDatasetAccess)

    # test get_train & get_test
    assert isinstance(source["rgb"].get_train(), torch.utils.data.Dataset)
    assert isinstance(source["rgb"].get_test(), torch.utils.data.Dataset)
    assert isinstance(source["flow"].get_train(), torch.utils.data.Dataset)
    assert isinstance(source["flow"].get_test(), torch.utils.data.Dataset)

    # test get_train_val
    assert isinstance(source["rgb"].get_train_val(val_ratio), list)
    assert isinstance(source["rgb"].get_train_val(val_ratio)[0], torch.utils.data.Dataset)
    assert isinstance(source["rgb"].get_train_val(val_ratio)[1], torch.utils.data.Dataset)
