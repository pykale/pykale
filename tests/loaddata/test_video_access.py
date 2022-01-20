import os
from pathlib import Path

import pytest
import torch
from yacs.config import CfgNode as CN

from kale.loaddata.dataset_access import get_class_subset
from kale.loaddata.multi_domain import DomainsDatasetBase
from kale.loaddata.video_access import get_image_modality, VideoDataset, VideoDatasetAccess
from kale.loaddata.video_multi_domain import VideoMultiDomainDatasets
from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

SOURCES = [
    "EPIC;8;epic_D1_train.pkl;epic_D1_test.pkl",
    "ADL;7;adl_P_11_train.pkl;adl_P_11_test.pkl",
    "GTEA;6;gtea_train.pkl;gtea_test.pkl",
    "KITCHEN;6;kitchen_train.pkl;kitchen_test.pkl",
]
TARGETS = [
    "EPIC;8;epic_D1_train.pkl;epic_D1_test.pkl",
    # "ADL;7;adl_P_04_train.pkl;adl_P_04_test.pkl",
    # "GTEA;6;gtea_train.pkl;gtea_test.pkl",
    # "KITCHEN;6;kitchen_train.pkl;kitchen_test.pkl",
]
ALL = SOURCES + TARGETS
IMAGE_MODALITY = ["rgb", "flow", "joint"]
WEIGHT_TYPE = ["natural", "balanced", "preset0"]
# DATASIZE_TYPE = ["max", "source"]
DATASIZE_TYPE = ["max"]
VALID_RATIO = [0.1]
seed = 36
set_seed(seed)
CLASS_SUBSETS = [[1, 3, 8]]

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/raw/main/videos/video_test_data.zip"


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
@pytest.mark.parametrize("valid_ratio", VALID_RATIO)
@pytest.mark.parametrize("weight_type", WEIGHT_TYPE)
@pytest.mark.parametrize("datasize_type", DATASIZE_TYPE)
@pytest.mark.parametrize("class_subset", CLASS_SUBSETS)
def test_get_source_target(source_cfg, target_cfg, valid_ratio, weight_type, datasize_type, testing_cfg, class_subset):
    source_name, source_n_class, source_trainlist, source_testlist = source_cfg.split(";")
    target_name, target_n_class, target_trainlist, target_testlist = target_cfg.split(";")
    n_class = eval(min(source_n_class, target_n_class))

    # get cfg parameters
    cfg = testing_cfg
    cfg.DATASET.SOURCE = source_name
    cfg.DATASET.SRC_TRAINLIST = source_trainlist
    cfg.DATASET.SRC_TESTLIST = source_testlist
    cfg.DATASET.TARGET = target_name
    cfg.DATASET.TGT_TRAINLIST = target_trainlist
    cfg.DATASET.TGT_TESTLIST = target_testlist
    cfg.DATASET.WEIGHT_TYPE = weight_type
    cfg.DATASET.SIZE_TYPE = datasize_type

    download_file_by_url(
        url=url,
        output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()),
        output_file_name="video_test_data.zip",
        file_format="zip",
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

    # test get_train_valid
    train_valid = source["rgb"].get_train_valid(valid_ratio)
    assert isinstance(train_valid, list)
    assert isinstance(train_valid[0], torch.utils.data.Dataset)
    assert isinstance(train_valid[1], torch.utils.data.Dataset)

    # test action_multi_domain_datasets
    dataset = VideoMultiDomainDatasets(
        source,
        target,
        image_modality=cfg.DATASET.IMAGE_MODALITY,
        seed=seed,
        config_weight_type=cfg.DATASET.WEIGHT_TYPE,
        config_size_type=cfg.DATASET.SIZE_TYPE,
    )
    assert isinstance(dataset, DomainsDatasetBase)

    # test class subsets
    if source_cfg == SOURCES[1] and target_cfg == TARGETS[0]:
        dataset_subset = VideoMultiDomainDatasets(
            source,
            target,
            image_modality="rgb",
            seed=seed,
            config_weight_type=cfg.DATASET.WEIGHT_TYPE,
            config_size_type=cfg.DATASET.SIZE_TYPE,
            class_ids=class_subset,
        )

        train, valid = source["rgb"].get_train_valid(valid_ratio)
        test = source["rgb"].get_test()
        dataset_subset._rgb_source_by_split = {}
        dataset_subset._rgb_target_by_split = {}
        dataset_subset._rgb_source_by_split["train"] = get_class_subset(train, class_subset)
        dataset_subset._rgb_target_by_split["train"] = dataset_subset._rgb_source_by_split["train"]
        dataset_subset._rgb_source_by_split["valid"] = get_class_subset(valid, class_subset)
        dataset_subset._rgb_source_by_split["test"] = get_class_subset(test, class_subset)

        # Ground truth length of the subset dataset
        train_dataset_subset_length = len([1 for data in train if data[1] in class_subset])
        valid_dataset_subset_length = len([1 for data in valid if data[1] in class_subset])
        test_dataset_subset_length = len([1 for data in test if data[1] in class_subset])
        assert len(dataset_subset._rgb_source_by_split["train"]) == train_dataset_subset_length
        assert len(dataset_subset._rgb_source_by_split["valid"]) == valid_dataset_subset_length
        assert len(dataset_subset._rgb_source_by_split["test"]) == test_dataset_subset_length
        assert len(dataset_subset) == train_dataset_subset_length
