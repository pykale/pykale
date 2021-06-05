import os
from pathlib import Path

import pytest
import torch
from yacs.config import CfgNode as CN

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
    "ADL;7;adl_P_04_train.pkl;adl_P_04_test.pkl",
    "GTEA;6;gtea_train.pkl;gtea_test.pkl",
    "KITCHEN;6;kitchen_train.pkl;kitchen_test.pkl",
]
ALL = SOURCES + TARGETS
IMAGE_MODALITY = ["rgb", "flow", "joint"]
WEIGHT_TYPE = ["natural", "balanced", "preset0"]
DATASIZE_TYPE = ["max", "source"]
VAL_RATIO = [0.1]
seed = 36
set_seed(seed)
CLASS_SUB_SAMPLES = [[1, 3, 8]]

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
@pytest.mark.parametrize("weight_type", WEIGHT_TYPE)
@pytest.mark.parametrize("datasize_type", DATASIZE_TYPE)
@pytest.mark.parametrize("class_sub_sample", CLASS_SUB_SAMPLES)
def test_get_source_target(
    source_cfg, target_cfg, val_ratio, weight_type, datasize_type, testing_cfg, class_sub_sample
):
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

    # test get_train_val
    train_val = source["rgb"].get_train_val(val_ratio)
    assert isinstance(train_val, list)
    assert isinstance(train_val[0], torch.utils.data.Dataset)
    assert isinstance(train_val[1], torch.utils.data.Dataset)

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

    # test class sub-sampling
    if source_cfg == SOURCES[1] and target_cfg == TARGETS[1]:
        subsampled_train_val = source["rgb"].get_train_val(val_ratio, class_sub_sample)
        assert isinstance(subsampled_train_val, list)
        assert isinstance(subsampled_train_val[0], torch.utils.data.Dataset)
        assert isinstance(subsampled_train_val[1], torch.utils.data.Dataset)

        assert len(subsampled_train_val[0]) <= len(train_val[0])
        assert len(subsampled_train_val[1]) <= len(train_val[1])

        dataset_subsampled = VideoMultiDomainDatasets(
            source,
            target,
            image_modality=cfg.DATASET.IMAGE_MODALITY,
            seed=seed,
            config_weight_type=cfg.DATASET.WEIGHT_TYPE,
            config_size_type=cfg.DATASET.SIZE_TYPE,
            sub_class_ids=class_sub_sample,
        )
        assert isinstance(dataset_subsampled, DomainsDatasetBase)
        if dataset.rgb:
            dataset._rgb_source_by_split = {"train": train_val[0]}
            dataset._rgb_target_by_split = {"train": train_val[0]}
        if dataset.flow:
            dataset._flow_source_by_split = {"train": train_val[0]}
            dataset._flow_target_by_split = {"train": train_val[0]}
        if dataset_subsampled.rgb:
            dataset_subsampled._rgb_source_by_split = {"train": subsampled_train_val[0]}
            dataset_subsampled._rgb_target_by_split = {"train": subsampled_train_val[0]}
        if dataset_subsampled.flow:
            dataset_subsampled._flow_source_by_split = {"train": subsampled_train_val[0]}
            dataset_subsampled._flow_target_by_split = {"train": subsampled_train_val[0]}
        assert len(dataset_subsampled) <= len(dataset)
