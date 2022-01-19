import os

import numpy as np
import pytest

from kale.loaddata.tabular_access import load_csv_columns
from kale.utils.seed import set_seed

# from numpy import testing


# from tests.conftest import landmark_uncertainty_dl

# TODO: change the url when data is merged to main

# What tests
# 1)
# 2) ensure if setting is something else than "All", the correct columns are returned
# 3) ensure the correct split is returned
# 4) ensure that a single fold can be returned
# 5) ensure that a list of folds can be returned


seed = 36
set_seed(seed)

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/blob/landmark-data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip?raw=true"

# Download Landmark Uncertainty data
# test_data_path = landmark_uncertainty_dl("Uncertainty_tuples")


EXPECTED_COLS = [
    "uid",
    "E-CPV Error",
    "E-CPV Uncertainty",
    "E-MHA Error",
    "E-MHA Uncertainty",
    "S-MHA Error",
    "S-MHA Uncertainty",
    "Validation Fold",
    "Testing Fold",
]


@pytest.mark.parametrize("source_test_file", ["PHD-Net/4CH/uncertainty_pairs_test_l0"])
def test_load_csv_columns(landmark_uncertainty_dl, source_test_file):

    # ensure if setting is "All" that all columns are returned
    returned_cols = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file), "Testing Fold", np.arange(8), cols_to_return="All"
    )
    assert list(returned_cols.columns) == EXPECTED_COLS


# @pytest.fixture(scope="module")
# @pytest.mark.parametrize("download_path", "../../../landmark_data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples")
# def testing_cfg(download_path):
#     cfg = CN()
#     cfg.DATASET = CN()
#     cfg.DATASET.ROOT = root_dir + "/" + download_path
#     yield cfg


# @pytest.mark.parametrize("data_modalities", TARGET_MODALITIES)
# @pytest.mark.parametrize("models", TARGET_MODELS)
# @pytest.mark.parametrize("num_landmarks", TARGET_LANDMARKS)

# @pytest.mark.parametrize("target_lengths", TARGET_DATA_LENGTH)
# @pytest.mark.parametrize("num_folds", TARGET_FOLDS)


# def test_get_image_modality(image_modality):
#     rgb, flow = get_image_modality(image_modality)

#     assert isinstance(rgb, bool)
#     assert isinstance(flow, bool)


# @pytest.mark.parametrize("source_cfg", SOURCES)
# @pytest.mark.parametrize("target_cfg", TARGETS)
# @pytest.mark.parametrize("val_ratio", VAL_RATIO)
# @pytest.mark.parametrize("weight_type", WEIGHT_TYPE)
# @pytest.mark.parametrize("datasize_type", DATASIZE_TYPE)
# @pytest.mark.parametrize("class_subset", CLASS_SUBSETS)
# def test_get_source_target(source_cfg, target_cfg, val_ratio, weight_type, datasize_type, testing_cfg, class_subset):
#     source_name, source_n_class, source_trainlist, source_testlist = source_cfg.split(";")
#     target_name, target_n_class, target_trainlist, target_testlist = target_cfg.split(";")
#     n_class = eval(min(source_n_class, target_n_class))

#     # get cfg parameters
#     cfg = testing_cfg
#     cfg.DATASET.SOURCE = source_name
#     cfg.DATASET.SRC_TRAINLIST = source_trainlist
#     cfg.DATASET.SRC_TESTLIST = source_testlist
#     cfg.DATASET.TARGET = target_name
#     cfg.DATASET.TGT_TRAINLIST = target_trainlist
#     cfg.DATASET.TGT_TESTLIST = target_testlist
#     cfg.DATASET.WEIGHT_TYPE = weight_type
#     cfg.DATASET.SIZE_TYPE = datasize_type

#     download_file_by_url(
#         url=url,
#         output_directory=str(Path(cfg.DATASET.ROOT).parent.absolute()),
#         output_file_name="video_test_data.zip",
#         file_format="zip",
#     )

#     # test get_source_target
#     source, target, num_classes = VideoDataset.get_source_target(
#         VideoDataset(source_name), VideoDataset(target_name), seed, cfg
#     )

#     assert num_classes == n_class
#     assert isinstance(source, dict)
#     assert isinstance(target, dict)
#     assert isinstance(source["rgb"], VideoDatasetAccess)
#     assert isinstance(target["rgb"], VideoDatasetAccess)
#     assert isinstance(source["flow"], VideoDatasetAccess)
#     assert isinstance(target["flow"], VideoDatasetAccess)

#     # test get_train & get_test
#     assert isinstance(source["rgb"].get_train(), torch.utils.data.Dataset)
#     assert isinstance(source["rgb"].get_test(), torch.utils.data.Dataset)
#     assert isinstance(source["flow"].get_train(), torch.utils.data.Dataset)
#     assert isinstance(source["flow"].get_test(), torch.utils.data.Dataset)

#     # test get_train_val
#     train_val = source["rgb"].get_train_val(val_ratio)
#     assert isinstance(train_val, list)
#     assert isinstance(train_val[0], torch.utils.data.Dataset)
#     assert isinstance(train_val[1], torch.utils.data.Dataset)

#     # test action_multi_domain_datasets
#     dataset = VideoMultiDomainDatasets(
#         source,
#         target,
#         image_modality=cfg.DATASET.IMAGE_MODALITY,
#         seed=seed,
#         config_weight_type=cfg.DATASET.WEIGHT_TYPE,
#         config_size_type=cfg.DATASET.SIZE_TYPE,
#     )
#     assert isinstance(dataset, DomainsDatasetBase)

#     # test class subsets
#     if source_cfg == SOURCES[1] and target_cfg == TARGETS[0]:
#         dataset_subset = VideoMultiDomainDatasets(
#             source,
#             target,
#             image_modality="rgb",
#             seed=seed,
#             config_weight_type=cfg.DATASET.WEIGHT_TYPE,
#             config_size_type=cfg.DATASET.SIZE_TYPE,
#             class_ids=class_subset,
#         )

#         train, val = source["rgb"].get_train_val(val_ratio)
#         test = source["rgb"].get_test()
#         dataset_subset._rgb_source_by_split = {}
#         dataset_subset._rgb_target_by_split = {}
#         dataset_subset._rgb_source_by_split["train"] = get_class_subset(train, class_subset)
#         dataset_subset._rgb_target_by_split["train"] = dataset_subset._rgb_source_by_split["train"]
#         dataset_subset._rgb_source_by_split["val"] = get_class_subset(val, class_subset)
#         dataset_subset._rgb_source_by_split["test"] = get_class_subset(test, class_subset)

#         # Ground truth length of the subset dataset
#         train_dataset_subset_length = len([1 for data in train if data[1] in class_subset])
#         val_dataset_subset_length = len([1 for data in val if data[1] in class_subset])
#         test_dataset_subset_length = len([1 for data in test if data[1] in class_subset])
#         assert len(dataset_subset._rgb_source_by_split["train"]) == train_dataset_subset_length
#         assert len(dataset_subset._rgb_source_by_split["val"]) == val_dataset_subset_length
#         assert len(dataset_subset._rgb_source_by_split["test"]) == test_dataset_subset_length
#         assert len(dataset_subset) == train_dataset_subset_length
