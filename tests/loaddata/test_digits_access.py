import pytest
import torch

from kale.loaddata.dataset_access import get_class_subset
from kale.loaddata.digits_access import DigitDataset, DigitDatasetAccess
from kale.loaddata.multi_domain import DomainsDatasetBase, MultiDomainDatasets

# from typing import Dict


SOURCES = ["MNIST", "USPS"]
TARGETS = ["MNISTM", "SVHN"]
ALL = ["SVHN", "USPS"]  # SOURCES + TARGETS

WEIGHT_TYPE = ["natural", "balanced", "preset0"]
DATASIZE_TYPE = ["max", "source"]
VAL_RATIO = [0.1]

CLASS_SUBSETS = [[1, 3, 8]]


@pytest.mark.parametrize("source_name", SOURCES)
@pytest.mark.parametrize("target_name", TARGETS)
def test_get_source_target(source_name, target_name, download_path):
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(source_name), DigitDataset(target_name), download_path
    )
    assert num_channels == 3
    assert isinstance(source, DigitDatasetAccess)
    assert isinstance(target, DigitDatasetAccess)


@pytest.mark.parametrize("weight_type", WEIGHT_TYPE)
@pytest.mark.parametrize("datasize_type", DATASIZE_TYPE)
def test_multi_domain_datasets(weight_type, datasize_type, download_path):
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(SOURCES[0]), DigitDataset(TARGETS[0]), download_path
    )
    assert num_channels == 3
    assert isinstance(source, DigitDatasetAccess)
    assert isinstance(target, DigitDatasetAccess)

    dataset = MultiDomainDatasets(source, target, config_weight_type=weight_type, config_size_type=datasize_type)
    assert isinstance(dataset, DomainsDatasetBase)


@pytest.mark.parametrize("dataset_name", ALL)
def test_get_train_test(dataset_name, download_path):
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(dataset_name), DigitDataset(dataset_name), download_path
    )
    source_train = source.get_train()
    source_test = source.get_test()
    assert source.n_classes() == 10
    assert isinstance(source_train, torch.utils.data.Dataset)
    assert isinstance(source_test, torch.utils.data.Dataset)


@pytest.mark.parametrize("class_subset", CLASS_SUBSETS)
@pytest.mark.parametrize("val_ratio", VAL_RATIO)
def test_class_subsets(class_subset, val_ratio, download_path):
    dataset_name = ALL[1]
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(dataset_name), DigitDataset(dataset_name), download_path
    )

    dataset_subset = MultiDomainDatasets(
        source, target, config_weight_type=WEIGHT_TYPE[0], config_size_type=DATASIZE_TYPE[1], class_ids=class_subset,
    )

    train, val = source.get_train_val(val_ratio)
    test = source.get_test()
    dataset_subset._source_by_split["train"] = get_class_subset(train, class_subset)
    dataset_subset._target_by_split["train"] = dataset_subset._source_by_split["train"]
    dataset_subset._source_by_split["val"] = get_class_subset(val, class_subset)
    dataset_subset._source_by_split["test"] = get_class_subset(test, class_subset)

    # Ground truth lengths
    train_dataset_subset_length = len([1 for data in train if data[1] in class_subset])
    val_dataset_subset_length = len([1 for data in val if data[1] in class_subset])
    test_dataset_subset_length = len([1 for data in test if data[1] in class_subset])

    assert len(dataset_subset._source_by_split["train"]) == train_dataset_subset_length
    assert len(dataset_subset._source_by_split["val"]) == val_dataset_subset_length
    assert len(dataset_subset._source_by_split["test"]) == test_dataset_subset_length
    assert len(dataset_subset) == train_dataset_subset_length
