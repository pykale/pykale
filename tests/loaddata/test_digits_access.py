import pytest
import torch

from kale.loaddata.digits_access import DigitDataset, DigitDatasetAccess
from kale.loaddata.multi_domain import DomainsDatasetBase, MultiDomainDatasets

# from typing import Dict


SOURCES = ["MNIST", "USPS"]
TARGETS = ["MNISTM", "SVHN"]
ALL = SOURCES + TARGETS

WEIGHT_TYPE = ["natural", "balanced", "preset0"]
DATASIZE_TYPE = ["max", "source"]
VAL_RATIO = [0.1]


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
