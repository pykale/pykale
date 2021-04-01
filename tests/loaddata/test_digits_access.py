import pytest

from kale.loaddata.digits_access import DigitDataset, DigitDatasetAccess
from kale.loaddata.multi_domain import DomainsDatasetBase, MultiDomainDatasets

# import torch
# from typing import Dict


SOURCES = ["MNIST", "USPS"]
TARGETS = ["MNISTM", "SVHN"]
ALL = SOURCES + TARGETS

WEIGHT_TYPE = ["natural", "balanced", "preset0"]
DATASIZE_TYPE = ["max", "source"]

# @pytest.fixture(scope="module")
# def testing_access():
#     access = DatasetAccess(n_classes)
#     return access


# def test_n_classes(testing_access):
#     get_n_class = testing_access.n_classes()
#     assert get_n_class == n_classes


@pytest.mark.parametrize("source_name", SOURCES)
@pytest.mark.parametrize("target_name", TARGETS)
@pytest.mark.parametrize("weight_type", WEIGHT_TYPE)
@pytest.mark.parametrize("datasize_type", DATASIZE_TYPE)
def test_get_source_target(source_name, target_name, weight_type, datasize_type, download_path):
    source, target, num_channels = DigitDataset.get_source_target(DigitDataset(source_name), DigitDataset(target_name), download_path)
    assert num_channels == 3
    assert isinstance(source, DigitDatasetAccess)
    assert isinstance(target, DigitDatasetAccess)

    dataset = MultiDomainDatasets(source, target, config_weight_type=weight_type, config_size_type=datasize_type)
    assert isinstance(dataset, DomainsDatasetBase)


# @pytest.mark.parametrize("dataset", ALL)
# def test_get_train_test(dataset, download_path):
#     source, target, num_channels = DigitDataset.get_source_target(DigitDataset(source_name), DigitDataset(target_name), download_path)
#     assert num_channels == 3
#     assert isinstance(source, DigitDatasetAccess)
#     assert isinstance(target, DigitDatasetAccess)
