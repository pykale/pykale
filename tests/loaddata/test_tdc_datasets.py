import pytest
import torch

from kale.loaddata.tdc_datasets import BindingDBDataset

SOURCES = ["BindingDB_Kd", "BindingDB_Ki"]


@pytest.mark.parametrize("source_name", SOURCES)
def test_tdc_datasets(download_path, source_name):
    test_dataset = BindingDBDataset(name=source_name, split="test", path=download_path)
    assert isinstance(test_dataset, torch.utils.data.Dataset)
    attributes = ["Drug", "Target", "Y"]
    for attribute in attributes:
        assert attribute in test_dataset.data.columns
    drug, protein, label = test_dataset[0]
    assert drug.dtype == torch.int64
    assert protein.dtype == torch.int64
    assert label.dtype == torch.float32
