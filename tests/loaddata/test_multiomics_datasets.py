import os

import pytest
import torch
from torch_geometric.data import Data, DataLoader

from kale.loaddata.multiomics_datasets import MultiomicsDataset, SparseMultiomicsDataset
from kale.prepdata.tabular_transform import ToOneHotEncoding, ToTensor
from kale.utils.seed import set_seed


@pytest.mark.parametrize("random_split", [True, False])
def test_multiomics_datasets(random_split):
    num_modalities = 3
    num_classes = 2
    url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"

    if random_split:
        root = "tests/test_data/multiomics/random/"
        set_seed(2023)
    else:
        root = "tests/test_data/multiomics/pre_split/"

    file_names = []
    for modality in range(1, num_modalities + 1):
        file_names.append(f"{modality}_tr.csv")
        file_names.append(f"{modality}_lbl_tr.csv")
        if not random_split:
            file_names.append(f"{modality}_te.csv")
            file_names.append(f"{modality}_lbl_te.csv")

    dataset = MultiomicsDataset(
        root=root,
        num_modalities=num_modalities,
        num_classes=num_classes,
        url=url,
        raw_file_names=file_names,
        random_split=random_split,
        train_size=0.7,
        pre_transform=ToTensor(dtype=torch.float),
        target_pre_transform=ToOneHotEncoding(dtype=torch.float),
    )

    # Test download method
    assert os.path.isfile(os.path.join(root, "raw/ROSMAP.zip"))
    assert os.path.isfile(os.path.join(root, "processed/data.pt"))

    for modality in range(num_modalities * 4):
        assert os.path.exists(dataset.raw_paths[0])

    # Test load preprocessed data
    assert len(dataset) == 1
    dataloader = DataLoader(dataset, batch_size=1)
    next_batch = next(iter(dataloader))
    assert len(next_batch) == num_modalities

    # Test process method
    assert dataset.len() == num_modalities
    for modality in range(num_modalities):
        data = dataset.get(modality)
        assert isinstance(data, Data)
        assert len(data.x) == data.num_train + data.num_test
        assert len(data.y) == data.num_train + data.num_test
        assert len(data.train_idx) > 0
        assert len(data.train_idx) == data.num_train
        assert len(data.test_idx) > 0
        assert len(data.test_idx) == data.num_test

    for modality in range(num_modalities):
        assert dataset.get(modality).x.dtype == torch.float
        assert dataset.get(modality).y.dtype == torch.float


def test_sparse_multiomics_datasets():
    num_modalities = 3
    num_classes = 2
    url = "https://github.com/pykale/data/raw/main/multiomics/ROSMAP.zip"
    root = "tests/test_data/sparse/"
    file_names = []
    for modality in range(1, num_modalities + 1):
        file_names.append(f"{modality}_tr.csv")
        file_names.append(f"{modality}_lbl_tr.csv")
        file_names.append(f"{modality}_te.csv")
        file_names.append(f"{modality}_lbl_te.csv")

    dataset = SparseMultiomicsDataset(
        root=root,
        raw_file_names=file_names,
        num_modalities=num_modalities,
        num_classes=num_classes,
        edge_per_node=10,
        url=url,
        random_split=False,
        train_size=0.7,
        equal_weight=False,
        pre_transform=ToTensor(dtype=torch.float),
        target_pre_transform=ToOneHotEncoding(dtype=torch.float),
    )

    # Test download method
    assert os.path.isfile(os.path.join(root, "raw/ROSMAP.zip"))
    assert os.path.isfile(os.path.join(root, "processed/data.pt"))

    for modality in range(num_modalities * 4):
        assert os.path.exists(dataset.raw_paths[0])

    assert dataset.num_modalities == num_modalities
    assert dataset.num_classes == num_classes

    # Test process method
    assert dataset.len() == num_modalities

    for modality in range(num_modalities):
        data = dataset.get(modality)
        assert isinstance(data, Data)
        assert len(data.x) == data.num_train + data.num_test
        assert len(data.y) == data.num_train + data.num_test
        assert len(data.train_idx) > 0
        assert len(data.train_idx) == data.num_train
        assert len(data.test_idx) > 0
        assert len(data.test_idx) == data.num_test

    for modality in range(num_modalities):
        assert dataset.get(modality).x.dtype == torch.float
        assert dataset.get(modality).y.dtype == torch.float

    # Test extend_data method
    for modality in range(num_modalities):
        data = dataset.get(modality)
        assert len(data.train_sample_weight) == len(data.y[data.train_idx])
        assert hasattr(data, "train_sample_weight")
        assert hasattr(data, "edge_index_train")
        assert hasattr(data, "edge_weight_train")
        assert hasattr(data, "adj_t_train")

    assert str(dataset) is not None
    assert len(str(dataset)) > 0
