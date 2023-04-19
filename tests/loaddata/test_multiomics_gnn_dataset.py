import os

import torch
from torch_geometric.data import Data, DataLoader

import kale.prepdata.tabular_transform as T
from kale.loaddata.multiomics_gnn_dataset import MogonetDataset, MultiOmicsDataset
from kale.utils.seed import set_seed


def test_multiomics_dataset():
    num_view = 3
    num_class = 2
    url = "https://github.com/SinaTabakhi/pykale-data/raw/main/multiomics/ROSMAP.zip"
    root = "tests/test_data/multiomics/"
    file_names = []
    for view in range(1, num_view + 1):
        file_names.append(f"{view}_tr.csv")
        file_names.append(f"{view}_lbl_tr.csv")
        file_names.append(f"{view}_te.csv")
        file_names.append(f"{view}_lbl_te.csv")

    dataset = MultiOmicsDataset(
        root=root,
        num_view=num_view,
        num_class=num_class,
        url=url,
        raw_file_names=file_names,
        random_split=False,
        train_size=0.7,
        pre_transform=T.ToTensor(dtype=torch.float),
        target_pre_transform=T.ToOneHotEncoding(dtype=torch.float),
    )

    # Test download method
    assert os.path.isfile(os.path.join(root, "raw/ROSMAP.zip"))
    assert os.path.isfile(os.path.join(root, "processed/data.pt"))

    for view in range(num_view * 4):
        assert os.path.exists(dataset.raw_paths[0])

    # Test load preprocessed data
    assert len(dataset) == 1
    dataloader = DataLoader(dataset, batch_size=1)
    next_batch = next(iter(dataloader))
    assert len(next_batch) == num_view

    # Test process method
    assert dataset.len() == num_view
    for view in range(num_view):
        data = dataset.get(view)
        assert isinstance(data, Data)
        assert len(data.x) == data.num_train + data.num_test
        assert len(data.y) == data.num_train + data.num_test
        assert len(data.train_idx) > 0
        assert len(data.train_idx) == data.num_train
        assert len(data.test_idx) > 0
        assert len(data.test_idx) == data.num_test

    for view in range(num_view):
        assert dataset.get(view).x.dtype == torch.float
        assert dataset.get(view).y.dtype == torch.float


def test_multiomics_dataset_random_split():
    set_seed(2023)
    num_view = 3
    num_class = 2
    url = "https://github.com/SinaTabakhi/pykale-data/raw/main/multiomics/ROSMAP.zip"
    root = "tests/test_data/multiomics/random/"
    file_names = []
    for view in range(1, num_view + 1):
        file_names.append(f"{view}_tr.csv")
        file_names.append(f"{view}_lbl_tr.csv")

    dataset = MultiOmicsDataset(
        root=root,
        num_view=num_view,
        num_class=num_class,
        url=url,
        raw_file_names=file_names,
        random_split=True,
        train_size=0.7,
        pre_transform=T.ToTensor(dtype=torch.float),
        target_pre_transform=T.ToOneHotEncoding(dtype=torch.float),
    )

    # Test download method
    assert os.path.isfile(os.path.join(root, "raw/ROSMAP.zip"))
    assert os.path.isfile(os.path.join(root, "processed/data.pt"))

    for view in range(num_view * 4):
        assert os.path.exists(dataset.raw_paths[0])

    # Test process method
    assert dataset.len() == num_view
    for view in range(num_view):
        data = dataset.get(view)
        assert isinstance(data, Data)
        assert len(data.x) == data.num_train + data.num_test
        assert len(data.y) == data.num_train + data.num_test
        assert len(data.train_idx) > 0
        assert abs(len(data.train_idx) - 0.7 * len(data.y)) <= 2
        assert len(data.train_idx) == data.num_train
        assert len(data.test_idx) > 0
        assert abs(len(data.test_idx) - 0.3 * len(data.y)) <= 2
        assert len(data.test_idx) == data.num_test

    for view in range(num_view):
        assert dataset.get(view).x.dtype == torch.float
        assert dataset.get(view).y.dtype == torch.float


def test_mogonet_dataset():
    num_view = 3
    num_class = 2
    url = "https://github.com/SinaTabakhi/pykale-data/raw/main/multiomics/ROSMAP.zip"
    root = "tests/test_data/mogonet/"
    file_names = []
    for view in range(1, num_view + 1):
        file_names.append(f"{view}_tr.csv")
        file_names.append(f"{view}_lbl_tr.csv")
        file_names.append(f"{view}_te.csv")
        file_names.append(f"{view}_lbl_te.csv")

    dataset = MogonetDataset(
        root=root,
        raw_file_names=file_names,
        num_view=num_view,
        num_class=num_class,
        edge_per_node=10,
        url=url,
        random_split=False,
        train_size=0.7,
        equal_weight=False,
        pre_transform=T.ToTensor(dtype=torch.float),
        target_pre_transform=T.ToOneHotEncoding(dtype=torch.float),
    )

    # Test download method
    assert os.path.isfile(os.path.join(root, "raw/ROSMAP.zip"))
    assert os.path.isfile(os.path.join(root, "processed/data.pt"))

    for view in range(num_view * 4):
        assert os.path.exists(dataset.raw_paths[0])

    assert dataset.num_view == num_view
    assert dataset.num_class == num_class

    # Test process method
    assert dataset.len() == num_view

    for view in range(num_view):
        data = dataset.get(view)
        assert isinstance(data, Data)
        assert len(data.x) == data.num_train + data.num_test
        assert len(data.y) == data.num_train + data.num_test
        assert len(data.train_idx) > 0
        assert len(data.train_idx) == data.num_train
        assert len(data.test_idx) > 0
        assert len(data.test_idx) == data.num_test

    for view in range(num_view):
        assert dataset.get(view).x.dtype == torch.float
        assert dataset.get(view).y.dtype == torch.float

    # Test extend_data method
    for view in range(num_view):
        data = dataset.get(view)
        assert len(data.train_sample_weight) == len(data.y[data.train_idx])
        assert hasattr(data, "train_sample_weight")
        assert hasattr(data, "edge_index_train")
        assert hasattr(data, "edge_weight_train")
        assert hasattr(data, "adj_t_train")

    assert repr(dataset) is not None
    assert len(repr(dataset)) > 0
