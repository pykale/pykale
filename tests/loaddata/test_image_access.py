import pandas as pd
import pytest
import torch
from numpy import testing
from yacs.config import CfgNode

from kale.loaddata.dataset_access import get_class_subset
from kale.loaddata.image_access import DigitDataset, DigitDatasetAccess, get_cifar, ImageAccess, load_images_from_dir
from kale.loaddata.multi_domain import DomainsDatasetBase, MultiDomainAdapDataset, MultiDomainDatasets


@pytest.mark.parametrize("test_on_all", [True, False])
def test_office31(office_path, test_on_all):
    office_access = ImageAccess.get_multi_domain_images(
        "OFFICE31", office_path, download=False, return_domain_label=True
    )
    testing.assert_equal(len(office_access.class_to_idx), 31)
    testing.assert_equal(len(office_access.domain_to_idx), 3)
    dataset = MultiDomainAdapDataset(office_access, test_on_all=test_on_all)
    dataset.prepare_data_loaders()
    domain_labels = list(dataset.domain_to_idx.values())
    for split in ["train", "valid", "test"]:
        dataloader = dataset.get_domain_loaders(split=split)
        x, y, z = next(iter(dataloader))
        for domain_label_ in domain_labels:
            testing.assert_equal(y[z == domain_label_].shape[0], 10)


def test_office_caltech(office_path):
    office_access = ImageAccess.get_multi_domain_images(
        "OFFICE_CALTECH", office_path, download=False, return_domain_label=True
    )
    testing.assert_equal(len(office_access.class_to_idx), 10)
    testing.assert_equal(len(office_access.domain_to_idx), 4)


@pytest.mark.parametrize("split_ratio", [0.9, 1])
def test_custom_office(office_path, split_ratio):
    kwargs = {"download": False, "split_train_test": True, "split_ratio": split_ratio}
    source = ImageAccess.get_multi_domain_images("office", office_path, sub_domain_set=["dslr"], **kwargs)
    target = ImageAccess.get_multi_domain_images("office", office_path, sub_domain_set=["webcam"], **kwargs)
    dataset = MultiDomainDatasets(source_access=source, target_access=target)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders()
    testing.assert_equal(len(next(iter(dataloader))), 2)


SOURCES = ["MNIST", "USPS"]
TARGETS = ["MNISTM", "SVHN"]
ALL = SOURCES + TARGETS  # ["SVHN", "USPS", "MNISTM"]  # SOURCES + TARGETS

WEIGHT_TYPE = ["natural", "balanced", "preset0"]
DATASIZE_TYPE = ["max", "source"]
VALID_RATIO = [0.1]

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
@pytest.mark.parametrize("valid_ratio", VALID_RATIO)
def test_class_subsets(class_subset, valid_ratio, download_path):
    dataset_name = ALL[1]
    source, target, num_channels = DigitDataset.get_source_target(
        DigitDataset(dataset_name), DigitDataset(dataset_name), download_path
    )

    dataset_subset = MultiDomainDatasets(
        source,
        target,
        config_weight_type=WEIGHT_TYPE[0],
        config_size_type=DATASIZE_TYPE[1],
        class_ids=class_subset,
    )

    train, valid = source.get_train_valid(valid_ratio)
    test = source.get_test()
    dataset_subset._source_by_split["train"] = get_class_subset(train, class_subset)
    dataset_subset._target_by_split["train"] = dataset_subset._source_by_split["train"]
    dataset_subset._source_by_split["valid"] = get_class_subset(valid, class_subset)
    dataset_subset._source_by_split["test"] = get_class_subset(test, class_subset)

    # Ground truth lengths
    train_dataset_subset_length = len([1 for data in train if data[1] in class_subset])
    valid_dataset_subset_length = len([1 for data in valid if data[1] in class_subset])
    test_dataset_subset_length = len([1 for data in test if data[1] in class_subset])

    assert len(dataset_subset._source_by_split["train"]) == train_dataset_subset_length
    assert len(dataset_subset._source_by_split["valid"]) == valid_dataset_subset_length
    assert len(dataset_subset._source_by_split["test"]) == test_dataset_subset_length
    assert len(dataset_subset) == train_dataset_subset_length


def test_multi_domain_digits(download_path):
    data_access = ImageAccess.get_multi_domain_images(
        "DIGITS", download_path, sub_domain_set=["SVHN", "USPS_RGB", "MNISTM"], return_domain_label=True
    )
    dataset = MultiDomainAdapDataset(data_access)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders(split="test", batch_size=10)
    assert len(next(iter(dataloader))) == 3


DATASET_NAMES = ["CIFAR10", "CIFAR100"]


@pytest.fixture(scope="module")
def testing_cfg(download_path):
    cfg = CfgNode()
    cfg.DATASET = CfgNode()
    cfg.SOLVER = CfgNode()
    cfg.DATASET.ROOT = download_path
    cfg.DATASET.DOWNLOAD = False
    cfg.SOLVER.TRAIN_BATCH_SIZE = 16
    cfg.SOLVER.TEST_BATCH_SIZE = 20
    cfg.DATASET.NUM_WORKERS = 1
    yield cfg

    # Teardown: remove data files, or not?
    # files = glob.glob(cfg.DATASET.ROOT)
    # for f in files:
    #     os.remove(f)


@pytest.mark.parametrize("dataset", DATASET_NAMES)
def test_get_cifar(dataset, testing_cfg):
    cfg = testing_cfg
    cfg.DATASET.NAME = dataset
    train_loader, valid_loader = get_cifar(cfg)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(valid_loader, torch.utils.data.DataLoader)


# Example dummy prepare_image_tensor for the test
def prepare_image_tensor(image_path, resize_dim=(224, 224), channels=1):
    if channels == 1:
        return torch.ones(1, resize_dim[0], resize_dim[1])
    else:
        return torch.ones(3, resize_dim[0], resize_dim[1])


@pytest.mark.parametrize("channels", [1, 3])
def test_load_images_from_dir(tmp_path, channels, monkeypatch):
    root = tmp_path
    num_images = 5
    resize_dim = (32, 32)
    file_names = [f"img_{i}.png" for i in range(num_images)]

    for fname in file_names:
        img_path = root / fname
        with open(img_path, "wb") as f:
            f.write(b"")

    # Write a CSV file listing the images
    csv_path = root / "images.csv"
    df = pd.DataFrame({"file_path": file_names})
    df.to_csv(csv_path, index=False)

    def dummy_prepare_image_tensor(image_path, resize_dim=(32, 32), channels=1):
        return torch.ones(channels, *resize_dim)

    monkeypatch.setattr("kale.loaddata.image_access.prepare_image_tensor", dummy_prepare_image_tensor)

    batch = load_images_from_dir(root, "images.csv", resize_dim=resize_dim, channels=channels)
    assert batch.shape == (num_images, channels, *resize_dim)
    assert torch.all(batch == 1)
