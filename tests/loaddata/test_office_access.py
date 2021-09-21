from numpy import testing

from kale.loaddata.image_access import MultiDomainImageAccess, OfficeAccess
from kale.loaddata.multi_domain import MultiDomainAdapDataset, MultiDomainDatasets


def test_office31(office_path):
    office_access = MultiDomainImageAccess.get_image_access(
        "OFFICE31", office_path, download=True, return_domain_label=True
    )
    testing.assert_equal(len(office_access.class_to_idx), 31)
    testing.assert_equal(len(office_access.domain_to_idx), 3)
    dataset = MultiDomainAdapDataset(office_access)
    dataset.prepare_data_loaders()
    domain_labels = list(dataset.domain_to_idx.values())
    for split in ["train", "valid", "test"]:
        dataloader = dataset.get_domain_loaders(split=split)
        x, y, z = next(iter(dataloader))
        for domain_label_ in domain_labels:
            testing.assert_equal(y[z == domain_label_].shape[0], 10)


def test_office_caltech(office_path):
    office_access = MultiDomainImageAccess.get_image_access(
        "OFFICE_CALTECH", office_path, download=True, return_domain_label=True
    )
    testing.assert_equal(len(office_access.class_to_idx), 10)
    testing.assert_equal(len(office_access.domain_to_idx), 4)


def test_custom_office(office_path):
    source = OfficeAccess(root=office_path, download=True, sub_domain_set=["dslr"], split_train_test=True)
    target = OfficeAccess(root=office_path, download=True, sub_domain_set=["webcam"], split_train_test=True)
    dataset = MultiDomainDatasets(source_access=source, target_access=target)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders()
    testing.assert_equal(len(next(iter(dataloader))), 2)


def test_multi_domain_digits(download_path):
    data_access = MultiDomainImageAccess.get_image_access(
        "DIGITS", download_path, domain_names=["SVHN", "USPS_RGB"], return_domain_label=True
    )
    dataset = MultiDomainAdapDataset(data_access)
    dataset.prepare_data_loaders()
    dataloader = dataset.get_domain_loaders(split="test", batch_size=10)
    assert len(next(iter(dataloader))) == 3
