"""Digits dataset loading for MNIST, SVHN, MNIST-M (modified MNIST), and USPS
Modified by Haiping Lu from
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

from enum import Enum
from torchvision.datasets import MNIST, SVHN
import kale.prepdata.im_transform as im_transform
from kale.loaddata.usps import USPS
from kale.loaddata.mnistm import MNISTM
from kale.loaddata.splits import DatasetAccess


class DigitDataset(Enum):
    MNIST = "MNIST"
    MNISTM = "MNISTM"
    USPS = "USPS"
    SVHN = "SVHN"

    @staticmethod
    def get_accesses(source: "DigitDataset", target: "DigitDataset", data_path):
        channel_numbers = {
            DigitDataset.MNIST: 1,
            DigitDataset.MNISTM: 3,
            DigitDataset.USPS: 1,
            DigitDataset.SVHN: 3,
        }

        transform_names = {
            (DigitDataset.MNIST, 1): "mnist32",
            (DigitDataset.MNIST, 3): "mnist32rgb",
            (DigitDataset.MNISTM, 3): "mnistm",
            (DigitDataset.USPS, 1): "usps32",
            (DigitDataset.USPS, 3): "usps32rgb",
            (DigitDataset.SVHN, 3): "svhn",
        }

        factories = {
            DigitDataset.MNIST: MNISTDatasetAccess,
            DigitDataset.MNISTM: MNISTMDatasetAccess,
            DigitDataset.USPS: USPSDatasetAccess,
            DigitDataset.SVHN: SVHNDatasetAccess,
        }

        # handle color/nb channels
        num_channels = max(channel_numbers[source], channel_numbers[target])
        source_tf = transform_names[(source, num_channels)]
        target_tf = transform_names[(target, num_channels)]

        return (
            factories[source](data_path, source_tf),
            factories[target](data_path, target_tf),
            num_channels,
        )


class DigitDatasetAccess(DatasetAccess):
    def __init__(self, data_path, transform_kind):
        super().__init__(n_classes=10)
        self._data_path = data_path
        self._transform = im_transform.get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNIST(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNIST(
            self._data_path, train=False, transform=self._transform, download=True
        )


class MNISTMDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return MNISTM(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return MNISTM(
            self._data_path, train=False, transform=self._transform, download=True
        )


class USPSDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return USPS(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        return USPS(
            self._data_path, train=False, transform=self._transform, download=True
        )


class SVHNDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return SVHN(
            self._data_path, split="train", transform=self._transform, download=True
        )

    def get_test(self):
        return SVHN(
            self._data_path, split="test", transform=self._transform, download=True
        )
