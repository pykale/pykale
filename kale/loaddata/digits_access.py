"""
Digits dataset (source and target domain) loading for MNIST, SVHN, MNIST-M (modified MNIST), and USPS.
The code is based on
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/digits_dataset_access.py
"""

from enum import Enum

from torchvision.datasets import MNIST, SVHN

import kale.prepdata.image_transform as image_transform
from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.mnistm import MNISTM
from kale.loaddata.usps import USPS


class DigitDataset(Enum):
    MNIST = "MNIST"
    MNIST_RGB = "MNIST_RGB"
    MNISTM = "MNISTM"
    USPS = "USPS"
    USPS_RGB = "USPS_RGB"
    SVHN = "SVHN"

    @staticmethod
    def get_channel_numbers(dataset: "DigitDataset"):
        channel_numbers = {
            DigitDataset.MNIST: 1,
            DigitDataset.MNIST_RGB: 3,
            DigitDataset.MNISTM: 3,
            DigitDataset.USPS: 1,
            DigitDataset.USPS_RGB: 3,
            DigitDataset.SVHN: 3,
        }
        return channel_numbers[dataset]

    @staticmethod
    def get_digit_transform(dataset: "DigitDataset", n_channels):
        transform_names = {
            (DigitDataset.MNIST, 1): "mnist32",
            (DigitDataset.MNIST, 3): "mnist32rgb",
            (DigitDataset.MNIST_RGB, 3): "mnist32rgb",
            (DigitDataset.MNISTM, 3): "mnistm",
            (DigitDataset.USPS, 1): "usps32",
            (DigitDataset.USPS, 3): "usps32rgb",
            (DigitDataset.USPS_RGB, 3): "usps32rgb",
            (DigitDataset.SVHN, 3): "svhn",
        }

        return transform_names[(dataset, n_channels)]

    @staticmethod
    def get_access(dataset: "DigitDataset", data_path, num_channels=None):
        """Gets data loaders for digit datasets

        Args:
            dataset (DigitDataset): dataset name
            data_path (string): root directory of dataset
            num_channels (int): number of channels, defaults to None

        Examples::
            >>> data_access, num_channel = DigitDataset.get_access(dataset, data_path)
        """

        factories = {
            DigitDataset.MNIST: MNISTDatasetAccess,
            DigitDataset.MNIST_RGB: MNISTDatasetAccess,
            DigitDataset.MNISTM: MNISTMDatasetAccess,
            DigitDataset.USPS: USPSDatasetAccess,
            DigitDataset.USPS_RGB: USPSDatasetAccess,
            DigitDataset.SVHN: SVHNDatasetAccess,
        }
        if num_channels is None:
            num_channels = DigitDataset.get_channel_numbers(dataset)
        tf = DigitDataset.get_digit_transform(dataset, num_channels)

        return factories[dataset](data_path, tf), num_channels

    # Originally get_access
    @staticmethod
    def get_source_target(source: "DigitDataset", target: "DigitDataset", data_path):
        """Gets data loaders for source and target datasets

        Args:
            source (DigitDataset): source dataset name
            target (DigitDataset): target dataset name
            data_path (string): root directory of dataset

        Examples::
            >>> source_access, target_access, num_channel = DigitDataset.get_source_target(source, target, data_path)
        """
        src_n_channels = DigitDataset.get_channel_numbers(source)
        tgt_n_channels = DigitDataset.get_channel_numbers(target)
        num_channels = max(src_n_channels, tgt_n_channels)
        src_access, src_n_channels = DigitDataset.get_access(source, data_path, num_channels)
        tgt_access, tgt_n_channels = DigitDataset.get_access(target, data_path, num_channels)

        return src_access, tgt_access, num_channels


class DigitDatasetAccess(DatasetAccess):
    """Common API for digit dataset access

    Args:
        data_path (string): root directory of dataset
        transform_kind (string): types of image transforms
    """

    def __init__(self, data_path, transform_kind):
        super().__init__(n_classes=10)
        self._data_path = data_path
        self._transform = image_transform.get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    """
    MNIST data loader
    """

    def get_train(self):
        return MNIST(self._data_path, train=True, transform=self._transform, download=True)

    def get_test(self):
        return MNIST(self._data_path, train=False, transform=self._transform, download=True)


class MNISTMDatasetAccess(DigitDatasetAccess):
    """
    Modified MNIST (MNISTM) data loader
    """

    def get_train(self):
        return MNISTM(self._data_path, train=True, transform=self._transform, download=True)

    def get_test(self):
        return MNISTM(self._data_path, train=False, transform=self._transform, download=True)


class USPSDatasetAccess(DigitDatasetAccess):
    """
    USPS data loader
    """

    def get_train(self):
        return USPS(self._data_path, train=True, transform=self._transform, download=True)

    def get_test(self):
        return USPS(self._data_path, train=False, transform=self._transform, download=True)


class SVHNDatasetAccess(DigitDatasetAccess):
    """
    SVHN data loader
    """

    def get_train(self):
        return SVHN(self._data_path, split="train", transform=self._transform, download=True)

    def get_test(self):
        return SVHN(self._data_path, split="test", transform=self._transform, download=True)
