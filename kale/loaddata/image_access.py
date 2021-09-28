import logging
import os
from enum import Enum

import numpy as np
import pydicom
import torch
from torchvision import datasets, transforms

from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.mnistm import MNISTM
from kale.loaddata.multi_domain import MultiDomainAccess, MultiDomainImageFolder
from kale.loaddata.usps import USPS
from kale.prepdata.image_transform import get_transform
from kale.utils.download import download_file_by_url


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
        self._transform = get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    """
    MNIST data loader
    """

    def get_train(self):
        return datasets.MNIST(self._data_path, train=True, transform=self._transform, download=True)

    def get_test(self):
        return datasets.MNIST(self._data_path, train=False, transform=self._transform, download=True)


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
        return datasets.SVHN(self._data_path, split="train", transform=self._transform, download=True)

    def get_test(self):
        return datasets.SVHN(self._data_path, split="test", transform=self._transform, download=True)


OFFICE_DOMAINS = ["amazon", "caltech", "dslr", "webcam"]
office_transform = get_transform("office")


class OfficeAccess(MultiDomainImageFolder, DatasetAccess):
    """Common API for office dataset access

    Args:
        root (string): root directory of dataset
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed
            version. Defaults to office_transform.
        download (bool, optional): Whether to allow downloading the data if not found on disk. Defaults to False.

    References:
        [1] Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to
        new domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
        [2] Griffin, Gregory and Holub, Alex and Perona, Pietro, 2007. Caltech-256 Object Category Dataset.
        California Institute of Technology. (Unpublished).
        https://resolver.caltech.edu/CaltechAUTHORS:CNS-TR-2007-001.
        [3] Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012, June. Geodesic flow kernel for unsupervised
        domain adaptation. In IEEE Conference on Computer Vision and Pattern Recognition (pp. 2066-2073).
    """

    def __init__(self, root, transform=office_transform, download=False, **kwargs):
        if download:
            self.download(root)
        super(OfficeAccess, self).__init__(root, transform=transform, **kwargs)

    @staticmethod
    def download(path):
        """Download dataset.
            Office-31 source: https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code
            Caltech-256 source: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
            Data with this library is adapted from: http://www.stat.ucla.edu/~jxie/iFRAME/code/imageClassification.rar
        """
        url = "https://github.com/pykale/data/raw/main/images/office/"

        if not os.path.exists(path):
            os.makedirs(path)
        for domain_ in OFFICE_DOMAINS:
            filename = "%s.zip" % domain_
            data_path = os.path.join(path, filename)
            if os.path.exists(data_path):
                logging.info(f"Data file {filename} already exists.")
                continue
            else:
                data_url = "%s/%s" % (url, filename)
                download_file_by_url(data_url, path, filename, "zip")
                logging.info(f"Download {data_url} to {data_path}")

        logging.info("[DONE]")
        return


class Office31(OfficeAccess):
    def __init__(self, root, **kwargs):
        """Office-31 Dataset. Consists of three domains: 'amazon', 'dslr', and 'webcam', with 31 image classes.

        Args:
            root (string): path to directory where the office folder will be created (or exists).

        Reference:
            Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to new
            domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
        """
        sub_domain_set = ["amazon", "dslr", "webcam"]
        super(Office31, self).__init__(root, sub_domain_set=sub_domain_set, **kwargs)


class OfficeCaltech(OfficeAccess):
    def __init__(self, root, **kwargs):
        """Office-Caltech-10 Dataset. This dataset consists of four domains: 'amazon', 'caltech', 'dslr', and 'webcam',
            which are samples with overlapped 10 classes between Office-31 and Caltech-256.

        Args:
            root (string): path to directory where the office folder will be created (or exists).

        References:
            [1] Saenko, K., Kulis, B., Fritz, M. and Darrell, T., 2010, September. Adapting visual category models to
            new domains. In European Conference on Computer Vision (pp. 213-226). Springer, Berlin, Heidelberg.
            [2] Griffin, Gregory and Holub, Alex and Perona, Pietro, 2007. Caltech-256 Object Category Dataset.
            California Institute of Technology. (Unpublished).
            https://resolver.caltech.edu/CaltechAUTHORS:CNS-TR-2007-001.
            [3] Gong, B., Shi, Y., Sha, F. and Grauman, K., 2012, June. Geodesic flow kernel for unsupervised
            domain adaptation. In IEEE Conference on Computer Vision and Pattern Recognition (pp. 2066-2073).
        """
        sub_class_set = [
            "mouse",
            "calculator",
            "back_pack",
            "keyboard",
            "monitor",
            "projector",
            "headphones",
            "bike",
            "laptop_computer",
            "mug",
        ]
        super(OfficeCaltech, self).__init__(root, sub_class_set=sub_class_set, **kwargs)


class ImageAccess:
    @staticmethod
    def get_multi_domain_images(image_set_name: str, data_path: str, sub_domain_set=None, **kwargs):
        """Get multi-domain images as a dataset from the given data path.

        Args:
            image_set_name (str): name of image dataset
            data_path (str): path to the image dataset
            sub_domain_set (list, optional): A list of domain names, which should be a subset of domains under the
                directory of data path. If None, all available domains will be used. Defaults to None.

        Returns:
            [MultiDomainImageFolder, or MultiDomainAccess]: Multi-domain image dataset
        """
        image_set_name = image_set_name.upper()
        if image_set_name == "OFFICE_CALTECH":
            return OfficeCaltech(data_path, **kwargs)
        elif image_set_name == "OFFICE31":
            return Office31(data_path, **kwargs)
        elif image_set_name == "OFFICE":
            return OfficeAccess(data_path, sub_domain_set=sub_domain_set, **kwargs)
        elif image_set_name == "DIGITS":
            data_dict = dict()
            if sub_domain_set is None:
                sub_domain_set = ["SVHN", "USPS_RGB", "MNIST_RGB", "MNISTM"]
            for domain in sub_domain_set:
                data_dict[domain] = DigitDataset.get_access(DigitDataset(domain), data_path)[0]
            return MultiDomainAccess(data_dict, 10, **kwargs)
        else:
            # default image transform
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
            return MultiDomainImageFolder(data_path, transform=transform, sub_domain_set=sub_domain_set, **kwargs)


def get_cifar(cfg):
    """Gets training and validation data loaders for the CIFAR datasets

    Args:
        cfg: A YACS config object.
    """
    logging.info("==> Preparing to load data " + cfg.DATASET.NAME + " at " + cfg.DATASET.ROOT)
    cifar_train_transform = get_transform("cifar", augment=True)
    cifar_test_transform = get_transform("cifar", augment=False)

    if cfg.DATASET.NAME == "CIFAR10":
        train_set = datasets.CIFAR10(cfg.DATASET.ROOT, train=True, download=True, transform=cifar_train_transform)
        val_set = datasets.CIFAR10(cfg.DATASET.ROOT, train=False, download=True, transform=cifar_test_transform)
    elif cfg.DATASET.NAME == "CIFAR100":
        train_set = datasets.CIFAR100(cfg.DATASET.ROOT, train=True, download=True, transform=cifar_train_transform)
        val_set = datasets.CIFAR100(cfg.DATASET.ROOT, train=False, download=True, transform=cifar_test_transform)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg.SOLVER.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def read_dicom_images(dicom_path, sort_instance=True, sort_patient=False):
    """Read dicom images for multiple patients and multiple instances/phases.

    Args:
        dicom_path (str): Path to DICOM images.
        sort_instance (bool, optional): Whether sort images by InstanceNumber (i.e. phase number) for each subject.
            Defaults to True.
        sort_patient (bool, optional): Whether sort subjects' images by PatientID. Defaults to False.

    Returns:
        [array-like]: [description]
    """
    sub_dirs = os.listdir(dicom_path)
    all_ds = []
    sub_ids = []
    for sub_dir in sub_dirs:
        sub_ds = []
        sub_path = os.path.join(dicom_path, sub_dir)
        phase_files = os.listdir(sub_path)
        for phase_file in phase_files:
            dataset = pydicom.dcmread(os.path.join(sub_path, phase_file))
            sub_ds.append(dataset)
        if sort_instance:
            sub_ds.sort(key=lambda x: x.InstanceNumber, reverse=False)
        sub_ids.append(int(sub_ds[0].PatientID))
        all_ds.append(sub_ds)

    if sort_patient:
        all_ds.sort(key=lambda x: int(x[0].PatientID), reverse=False)

    n_sub = len(all_ds)
    n_phase = len(all_ds[0])
    img_shape = all_ds[0][0].pixel_array.shape
    images = np.zeros((n_sub, n_phase,) + img_shape)
    for i in range(n_sub):
        for j in range(n_phase):
            images[i, j, ...] = all_ds[i][j].pixel_array

    return images
