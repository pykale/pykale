"""
Dataset setting and data loader for MNIST-M, from
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/dataset_mnistm.py
(based on https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py)
CREDIT: https://github.com/corenel
"""

from __future__ import print_function

import errno
import logging
import os

import torch
import torch.utils.data as data
from PIL import Image


class MNISTM(data.Dataset):
    """
    MNIST-M Dataset.
    Auto-downloads the dataset and provide the torch Dataset API.

    Args:
        root (str): path to directory where the MNISTM folder will be created (or exists.)
        train (bool, optional): defaults to True.
            If True, loads the training data. Otherwise, loads the test data.
        transform (callable, optional): defaults to None.
            A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g., ``transforms.RandomCrop``
            This preprocessing function applied to all images (whether source or target)

        target_transform (callable, optional): default to None, similar to transform.
            This preprocessing function applied to all target images, after `transform`

        download (bool optional): defaults to False.
            Whether to allow downloading the data if not found on disk.
    """

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False,
    ):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.join(root, "MNISTM")
        self.mnist_root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        if self.train:
            self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.data, self.targets = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNISTM data."""
        # import essential packages
        import gzip
        import pickle

        from six.moves import urllib
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        logging.info("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        logging.info("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root, train=True, download=True).targets
        mnist_test_labels = datasets.MNIST(root=self.mnist_root, train=False, download=True).targets

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        logging.info("[DONE]")
