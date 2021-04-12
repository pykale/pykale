"""
Dataset Access API adapted from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/dataset_access.py
"""

import torch
import torch.utils.data


class DatasetAccess:
    """
    This class ensures a unique API is used to access training, validation and test splits
    of any dataset.

    Args:
        n_classes (int): the number of classes.
    """

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def n_classes(self):
        return self._n_classes

    def get_train(self):
        """
        Returns: a torch.utils.data.Dataset
            Dataset: a torch.utils.data.Dataset
        """
        raise NotImplementedError()

    def get_train_val(self, val_ratio):
        """
        Randomly split a dataset into non-overlapping training and validation datasets.

        Args:
            val_ratio (float): the ratio for validation set

        Returns:
            Dataset: a torch.utils.data.Dataset
        """
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - val_ratio) * ntotal)

        return torch.utils.data.random_split(train_dataset, [ntrain, ntotal - ntrain])

    def get_test(self):
        raise NotImplementedError()
