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

    def get_class_subsampled_train(self, class_ids=None):
        """
        Returns: a torch.utils.data.Dataset
            Dataset: a torch.utils.data.Dataset with only classes in class_ids
        """
        train_dataset = self.get_train()
        if class_ids is None:
            return train_dataset
        else:
            sub_indices = [i for i in range(0, len(train_dataset)) if train_dataset[i][1] in class_ids]
            return torch.utils.data.Subset(train_dataset, sub_indices)

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

    def get_class_subsampled_test(self, class_ids):
        """
        Returns: a torch.utils.data.Dataset
            Dataset: a torch.utils.data.Dataset with only classes in class_ids
        """
        test_dataset = self.get_test()
        if class_ids is None:
            return test_dataset
        else:
            sub_indices = [i for i in range(0, len(test_dataset)) if test_dataset[i][1] in class_ids]
            return torch.utils.data.Subset(test_dataset, sub_indices)
