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

        return split_by_ratios(train_dataset, [val_ratio])

    def get_test(self):
        raise NotImplementedError()


def split_by_ratios(dataset, split_ratios, random_state=144):
    n_total = len(dataset)
    ratio_sum = sum(split_ratios)
    if ratio_sum > 1 or ratio_sum <= 0:
        raise ValueError("The sum of ratios should be in range(0, 1]")
    elif ratio_sum == 1:
        split_ratios_ = split_ratios[:-1]
    else:
        split_ratios_ = split_ratios.copy()
    split_lengths = [int(n_total * ratio_) for ratio_ in split_ratios_]
    split_lengths.append(n_total - sum(split_lengths))

    return torch.utils.data.random_split(dataset, split_lengths, generator=torch.Generator().manual_seed(random_state))
