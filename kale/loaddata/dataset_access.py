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

    def get_train_valid(self, valid_ratio):
        """
        Randomly split a dataset into non-overlapping training and validation datasets.

        Args:
            valid_ratio (float): the ratio for validation set

        Returns:
            Dataset: a torch.utils.data.Dataset
        """
        train_dataset = self.get_train()

        if valid_ratio == 0:
            return train_dataset, train_dataset
        else:
            return split_by_ratios(train_dataset, [valid_ratio])

    def get_test(self):
        raise NotImplementedError()


def get_class_subset(dataset, class_ids):
    """
    Args:
        dataset: a torch.utils.data.Dataset
        class_ids (list, optional): List of chosen subset of class ids.
    Returns: a torch.utils.data.Dataset
        Dataset: a torch.utils.data.Dataset with only classes in class_ids
    """
    sub_indices = [i for i in range(0, len(dataset)) if dataset[i][1] in class_ids]
    return torch.utils.data.Subset(dataset, sub_indices)


def split_by_ratios(dataset, split_ratios):
    """Randomly split a dataset into non-overlapping new datasets of given ratios.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be split.
        split_ratios (list): Ratios of splits to be produced, where 0 < sum(split_ratios) <= 1.

    Returns:
        [List]: A list of subsets.

    Examples:
        >>> import torch
        >>> from kale.loaddata.dataset_access import split_by_ratios
        >>> subset1, subset2 = split_by_ratios(range(10), [0.3, 0.7])
        >>> len(subset1)
        3
        >>> len(subset2)
        7
        >>> subset1, subset2 = split_by_ratios(range(10), [0.3])
        >>> len(subset1)
        3
        >>> len(subset2)
        7
        >>> subset1, subset2, subset3 = split_by_ratios(range(10), [0.3, 0.3])
        >>> len(subset1)
        3
        >>> len(subset2)
        3
        >>> len(subset3)
        4
    """
    n_total = len(dataset)
    ratio_sum = sum(split_ratios)
    if ratio_sum > 1 or ratio_sum <= 0:
        raise ValueError("The sum of ratios should be in range(0, 1]")
    elif ratio_sum == 1:
        split_ratios_ = split_ratios[:-1]
    else:
        split_ratios_ = split_ratios.copy()
    split_sizes = [int(n_total * ratio_) for ratio_ in split_ratios_]
    split_sizes.append(n_total - sum(split_sizes))

    return torch.utils.data.random_split(dataset, split_sizes)
