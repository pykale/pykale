import torch
import torch.utils.data

# From https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/dataset_access.py
class DatasetSplit:
    """
    This class ensures a unique API is used to access training, validation and test splits
    of any dataset.
    """

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def n_classes(self):
        return self._n_classes

    def get_train(self):
        """
        returns: a torch.utils.data.Dataset
        """
        raise NotImplementedError()

    def get_train_val(self, val_ratio):
        train_dataset = self.get_train()
        ntotal = len(train_dataset)
        ntrain = int((1 - val_ratio) * ntotal)
        torch.manual_seed(torch.initial_seed())
        return torch.utils.data.random_split(train_dataset, [ntrain, ntotal - ntrain])

    def get_test(self):
        raise NotImplementedError()
