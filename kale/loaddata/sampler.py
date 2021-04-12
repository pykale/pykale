"""Various sampling strategies for datasets to construct dataloader,
from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/sampler.py
"""

import logging

import numpy as np
import torch.utils.data
import torchvision
from torch.utils.data.sampler import BatchSampler, RandomSampler


class SamplingConfig:
    def __init__(self, balance=False, class_weights=None):
        if balance and class_weights is not None:
            raise ValueError("Params 'balance' and 'weights' are incompatible")
        self._balance = balance
        self._class_weights = class_weights

    def create_loader(self, dataset, batch_size):
        """Create the data loader

        Reference: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int): how many samples per batch to load
        """
        if self._balance:
            sampler = BalancedBatchSampler(dataset, batch_size=batch_size)
        elif self._class_weights is not None:
            sampler = ReweightedBatchSampler(dataset, batch_size=batch_size, class_weights=self._class_weights)
        else:
            if len(dataset) < batch_size:
                sub_sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size)
            else:
                sub_sampler = RandomSampler(dataset)
            sampler = BatchSampler(sub_sampler, batch_size=batch_size, drop_last=True)
        return torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)


# TODO: deterministic shuffle?
class MultiDataLoader:
    """
    Batch Sampler for a MultiDataset. Iterates in parallel over different batch samplers for each dataset.
    Yields batches [(x_1, y_1), ..., (x_s, y_s)] for s datasets.
    """

    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                logging.debug(f"reinit loader {di} of type {type(dl)}")
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, dataset, batch_size):
        labels = get_labels(dataset)
        classes = sorted(set(labels))

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")

        self._class_iters = [InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) for class_ in classes]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {batch_size}")
        logging.debug("K=", n_classes, "nk=", self._n_samples)
        logging.debug("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class ReweightedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples batch_size according to given input distribution
    assuming multi-class labels
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    # /!\ 'class_weights' should be provided in the "natural order" of the classes (i.e. sorted(classes)) /!\
    def __init__(self, dataset, batch_size, class_weights):
        labels = get_labels(dataset)
        self._classes = sorted(set(labels))

        n_classes = len(self._classes)
        if n_classes > len(class_weights):
            k = len(class_weights)
            sum_w = np.sum(class_weights)
            if sum_w >= 1:
                # normalize attributing equal weight to weighted part and remaining part
                class_weights /= sum_w * k / n_classes + (n_classes - k) / n_classes
            krem = k - n_classes
            wrem = 1 - sum_w
            logging.warning(f"will assume uniform distribution for labels > {len(class_weights)}")
            self._class_weights = np.ones(n_classes, dtype=np.float)
            self._class_weights[:k] = class_weights
            self._class_weights[k:] = wrem / krem
        else:
            self._class_weights = class_weights[:n_classes]

        if np.sum(self._class_weights) != 1:
            self._class_weights = self._class_weights / np.sum(self._class_weights)

        logging.debug("Using weights=", self._class_weights)
        if batch_size == 0:
            raise ValueError(f"batch_size should be bigger than the number of classes, got {batch_size}")

        self._class_to_iter = {
            class_: InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_) for class_ in self._classes
        }

        self.n_dataset = len(labels)
        self._batch_size = batch_size
        self._n_batches = self.n_dataset // self._batch_size
        if self._n_batches == 0:
            raise ValueError(f"Dataset is not big enough to generate batches with size {self._batch_size}")
        logging.debug("K=", n_classes, "nk=", self._batch_size)
        logging.debug("Batch size = ", self._batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            # sample batch_size classes
            class_idx = np.random.choice(self._classes, p=self._class_weights, replace=True, size=self._batch_size,)
            indices = []
            for class_, num in zip(*np.unique(class_idx, return_counts=True)):
                indices.extend(self._class_to_iter[class_].get(num))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_to_iter.values():
            class_iter.reset()

    def __len__(self):
        return self._n_batches


def get_labels(dataset):
    """
    Get class labels for dataset
    """
    dataset_type = type(dataset)
    if dataset_type is torchvision.datasets.SVHN:
        return dataset.labels
    if dataset_type is torchvision.datasets.ImageFolder:
        return dataset.imgs[:][1]

    # Handle subset, recurses into non-subset version
    if dataset_type is torch.utils.data.Subset:
        indices = dataset.indices
        all_labels = get_labels(dataset.dataset)
        logging.debug(f"data subset of len {len(indices)} from {len(all_labels)}")
        labels = all_labels[indices]
        if isinstance(labels, torch.Tensor):
            return labels.numpy()
        return labels

    try:
        logging.debug(dataset.targets.shape, type(dataset.targets))
        if isinstance(dataset.targets, torch.Tensor):
            return dataset.targets.numpy()
        return dataset.targets
    except AttributeError:
        logging.error(type(dataset))


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            logging.debug(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i : self.i]
