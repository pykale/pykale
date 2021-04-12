"""
Construct a dataset with (multiple) source and target domains, from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/multisource.py
"""

import logging
from enum import Enum
from typing import Dict

import numpy as np
import torch.utils.data
from sklearn.utils import check_random_state

from kale.loaddata.dataset_access import DatasetAccess
from kale.loaddata.sampler import get_labels, MultiDataLoader, SamplingConfig


class WeightingType(Enum):
    NATURAL = "natural"
    BALANCED = "balanced"
    PRESET0 = "preset0"


class DatasetSizeType(Enum):
    Max = "max"  # size of the biggest dataset
    Source = "source"  # size of the source dataset

    @staticmethod
    def get_size(size_type, source_dataset, *other_datasets):
        if size_type is DatasetSizeType.Max:
            return max(list(map(len, other_datasets)) + [len(source_dataset)])
        elif size_type is DatasetSizeType.Source:
            return len(source_dataset)
        else:
            raise ValueError(f"Size type size must be 'max' or 'source', had '{size_type}'")


class DomainsDatasetBase:
    def prepare_data_loaders(self):
        """
        handles train/validation/test split to have 3 datasets each with data from all domains
        """
        raise NotImplementedError()

    def get_domain_loaders(self, split="train", batch_size=32):
        """
        handles the sampling of a dataset containing multiple domains

        Args:
            split (string, optional): ["train"|"valid"|"test"]. Which dataset to iterate on. Defaults to "train".
            batch_size (int, optional): Defaults to 32.

        Returns:
            MultiDataLoader: A dataloader with API similar to the torch.dataloader, but returning
            batches from several domains at each iteration.
        """
        raise NotImplementedError()


class MultiDomainDatasets(DomainsDatasetBase):
    def __init__(
        self,
        source_access: DatasetAccess,
        target_access: DatasetAccess,
        config_weight_type="natural",
        config_size_type=DatasetSizeType.Max,
        val_split_ratio=0.1,
        source_sampling_config=None,
        target_sampling_config=None,
        n_fewshot=None,
        random_state=None,
    ):
        """The class controlling how the source and target domains are
            iterated over.

        Args:
            source_access (DatasetAccess): accessor for the source dataset
            target_access (DatasetAccess): accessor for the target dataset
            val_split_ratio (float, optional): ratio for the validation part of the train dataset. Defaults to 0.1.
            source_sampling_config (SamplingConfig, optional): How to sample from the source. Defaults to None (=> RandomSampler).
            target_sampling_config (SamplingConfig, optional): How to sample from the target. Defaults to None (=> RandomSampler).
            size_type (DatasetSizeType, optional): Which dataset size to use to define the number of epochs vs batch_size. Defaults to DatasetSizeType.Max.
            n_fewshot (int, optional): Number of target samples for which the label may be used,
                to define the few-shot, semi-supervised setting. Defaults to None.
            random_state ([int|np.random.RandomState], optional): Used for deterministic sampling/few-shot label selection. Defaults to None.
        Examples::
            >>> dataset = MultiDomainDatasets(source, target)
        """
        weight_type = WeightingType(config_weight_type)
        size_type = DatasetSizeType(config_size_type)

        if weight_type is WeightingType.PRESET0:
            self._source_sampling_config = SamplingConfig(class_weights=np.arange(source_access.n_classes(), 0, -1))
            self._target_sampling_config = SamplingConfig(
                # class_weights=random_state.randint(1, 4, size=target_access.n_classes())
                class_weights=np.random.randint(1, 4, size=target_access.n_classes())
            )
        elif weight_type is WeightingType.BALANCED:
            self._source_sampling_config = SamplingConfig(balance=True)
            self._target_sampling_config = SamplingConfig(balance=True)
        elif weight_type not in WeightingType:
            raise ValueError(f"Unknown weighting method {weight_type}.")
        else:
            self._source_sampling_config = SamplingConfig()
            self._target_sampling_config = SamplingConfig()

        self._source_access = source_access
        self._target_access = target_access
        self._val_split_ratio = val_split_ratio
        # self._source_sampling_config = (
        #     source_sampling_config
        #     if source_sampling_config is not None
        #     else SamplingConfig()
        # )
        # self._target_sampling_config = (
        #     target_sampling_config
        #     if target_sampling_config is not None
        #     else SamplingConfig()
        # )
        self._size_type = size_type
        self._n_fewshot = n_fewshot
        self._random_state = check_random_state(random_state)
        self._source_by_split: Dict[str, torch.utils.data.Subset] = {}
        self._labeled_target_by_split = None
        self._target_by_split: Dict[str, torch.utils.data.Subset] = {}

    def is_semi_supervised(self):
        return self._n_fewshot is not None and self._n_fewshot > 0

    def prepare_data_loaders(self):
        logging.debug("Load source")
        (self._source_by_split["train"], self._source_by_split["valid"],) = self._source_access.get_train_val(
            self._val_split_ratio
        )

        logging.debug("Load target")
        (self._target_by_split["train"], self._target_by_split["valid"],) = self._target_access.get_train_val(
            self._val_split_ratio
        )

        logging.debug("Load source Test")
        self._source_by_split["test"] = self._source_access.get_test()
        logging.debug("Load target Test")
        self._target_by_split["test"] = self._target_access.get_test()

        if self._n_fewshot is not None and self._n_fewshot > 0:
            # semi-supervised target domain
            self._labeled_target_by_split = {}
            for part in ["train", "valid", "test"]:
                (self._labeled_target_by_split[part], self._target_by_split[part],) = _split_dataset_few_shot(
                    self._target_by_split[part], self._n_fewshot
                )

    def get_domain_loaders(self, split="train", batch_size=32):
        source_ds = self._source_by_split[split]
        source_loader = self._source_sampling_config.create_loader(source_ds, batch_size)
        target_ds = self._target_by_split[split]

        if self._labeled_target_by_split is None:
            # unsupervised target domain
            target_loader = self._target_sampling_config.create_loader(target_ds, batch_size)
            n_dataset = DatasetSizeType.get_size(self._size_type, source_ds, target_ds)
            return MultiDataLoader(
                dataloaders=[source_loader, target_loader], n_batches=max(n_dataset // batch_size, 1),
            )
        else:
            # semi-supervised target domain
            target_labeled_ds = self._labeled_target_by_split[split]
            target_unlabeled_ds = target_ds
            # label domain: always balanced
            target_labeled_loader = SamplingConfig(balance=True, class_weights=None).create_loader(
                target_labeled_ds, batch_size=min(len(target_labeled_ds), batch_size)
            )
            target_unlabeled_loader = self._target_sampling_config.create_loader(target_unlabeled_ds, batch_size)
            n_dataset = DatasetSizeType.get_size(self._size_type, source_ds, target_labeled_ds, target_unlabeled_ds)
            return MultiDataLoader(
                dataloaders=[source_loader, target_labeled_loader, target_unlabeled_loader],
                n_batches=max(n_dataset // batch_size, 1),
            )

    def __len__(self):
        source_ds = self._source_by_split["train"]
        target_ds = self._target_by_split["train"]
        if self._labeled_target_by_split is None:
            return DatasetSizeType.get_size(self._size_type, source_ds, target_ds)
        else:
            labeled_target_ds = self._labeled_target_by_split["train"]
            return DatasetSizeType.get_size(self._size_type, source_ds, labeled_target_ds, target_ds)


def _split_dataset_few_shot(dataset, n_fewshot, random_state=None):
    if n_fewshot <= 0:
        raise ValueError(f"n_fewshot should be > 0, not '{n_fewshot}'")
    assert n_fewshot > 0
    labels = get_labels(dataset)
    classes = sorted(set(labels))
    if n_fewshot < 1:
        max_few = len(dataset) // len(classes)
        n_fewshot = round(max_few * n_fewshot)
    n_fewshot = int(round(n_fewshot))

    random_state = check_random_state(random_state)
    # sample n_fewshot items per class from last dataset
    tindices = []
    uindices = []
    for class_ in classes:
        indices = np.where(labels == class_)[0]
        random_state.shuffle(indices)
        head, tail = np.split(indices, [n_fewshot])
        assert len(head) == n_fewshot
        tindices.append(head)
        uindices.append(tail)
    tindices = np.concatenate(tindices)
    uindices = np.concatenate(uindices)
    assert len(tindices) == len(classes) * n_fewshot
    labeled_dataset = torch.utils.data.Subset(dataset, tindices)
    unlabeled_dataset = torch.utils.data.Subset(dataset, uindices)
    return labeled_dataset, unlabeled_dataset
