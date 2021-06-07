"""
Construct a dataset with (multiple) source and target domains,
from https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/datasets/multisource.py
"""

import logging
import os
from enum import Enum
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import numpy as np
import torch.utils.data
from sklearn.utils import check_random_state
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension, IMG_EXTENSIONS

from kale.loaddata.dataset_access import DatasetAccess, split_by_ratios
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
            config_weight_type (WeightingType, optional): The weight type for sampling. Defaults to 'natural'.
            config_size_type (DatasetSizeType, optional): Which dataset size to use to define the number of epochs vs
                batch_size. Defaults to DatasetSizeType.Max.
            val_split_ratio (float, optional): ratio for the validation part of the train dataset. Defaults to 0.1.
            source_sampling_config (SamplingConfig, optional): How to sample from the source. Defaults to None
                (=> RandomSampler).
            target_sampling_config (SamplingConfig, optional): How to sample from the target. Defaults to None
                (=> RandomSampler).
            n_fewshot (int, optional): Number of target samples for which the label may be used,
                to define the few-shot, semi-supervised setting. Defaults to None.
            random_state ([int|np.random.RandomState], optional): Used for deterministic sampling/few-shot label
                selection. Defaults to None.
        Examples::
            >>> dataset = MultiDomainDatasets(source_access, target_access)
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


class MultiDomainImageFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

            root/domain_a/class_x/xxx.ext
            root/domain_a/class_x/xxy.ext
            ...
            root/domain_a/class_y/xxz.ext
            ...
            root/domain_k/class_x/123.ext
            root/domain_k/class_x/abc3.ext
            ...
            root/domain_k/class_y/asd932_.ext

        Args:
            root (string): Root directory path.
            loader (callable): A function to load a sample given its path.
            extensions (tuple[string]): A list of allowed extensions.
                both extensions and is_valid_file should not be passed.
            transform (callable, optional): A function/transform that takes in
                a sample and returns a transformed version.
                E.g, ``transforms.RandomCrop`` for images.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it.
            is_valid_file (callable, optional): A function that takes path of a file
                and check if the file is a valid file (used to check of corrupt files)
                both extensions and is_valid_file should not be passed.

         Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
            domains (list): List of the domain names sorted alphabetically.
            domain_to_idx (dict): Dict with items (domain_name, domain_index).
            domain_labels (list): The domain_index value for each image in the dataset
        """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        return_domain_label: Optional[bool] = False,
    ) -> None:
        super(MultiDomainImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        domains, domain_to_idx = self._find_classes(self.root)
        classes, class_to_idx = self._find_classes(os.path.join(self.root, domains[0]))
        for domain in domains:
            domain_path = os.path.join(self.root, domain)
            classes_, class_to_idx_ = self._find_classes(domain_path)
            if not classes == classes_:
                raise ValueError("Classes for different domains are expected to be the same.")
        samples = make_multi_domain_set(self.root, class_to_idx, domain_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in sub-folders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.domains = domains
        self.domain_to_idx = domain_to_idx
        self.domain_labels = [s[2] for s in samples]
        self.return_domain_label = return_domain_label

    @staticmethod
    def _find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
            Finds the class folders in a dataset.

            Args:
                directory (string): Directory path.

            Returns:
                tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

            Ensures:
                No class is a subdirectory of another.
            """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple:
        """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target, domain) where target is class_index of the target class.
            """
        path, target, domain = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_domain_label:
            return sample, target, domain
        else:
            return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def make_multi_domain_set(
    directory: str,
    class_to_idx: Dict[str, int],
    domain_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int, int]]:
    """Generates a list of samples of a form (path_to_sample, class, domain).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        domain_to_idx (Dict[str, int]): dictionary mapping d name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int, int]]: samples of a form (path_to_sample, class, domain)
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_domain in sorted(domain_to_idx.keys()):
        domain_index = domain_to_idx[target_domain]
        domain_dir = os.path.join(directory, target_domain)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(domain_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index, domain_index
                        instances.append(item)
    return instances


class DomainWeightType(Enum):
    Rand = "random"
    BALANCED = "balanced"


class MultiDomainAdapDataset(DomainsDatasetBase):
    def __init__(
        self,
        data_access: MultiDomainImageFolder,
        # domain_weight_type="balanced",
        # config_weight_type="natural",
        # config_size_type=DatasetSizeType.Max,
        # target_label=0,
        # target=None,
        val_split_ratio=0.1,
        test_split_ratio=0.2,
        random_state=1,
    ):
        # domain_weight_type = DomainWeightType(domain_weight_type)
        # if domain_weight_type is DomainWeightType.BALANCED:
        #     sampling =
        # weight_type = WeightingType(config_weight_type)
        # size_type = DatasetSizeType(config_size_type)
        # if target is None:
        #     for key in data_access.domain_to_idx:
        #         if data_access.domain_to_idx[key] == target_label:
        #             target = key
        #             break
        # else:
        #     target_idx = data_access.domain_to_idx[target]
        #     if target_idx != target_label:
        #         raise ValueError(
        #             "Given target domain label in given dataset is %s but not %s" % (target_label, target_idx)
        #         )
        self.domain_labels = np.array(data_access.domain_labels)
        self.domain_to_idx = data_access.domain_to_idx
        self.n_domains = len(data_access.domain_to_idx)
        self.data_access = data_access
        # self._source_access = torch.utils.data.Subset(data_access, np.where(domain_labels != target_label)[0])
        # self._target_access = torch.utils.data.Subset(data_access, np.where(domain_labels == target_label)[0])
        # self.target_label = target_label
        # self.target = target
        self._val_split_ratio = val_split_ratio
        self._test_split_ratio = test_split_ratio
        self._sample_by_split: Dict[str, torch.utils.data.Subset] = {}
        self._sampling_config = SamplingConfig()
        self._random_state = check_random_state(random_state)

    def prepare_data_loaders(self):
        # (self._source_by_split["test"], self._source_by_split["valid"], self._source_by_split["train"],) = \
        #     split_by_ratios(self._source_access, [self._test_split_ratio, self._val_split_ratio])
        #
        # (self._target_by_split["test"], self._target_by_split["valid"], self._target_by_split["train"]) = \
        #     split_by_ratios(self._target_access, [self._test_split_ratio, self._val_split_ratio])
        subset_split = {"train": [], "valid": [], "test": []}
        for domain in self.domain_to_idx:
            subset = dict()
            domain_label_ = self.domain_to_idx[domain]
            domain_idx = np.where(self.domain_labels == domain_label_)[0]
            subset["test"], subset["valid"], subset["train"] = split_by_ratios(
                torch.utils.data.Subset(self.data_access, domain_idx), [self._test_split_ratio, self._val_split_ratio]
            )
            for split in ["train", "valid", "test"]:
                subset_split[split].append(subset[split])

        for split in ["train", "valid", "test"]:
            self._sample_by_split[split] = torch.utils.data.ConcatDataset(subset_split[split])

    def get_domain_loaders(self, split="train", batch_size=32):

        return self._sampling_config.create_loader(self._sample_by_split[split], batch_size)

    def __len__(self):

        return len(self.data_access)
