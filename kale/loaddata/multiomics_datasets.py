# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Construct a dataset with multiple omics modalities based on PyTorch Geometric.

This code is written by refactoring the MOGONET dataset code (https://github.com/txWang/MOGONET/blob/main/train_test.py)
within the 'Dataset' class provided in the PyTorch Geometric.

Reference:
Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K. (2021). MOGONET integrates multi-omics data
using graph convolutional networks allowing patient classification and biomarker identification. Nature communications.
https://www.nature.com/articles/s41467-021-23774-w
"""

import os.path as osp
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, download_url, extract_zip
from torch_sparse import SparseTensor

from kale.utils.distance import calculate_distance, DistanceMetric


class MultiomicsDataset(Dataset):
    r"""The multiomics data for creating graph dataset.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html>`__ in PyTorch Geometric
    for the accompanying tutorial.

    Args:
        root (string): Root directory where the dataset should be saved.
        num_modalities (int): The total number of modalities in the dataset.
        num_classes (int): The total number of classes in the dataset.
        url (string, optional): The url to download the dataset from.
        raw_file_names (list[callable], optional): The name of the files in the ``self.raw_dir`` folder that must be
            present in order to skip downloading.
        random_split (bool, optional): Whether to split the dataset into random train and test subsets. (default:
            ``False``)
        train_size (float, optional): The proportion of the dataset to include in the train split that should be between
            0.0 and 1.0. This parameter is used when ``random_split`` is ``True``.
        transform (callable, optional): A function/transform that takes in an array_like data and returns a transformed
            version. The data object will be transformed before every access. (default: ``None``)
        pre_transform (callable, optional): A function/transform that takes in an array_like data and returns a
            transformed version. The data object will be transformed before being saved to disk. (default: ``None``)
        target_pre_transform (callable, optional): A function/transform that takes in an array_like of labels and
            returns a transformed version. The label object will be transformed before being saved to disk. (default:
            ``None``)
    """

    def __init__(
        self,
        root: str,
        num_modalities: int,
        num_classes: int,
        url: str = None,
        raw_file_names: List[str] = None,
        random_split: bool = False,
        train_size: float = 0.7,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        target_pre_transform: Optional[Callable] = None,
    ) -> None:
        self._url = url
        self._raw_file_names = raw_file_names
        self._num_modalities = num_modalities
        self._num_classes = num_classes
        self._random_split = random_split
        self._train_size = train_size
        self._target_pre_transform = target_pre_transform
        self._processed_file_names = "data.pt"
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> Optional[List[str]]:
        r"""The name of the files in the ``self.raw_dir`` folder that must be present in order to skip
        downloading."""
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the ``self.processed_dir`` folder that must be present in order to skip
        processing."""
        return self._processed_file_names

    def download(self) -> None:
        r"""Downloads the dataset to the ``self.raw_dir`` folder."""
        path = download_url(self._url, self.raw_dir)
        extract_zip(path, self.raw_dir)

    def process(self) -> None:
        r"""Processes the dataset to the ``self.processed_dir`` folder. This function reads input files, creates a
        ''Data'' object, and saves it into the ''processed_dir''."""
        data_list = []

        for modality in range(self.num_modalities):
            if self._random_split:
                full_data = np.loadtxt(self.raw_paths[modality * 2], delimiter=",")
                full_labels = np.loadtxt(self.raw_paths[(modality * 2) + 1], delimiter=",")
                full_labels = full_labels.astype(int)

                train_idx, test_idx = self.get_random_split(full_labels, self._num_classes, self._train_size)
                num_train = len(train_idx)
                num_tests = len(test_idx)
            else:
                # The datasets provided here have already been pre-split into training and test sets.
                train_data = np.loadtxt(self.raw_paths[modality * 4], delimiter=",")
                train_labels = np.loadtxt(self.raw_paths[(modality * 4) + 1], delimiter=",")
                train_labels = train_labels.astype(int)
                num_train = len(train_labels)
                train_idx = torch.tensor(list(range(num_train)), dtype=torch.long)

                test_data = np.loadtxt(self.raw_paths[(modality * 4) + 2], delimiter=",")
                test_labels = np.loadtxt(self.raw_paths[(modality * 4) + 3], delimiter=",")
                test_labels = test_labels.astype(int)
                num_tests = len(test_labels)
                test_idx = torch.tensor(list(range(num_train, num_train + num_tests)), dtype=torch.long)

                full_data = np.concatenate((train_data, test_data), axis=0)
                full_labels = np.concatenate((train_labels, test_labels))
                full_labels = full_labels.astype(int)

            full_data = full_data if self.pre_transform is None else self.pre_transform(full_data)
            full_labels = full_labels if self._target_pre_transform is None else self._target_pre_transform(full_labels)

            edge_index, edge_weight = self.get_adjacency_info(full_data)
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)

            data = Data(
                x=full_data,
                edge_index=edge_index,
                edge_weight=edge_weight,
                adj_t=adj_t,
                y=full_labels,
                train_idx=train_idx,
                test_idx=test_idx,
                num_train=num_train,
                num_test=num_tests,
            )

            data = self.extend_data(data)
            data_list.append(data)

        torch.save(data_list, osp.join(self.processed_dir, "data.pt"))

    @staticmethod
    def get_random_split(labels, num_classes: int, train_size: float = 0.7) -> Tuple:
        """Split arrays into random train and test indices.

        Args:
            labels (array-like): Array-like object that represents the labels of the dataset.
            num_classes (int): The total number of classes in the dataset.
            train_size (float, optional): The proportion of the dataset to include in the train split that should be
                between 0.0 and 1.0. (default: 0.7)

        Returns:
            A tuple of two arrays containing the indices for the train and test sets.
        """
        train_idx = []
        test_idx = []
        for c in range(num_classes):
            idx = (labels == c).nonzero()[0]
            idx = idx[torch.randperm(len(idx))]
            num_train = int(len(idx) * train_size)
            train_idx.append(idx[:num_train])
            test_idx.append(idx[num_train:])

        train_idx = np.concatenate(train_idx)
        test_idx = np.concatenate(test_idx)
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)

        return train_idx, test_idx

    @staticmethod
    def get_adjacency_info(data: torch.Tensor) -> Tuple:
        """Calculate a sparse adjacency matrix of the input dataset defined by edge indices and edge attributes.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            A tuple of edge indices and edge attributes.
        """
        adj = torch.ones(data.shape[0], data.shape[0], dtype=torch.long)
        adj.fill_diagonal_(0)
        edge_index = (adj > 0).nonzero().t()

        return edge_index, None

    def extend_data(self, data: Data) -> Data:
        """Extend data object by adding additional attributes.

        Args:
            data (Data): An input data object.

        Returns:
            Extended data object with additional attributes.
        """
        return data

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self.num_modalities

    def get(self, modality_idx) -> Data:
        r"""Gets the data object at index ``idx``."""
        data_list = torch.load(osp.join(self.processed_dir, "data.pt"))
        return data_list[modality_idx]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> Union["Dataset", Data]:
        data_list = torch.load(osp.join(self.processed_dir, "data.pt"))
        return data_list

    @property
    def num_modalities(self) -> int:
        r"""Returns the number of modalities in the dataset."""
        return self._num_modalities

    @property
    def num_classes(self) -> int:
        r"""Returns the number of classes in the dataset."""
        return self._num_classes


class SparseMultiomicsDataset(MultiomicsDataset):
    r"""The multiomics data for creating sparse graph dataset based on the settings in the MOGONET paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        raw_file_names (list[callable], optional): The name of the files in the ``self.raw_dir`` folder that must be
            present in order to skip downloading.
        num_modalities (int): The total number of modalities in the dataset.
        num_classes (int): The total number of classes in the dataset.
        edge_per_node (int): Predefined number of edges per nodes in computing adjacency matrix.
        url (string, optional): The url to download the dataset from.
        random_split (bool, optional): Whether to split the dataset into random train and test subsets. (default:
            ``False``)
        train_size (float, optional): The proportion of the dataset to include in the train split that should be between
            0.0 and 1.0. This parameter is used when ``random_split`` is ``True``.
        equal_weight (bool, optional): Whether to use equal weights for all samples. (default: ``False``)
        transform (callable, optional): A function/transform that takes in an array_like data and returns a transformed
            version. The data object will be transformed before every access. (default: ``None``)
        pre_transform (callable, optional): A function/transform that takes in an array_like data and returns a
            transformed version. The data object will be transformed before being saved to disk. (default: ``None``)
        target_pre_transform (callable, optional): A function/transform that takes in an array_like of labels and
            returns a transformed version. The label object will be transformed before being saved to disk. (default:
            ``None``)
    """

    def __init__(
        self,
        root: str,
        raw_file_names: List[str],
        num_modalities: int,
        num_classes: int,
        edge_per_node: int,
        url: str = None,
        random_split: bool = False,
        train_size: float = 0.7,
        equal_weight: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        target_pre_transform: Optional[Callable] = None,
    ):
        self.edge_per_node = edge_per_node
        self.equal_weight = equal_weight
        self.sim_threshold = None  # similarity threshold to keep the desired number of edges in the adjacency matrix.
        super().__init__(
            root,
            num_modalities,
            num_classes,
            url,
            raw_file_names,
            random_split,
            train_size,
            transform,
            pre_transform,
            target_pre_transform,
        )

    def extend_data(self, data: Data) -> Data:
        """Extend data object by adding additional attributes.

        Args:
            data (Data): An input data object.

        Returns:
            Extended data object with additional attributes.
        """
        # Add train sample weights to the data object
        train_labels = torch.argmax(data.y[data.train_idx], dim=1)
        train_sample_weight = self._get_sample_weight(train_labels)
        data.train_sample_weight = train_sample_weight

        # Add adjacency matrices to the data object
        edge_index_train, edge_weight_train = self._get_adjacency_info(data.x[data.train_idx], train=True)
        adj_t_train = SparseTensor(row=edge_index_train[0], col=edge_index_train[1], value=edge_weight_train)

        edge_index, edge_weight = self._get_adjacency_info(
            data.x[data.train_idx], test_data=data.x[data.test_idx], train=False
        )
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)

        data.edge_index = edge_index
        data.edge_weight = edge_weight
        data.adj_t = adj_t
        data.edge_index_train = edge_index_train
        data.edge_weight_train = edge_weight_train
        data.adj_t_train = adj_t_train

        return data

    def _get_adjacency_info(
        self,
        train_data: torch.Tensor,
        test_data: torch.Tensor = None,
        train: bool = True,
        eps: float = 1e-8,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> Tuple:
        """Calculate a sparse adjacency matrix of the input dataset defined by edge indices and edge attributes.

        Args:
            train_data (torch.Tensor): The training data.
            test_data (torch.Tensor, optional): The test data. If ''None'' then the adjacency matrix is only calculated
                on the train data. (default: ``None``)
            train (bool, optional): Whether to use only train data to calculate the adjacency matrix. If ''False'' then
                the entire dataset (training and test sets) is used. (default: ``True``)
            eps (float, optional): Small value to avoid division by zero. (default: 1e-8)
            metric (DistanceMetric, optional): The metric to compute distance between input matrices. (default:
                ``DistanceMetric.COSINE``)

        Returns:
            A tuple of edge indices and edge attributes.
        """
        num_train = train_data.shape[0]
        num_test = 0 if test_data is None else test_data.shape[0]
        if train:
            adj_matrix = calculate_distance(train_data, eps=eps, metric=metric)
            self._find_sim_threshold(adj_matrix, num_train)
            non_zero_entries = self._generate_sparse_adj(adj_matrix, self_loop=train)
            adj_matrix = torch.mul(adj_matrix, non_zero_entries)
        else:
            adj_matrix = torch.zeros((num_train + num_test, num_train + num_test))
            dist = calculate_distance(train_data, test_data, eps=eps, metric=metric)
            non_zero_entries = self._generate_sparse_adj(dist, self_loop=train)
            adj_matrix[:num_train, num_train:] = torch.mul(dist, non_zero_entries)

            dist = calculate_distance(test_data, train_data, eps=eps, metric=metric)
            non_zero_entries = self._generate_sparse_adj(dist, self_loop=train)
            adj_matrix[num_train:, :num_train] = torch.mul(dist, non_zero_entries)

        adj_matrix_trans = adj_matrix.transpose(0, 1)
        adj_matrix = (
            adj_matrix
            + adj_matrix_trans * (adj_matrix_trans > adj_matrix).float()
            - adj_matrix * (adj_matrix_trans > adj_matrix).float()
        )
        identity_mat = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
        adj_matrix = F.normalize(adj_matrix + identity_mat, p=1)
        adj_matrix = adj_matrix.to_sparse()

        return adj_matrix.indices(), adj_matrix.values()

    def _find_sim_threshold(self, adj_mat: torch.Tensor, num_train: int) -> None:
        r"""Finds a similarity threshold for the adjacency matrix in order to keep the predefined number of edges per
        nodes.

        Args:
            adj_mat (torch.Tensor): The dense adjacency matrix.
            num_train (int): The number of samples in training data.
        """
        sorted_adj_mat = torch.sort(adj_mat.reshape(-1,), descending=True).values[self.edge_per_node * num_train]
        self.sim_threshold = sorted_adj_mat.item()

    def _generate_sparse_adj(self, adj_mat: torch.Tensor, self_loop: bool = True) -> torch.Tensor:
        r"""Returns a sparse adjacency matrix by setting entries below the ``sim_threshold`` to 0.

        Args:
            adj_mat (torch.Tensor): The dense adjacency matrix.
            self_loop (bool, optional): Whether to fill the main diagonal with zero. (default: ``True``)

        Returns:
            torch.Tensor: Computed sparse adjacency matrix.
        """
        non_zero_entries = (adj_mat >= self.sim_threshold).float()
        if self_loop:
            non_zero_entries.fill_diagonal_(0)

        return non_zero_entries

    def _get_sample_weight(self, labels: np.ndarray) -> torch.Tensor:
        r"""Get sample weights based on the class distribution.

        Args:
            labels (np.ndarray): A list of ground truth labels of samples.

        Returns:
            torch.Tensor: A list of label weights calculated for each sample.
        """
        if self.equal_weight:
            sample_weight = np.ones(len(labels)) / len(labels)
        else:
            count = np.bincount(labels, minlength=self.num_classes)
            sample_weight = count[labels] / np.sum(count)

        sample_weight = torch.tensor(sample_weight, dtype=torch.float)

        return sample_weight

    def __str__(self) -> str:
        r"""Returns a string representation of the dataset object.

        Returns:
            str: The string representation of the dataset object.
        """
        modalities_str = [
            "\nDataset info:",
            f"\n   number of modalities: {self.num_modalities}",
            f"\n   number of classes: {self.num_classes}",
            "\n\n   modality | total samples | num train | num test  | num features",
            f"\n   {'-' * 65}",
        ]
        for modality in range(self.num_modalities):
            modality_data = self.get(modality)
            modalities_str.append(
                f"\n   {modality + 1:<8} | "
                f"{len(modality_data.x):<13} | "
                f"{len(modality_data.x[modality_data.train_idx]):<9} | "
                f"{len(modality_data.x[modality_data.test_idx]):<9} | "
                f"{modality_data.num_features:<12}"
            )

        modalities_str.append(f"\n   {'-' * 65}\n\n")
        return "".join(modalities_str)
