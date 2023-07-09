import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data.data import Data

from kale.utils.download import download_file_by_url


class PolypharmacyDataset(Dataset):
    r"""Polypharmacy side effect prediction dataset. Only for full-batch training.

    Args:
        url (string): The url to download the dataset from.
        root (string): The root directory containing the dataset file.
        name (string): Name of the dataset.
        mode (string): "train", "valid" or "test". Defaults to "train".
    """

    def __init__(self, url: str, root: str, name: str, mode: str = "train"):
        super(PolypharmacyDataset, self).__init__()

        self.url = url
        self.root = root
        self.name = name
        data = self.load_data()

        self.edge_index = data.__getitem__(f"{mode}_idx")
        self.edge_type = data.__getitem__(f"{mode}_et")
        self.edge_type_range = data.__getitem__(f"{mode}_range")

        if mode == "train":
            self.protein_feat = data.g_feat
            self.protein_edge_index = data.gg_edge_index
            self.drug_feat = data.d_feat
            self.protein_drug_edge_index = data.gd_edge_index

        self.len = self.edge_type_range.shape[0]

    def load_data(self) -> Data:
        """Setup dataset: download if need and load it."""

        # download data if not exist
        download_file_by_url(self.url, self.root, f"{self.name}.pt")
        data_path = os.path.join(self.root, f"{self.name}.pt")

        # load data
        return torch.load(data_path)

    def __len__(self):

        return 1

    def __getitem__(self, idx):

        return self.edge_index, self.edge_type, self.edge_type_range
