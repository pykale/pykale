import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data.data import Data
from yacs.config import CfgNode

from kale.utils.download import download_file_by_url


class PolypharmacyDataset(Dataset):
    r"""Polypharmacy side effect prediction dataset. Only for full-batch training.

    Args:
        cfg_dataset (CfgNode): configurations of the dataset.
        mode (str): "train", "valid" or "test". Defaults to "train".
    """

    def __init__(self, cfg_dataset: CfgNode, mode: str = "train"):

        super(PolypharmacyDataset, self).__init__()

        self.cfg_dataset = cfg_dataset
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
        download_file_by_url(self.cfg_dataset.URL, self.cfg_dataset.ROOT, f"{self.cfg_dataset.NAME}.pt")
        data_path = os.path.join(self.cfg_dataset.ROOT, f"{self.cfg_dataset.NAME}.pt")

        # load data
        return torch.load(data_path)

    def __len__(self):

        return 1

    def __getitem__(self, idx):

        return self.edge_index, self.edge_type, self.edge_type_range
