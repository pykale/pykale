import os

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data.data import Data
from yacs.config import CfgNode

from kale.prepdata.supergraph_construct import SuperVertexParaSetting
from kale.utils.download import download_file_by_url

EPS = 1e-13


def load_data(cfg_dataset: CfgNode) -> Data:
    """Setup dataset: download if need and load it."""

    # download data if not exist
    download_file_by_url(cfg_dataset.URL, cfg_dataset.ROOT, f"{cfg_dataset.NAME}.pt")
    data_path = os.path.join(cfg_dataset.ROOT, f"{cfg_dataset.NAME}.pt")

    # load data
    return torch.load(data_path)


class PolypharmacyDataset(Dataset):
    """Polypharmacy side effect prediction dataset. Only for full-batch training."""

    def __init__(self, data: Data, mode: str = "train"):
        super(PolypharmacyDataset, self).__init__()

        self.edge_index = data.__getitem__(f"{mode}_idx")
        self.edge_type = data.__getitem__(f"{mode}_et")
        self.edge_type_range = data.__getitem__(f"{mode}_range")

        self.len = self.edge_type_range.shape[0]

    def __len__(self):

        return 1

    def __getitem__(self, idx):

        return self.edge_index, self.edge_type, self.edge_type_range


def get_all_dataloader(data: Data):
    """Get train and test dataloader"""

    dataloader_list = []
    for mode in ["train", "test"]:
        dataset = PolypharmacyDataset(data, mode=mode)
        loader = DataLoader(dataset, batch_size=1)
        dataloader_list.append(loader)

    return dataloader_list
