import imp
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


def setup_supervertex(sv_configs: CfgNode) -> SuperVertexParaSetting:
    """Get supervertex parameter setting from configurations."""

    exter_list = sv_configs.EXTER_AGG_CHANNELS_LIST

    if len(exter_list):
        exter_dict = {k: v for k, v in exter_list}

        return SuperVertexParaSetting(
            sv_configs.NAME,
            sv_configs.INTER_FEAT_CHANNELS,
            sv_configs.INTER_AGG_CHANNELS_LIST,
            exter_agg_channels_dict=exter_dict,
            mode=sv_configs.MODE,
        )

    return SuperVertexParaSetting(sv_configs.NAME, sv_configs.INTER_FEAT_CHANNELS, sv_configs.INTER_AGG_CHANNELS_LIST,)


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


# ----------------------------------------------------------------------------
# Copy-paste from https://github.com/NYXFLOWER/GripNet


def negative_sampling(pos_edge_index, num_nodes):
    idx = pos_edge_index[0] * num_nodes + pos_edge_index[1]
    idx = idx.to(torch.device("cpu"))

    perm = torch.tensor(np.random.choice(num_nodes ** 2, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero().view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(np.random.choice(num_nodes ** 2, rest.size(0)))
        mask = torch.from_numpy(np.isin(tmp, idx).astype(np.uint8))
        perm[rest] = tmp
        rest = mask.nonzero().view(-1)

    row, col = perm / num_nodes, perm % num_nodes
    return torch.stack([row, col], dim=0).long().to(pos_edge_index.device)


def typed_negative_sampling(pos_edge_index, num_nodes, range_list):
    tmp = []
    for start, end in range_list:
        tmp.append(negative_sampling(pos_edge_index[:, start:end], num_nodes))
    return torch.cat(tmp, dim=1)


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics.precision_recall_curve(y, pred)
    auprc = metrics.auc(xx, y)

    return auprc, auroc, ap


# ----------------------------------------------------------------------------
