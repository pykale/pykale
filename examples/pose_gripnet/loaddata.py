import torch
import numpy as np
import os
from torch_geometric.data import Data


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat([edge_type, edge_type])


def get_range_list(edge_list, is_node=False):
    idx = 0 if is_node else 1
    tmp = []
    s = 0
    for i in edge_list:
        tmp.append((s, s + i.shape[idx]))
        s += i.shape[idx]
    return torch.tensor(tmp)


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def process_edge_multirelational(raw_edge_list, p=0.9):
    train_list = []
    test_list = []
    train_label_list = []
    test_label_list = []

    for i, idx in enumerate(raw_edge_list):
        train_mask = np.random.binomial(1, p, idx.shape[1])
        test_mask = 1 - train_mask
        train_set = train_mask.nonzero()[0]
        test_set = test_mask.nonzero()[0]

        train_list.append(idx[:, train_set])
        test_list.append(idx[:, test_set])

        train_label_list.append(torch.ones(2 * train_set.size, dtype=torch.long) * i)
        test_label_list.append(torch.ones(2 * test_set.size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx) for idx in train_list]
    test_list = [to_bidirection(idx) for idx in test_list]

    train_range = get_range_list(train_list)
    test_range = get_range_list(test_list)

    train_edge_idx = torch.cat(train_list, dim=1)
    test_edge_idx = torch.cat(test_list, dim=1)

    train_et = torch.cat(train_label_list)
    test_et = torch.cat(test_label_list)

    return train_edge_idx, train_et, train_range, test_edge_idx, test_et, test_range


def construct_dataset(cfg):
    print('==> Preparing data ' + cfg.DATASET.NAME + ' at ' + cfg.DATASET.ROOT)
    source_data_dir = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME)
    data_dd = torch.load(os.path.join(source_data_dir, cfg.DATASET.DD))
    data_gd = torch.load(os.path.join(source_data_dir, cfg.DATASET.GD))
    data_gg = torch.load(os.path.join(source_data_dir, cfg.DATASET.GG))
    train_idx, train_et, train_range, test_idx, test_et, test_range = process_edge_multirelational(data_dd.edge_index,
                                                                                                   p=0.9)
    data = Data()
    data.g_feat = sparse_id(data_gg.n_node)
    data.d_feat = sparse_id(data_dd.n_node)
    data.edge_weight = torch.ones(data_gg.n_edge)
    data.gd_edge_index = torch.tensor(data_gd.edge_index, dtype=torch.long)
    data.gg_edge_index = torch.tensor(data_gg.edge_index, dtype=torch.long)

    data.train_idx = train_idx
    data.train_et = train_et
    data.train_range = train_range
    data.test_idx = test_idx
    data.test_et = test_et
    data.test_range = test_range
    data.n_edges_per_type = [(i[1] - i[0]).data.tolist() for i in data.test_range]

    data.n_g_node = data_gg.n_node
    data.n_d_node = data_dd.n_node
    data.n_dd_edge_type = data_dd.n_edge_type
    return data
