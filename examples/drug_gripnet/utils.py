import pickle

import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score

EPS = 1e-13


def normalize(input):
    norm_square = (input ** 2).sum(dim=1)
    return input / torch.sqrt(norm_square.view(-1, 1))


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def auprc_auroc_ap(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    auroc, ap = metrics.roc_auc_score(y, pred), metrics.average_precision_score(y, pred)
    y, xx, _ = metrics._ranking.precision_recall_curve(y, pred)
    auprc = metrics._ranking.auc(xx, y)

    return auprc, auroc, ap


def micro_macro(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    micro, macro = metrics.f1_score(y, pred, average="micro"), metrics.f1_score(y, pred, average="macro")

    return micro, macro


def acc(target_tensor, score_tensor):
    y = target_tensor.detach().cpu().numpy()
    pred = score_tensor.detach().cpu().numpy()
    return accuracy_score(y, pred)


def load_graph(pt_file_path="./sample_graph.pt"):
    """
    Parameters
    ----------
    pt_file_path : file path

    Returns
    -------
    graph : torch_geometric.data.Data
        - data.n_node: number of nodes
        - data.n_node_type: number of node types == (1 or 2)
        - data.n_edge: number of edges
        - data.n_edge_type: number of edge types
        - data.node_type: (source_node_type, target_node_type)

        - data.edge_index: [list of] torch.Tensor, int, shape (2, n_edge), [indexed by edge type]
            [0, :] : source node index
            [1, :] : target node index
        - data.edge_type: None or list of torch.Tensor, int, shape (n_edge,), indexed by edge type
        - data.edge_weight: None or list of torch.Tensor, float, shape (n_edge,)

        - data.source_node_idx_to_id: dict {idx : id}
        - data.target_node_idx_to_id: dict {idx : id}
    """

    return torch.load(pt_file_path)


def load_node_idx_to_id_dict(pkl_file_path="./data/pose-1/map.pkl"):
    """
    Parameters:
    -----------
    The path of index maps in the dataset directory

    Returns:
    --------
    a dictionary of map from node index to entity id/name
    """
    with open(pkl_file_path, "rb") as f:
        out = pickle.load(f)
    return out


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
