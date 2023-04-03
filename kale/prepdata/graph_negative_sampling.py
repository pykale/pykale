import numpy as np
import torch


def negative_sampling(pos_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    r"""
    Negative sampling for link prediction. Copy-paste from https://github.com/NYXFLOWER/GripNet.

    Args:
        pos_edge_index (torch.Tensor): edge indices in COO format with shape [2, num_edges].
        num_nodes (int): the number of nodes in the graph.

    Returns:
        torch.Tensor: edge indices in COO format with shape [2, num_edges].
    """

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


def typed_negative_sampling(pos_edge_index: torch.Tensor, num_nodes: int, range_list: torch.Tensor) -> torch.Tensor:
    r"""
    Typed negative sampling for link prediction. Copy-paste from https://github.com/NYXFLOWER/GripNet.

    Args:
        pos_edge_index (torch.Tensor): edge indices in COO format with shape [2, num_edges].
        num_nodes (int): the number of nodes in the graph.
        range_list (torch.Tensor): the range of edge types. [[start_index, end_index], ...]

    Returns:
        torch.Tensor: edge indices in COO format with shape [2, num_edges].
    """

    tmp = []
    for start, end in range_list:
        tmp.append(negative_sampling(pos_edge_index[:, start:end], num_nodes))
    return torch.cat(tmp, dim=1)
