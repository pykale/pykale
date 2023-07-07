# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Provide distance (i.e., similarity) calculation methods using various distance metrics.
"""

from enum import Enum

import torch


class DistanceMetric(Enum):
    COSINE = "COSINE"


def calculate_distance(
    x1: torch.Tensor, x2: torch.Tensor = None, eps: float = 1e-8, metric: DistanceMetric = DistanceMetric.COSINE
) -> torch.Tensor:
    r"""Returns similarity between :math:`x_1` and :math:`x_2`, computed along `dim`=1. This method calculates the
    similarity between each pair of data points in two input matrices.

    Note that this implementation differs from the existing implementations in
    `PyTorch <https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html>`__, as they calculate the
    similarity between each row of one matrix with its corresponding row in the other matrix (i.e., pairwise distance
    between columns of input matrices).

    Args:
        x1 (torch.Tensor): The tensor input data.
        x2 (torch.Tensor, optional): The tensor input data. (default ``None``)
        eps (float, optional): Small value to avoid division by zero. (default: 1e-8)
        metric (DistanceMetric, optional): The metric to compute distance between input matrices. (default: ``DistanceMetric.COSINE``)

        Returns:
        torch.Tensor: The computed similarity tensor between :math:`x_1` and :math:`x_2`.
    """
    if metric == DistanceMetric.COSINE:
        x2 = x1 if x2 is None else x2
        w1 = torch.norm(x1, p=2, dim=1, keepdim=True)
        w2 = w1 if x2 is None else torch.norm(x2, p=2, dim=1, keepdim=True)
        return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

    raise Exception("This metric is not still implemented")
