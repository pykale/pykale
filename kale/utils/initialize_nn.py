# =============================================================================
# Author: Sina Tabakhi, sina.tabakhi@gmail.com
# =============================================================================

"""
Provide methods for initializing neural network parameters (i.e., weights and biases).
"""

import torch.nn as nn


def xavier_init(module) -> None:
    r"""Fills the weight of the input Tensor with values using a normal distribution.

    Args:
        module (torch.Tensor): The input module.
    """
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight)


def bias_init(module) -> None:
    r"""Fills the bias of the input Tensor with zeros.

    Args:
        module (torch.Tensor): The input module.
    """
    if type(module) == nn.Linear and module.bias is not None:
        module.bias.data.fill_(0.0)
