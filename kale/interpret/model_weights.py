# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import numpy as np
from tensorly.base import fold, unfold


def select_top_weight(weights, select_ratio: float = 0.05):
    """Select top weights in magnitude, and the rest of weights will be zeros

    Args:
        weights (array-like): model weights, can be a vector or a higher order tensor
        select_ratio (float, optional): ratio of top weights to be selected. Defaults to 0.05.

    Returns:
        array-like: top weights in the same shape with the input model weights
    """
    if type(weights) != np.ndarray:
        weights = np.array(weights)
    orig_shape = weights.shape

    if len(orig_shape) > 1:
        weights = unfold(weights, mode=0)[0]
    n_top_weights = int(weights.size * select_ratio)
    top_weight_idx = (-1 * abs(weights)).argsort()[:n_top_weights]
    top_weights = np.zeros(weights.size)
    top_weights[top_weight_idx] = weights[top_weight_idx]
    if len(orig_shape) > 1:
        top_weights = fold(top_weights, mode=0, shape=orig_shape)

    return top_weights
