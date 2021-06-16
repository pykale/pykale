# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import matplotlib.pyplot as plt
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


def plot_weights(weight_img, background_img=None, color_marker_pos="r.", color_marker_neg="b.", marker_size=5):
    """Visualize model weights

    Args:
        weight_img (array-like): model weight/coefficients in 2D, could be a 2D slice of a 3D or higher order tensor.
        background_img (array-like, optional): 2D background image. Defaults to None.
        color_marker_pos (str, optional): Color and marker for weights in positive values. Defaults to red "r.".
        color_marker_neg (str, optional): Color and marker for weights in negative values. Defaults to blue "b.".
        marker_size (int, optional): Marker size. Defaults to 5.
    """
    if type(weight_img) != np.ndarray:
        weight_img = np.array(weight_img)
    if len(weight_img.shape) != 2:
        raise ValueError(
            "weight_img is expected to be a 2D matrix, but got an array in shape %s" % str(weight_img.shape)
        )
    weight_pos_coords = np.where(weight_img > 0)
    weight_neg_coords = np.where(weight_img < 0)

    fig = plt.figure()
    ax = fig.add_subplot()
    if background_img is not None:
        ax.imshow(background_img)

    ax.plot(weight_pos_coords[1], weight_pos_coords[0], color_marker_pos, markersize=marker_size)
    ax.plot(weight_neg_coords[1], weight_neg_coords[0], color_marker_neg, markersize=marker_size)

    return fig
