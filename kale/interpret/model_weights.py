# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils._param_validation import Interval, Real, validate_params
from tensorly.base import fold, unfold


def select_top_weight(weights, select_ratio: float = 0.05):
    """Select top weights in magnitude, and the rest of weights will be zeros

    Args:
        weights (array-like): model weights, can be a vector or a higher order tensor
        select_ratio (float, optional): ratio of top weights to be selected. Defaults to 0.05.

    Returns:
        array-like: top weights in the same shape with the input model weights
    """
    if not isinstance(weights, np.ndarray):
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


@validate_params({"rois": ["array-like"]}, prefer_skip_nested_validation=True)
def _get_pairwise_rois(rois):
    """
    Generate all unique ROI pair tuples (upper triangle only).

    Constructs unique combinations of region-of-interest (ROI) names to represent undirected pairwise connections.
    Useful for mapping pairwise connectivity weights or features.

    Args:
        rois (array-like):
            List or array of ROI names (e.g., region labels or identifiers).

    Returns:
        array-like:
            Array of ROI pair tuples corresponding to the upper triangle pairs.
    """
    pairs = list(combinations(rois, 2))
    return np.array(pairs)


@validate_params(
    {
        "weights": ["array-like"],
        "labels": ["array-like"],
        "coords": ["array-like"],
        "p": [Interval(Real, 0, 1, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def get_top_symmetric_weight(weights, labels, coords, p=0.001):
    """
    Construct a symmetric weight matrix for the top-p% ROI pairs.

    Selects the top `p` proportion of ROI pairwise weights (by absolute value),
    builds a symmetric matrix of the selected ROI connections, and extracts
    the labels and coordinates of the involved ROIs.

    Args:
        weights (array-like):
            1D array of edge weights for each ROI pair.
        labels (array-like):
            List of ROI names or indices in the full dataset.
        coords (array-like):
            2D array of ROI coordinates with shape (n_rois, 3).
        p (float, optional):
            Proportion of top weights to retain (default: 0.001). Must be in (0, 1).

    Returns:
        tuple:
            - sym_weights (array-like): Symmetric matrix of top-weighted ROI connections.
            - top_roi_labels (array-like): Labels of ROIs involved in top-weighted pairs.
            - top_roi_coords (array-like): Coordinates corresponding to `top_roi_labels`.

    Note:
        This function assumes that the input `weights` correspond to the ROI pairs
        in the upper triangle without duplicates, and converts them into a full symmetric matrix.
    """
    # Get lower triangle ROI pairs
    pairwise_rois = _get_pairwise_rois(labels)

    # Ensure weights and pairwise_rois have same length
    weights, pairwise_rois = indexable(weights, pairwise_rois)

    # Select top p% weights
    weights = pd.Series(np.copy(weights), pairwise_rois)
    rank = weights.abs().nlargest(int(len(weights.index) * p))
    weights = weights[rank.index]

    # Raises ValueError if no weights are selected
    if len(weights) == 0:
        raise ValueError("No weights selected. Please use larger p.")

    # Use tuple keys directly from weights.index
    pairs = np.array(list(weights.index))
    unique = np.unique(pairs)

    # Map unique ROIs to indices
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    indices = [label_to_index[roi] for roi in unique]

    # Create top roi mappings
    top_labels = np.array(labels)[indices]
    top_coords = np.array(coords)[indices]
    mappings = {roi: idx for idx, roi in enumerate(top_labels)}

    # Create symmetric weight matrix
    sym_weights = np.zeros([len(indices)] * 2)
    for (roi1, roi2), weight in zip(pairs, weights.values):
        i, j = mappings[roi1], mappings[roi2]
        sym_weights[i, j] = weight
        sym_weights[j, i] = weight

    return sym_weights, top_labels, top_coords
