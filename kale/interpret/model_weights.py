import matplotlib.pylab as plt
import numpy as np
from tensorly.base import unfold, fold


def select_top_weight(weights, select_rate: float = 0.05):
    if type(weights) != np.ndarray:
        weights = np.array(weights)
    orig_shape = weights.shape

    if len(orig_shape) > 1:
        weights = unfold(weights, mode=0)[0]
    n_weights = int(weights.size * select_rate)
    top_weight_idx = (abs(weights)).argsort()[::-1][:n_weights]
    top_weights = np.zeros(weights.size)
    top_weights[top_weight_idx] = weights[top_weight_idx]
    if len(orig_shape) > 1:
        top_weights = fold(top_weights, mode=0, shape=orig_shape)

    return top_weights


def plot_weights(weight_img, background_img=None, color_marker_pos="r.", color_marker_neg="b."):
    
    weight_pos_coords = np.where(weight_img > 0)
    weight_neg_coords = np.where(weight_img < 0)

    fig = plt.figure()
    ax = fig.add_subplot()
    if background_img is not None:
        ax.imshow(background_img)

    ax.plot(weight_pos_coords[1], weight_pos_coords[0], color_marker_pos, markersize=1)
    ax.plot(weight_neg_coords[1], weight_neg_coords[0], color_marker_neg, markersize=1)

    plt.show()
