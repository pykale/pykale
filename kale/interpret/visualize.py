# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np


def plot_weights(weight_img, background_img=None, color_marker_pos="rs", color_marker_neg="bs", marker_size=6):
    """Visualize model weights

    Args:
        weight_img (array-like): model weight/coefficients in 2D, could be a 2D slice of a 3D or higher order tensor.
        background_img (array-like, optional): 2D background image. Defaults to None.
        color_marker_pos (str, optional): Color and marker for weights in positive values. Defaults to red "r.".
        color_marker_neg (str, optional): Color and marker for weights in negative values. Defaults to blue "b.".
        marker_size (int, optional): Marker size. Defaults to 5.

    Returns:
        [matplotlib.figure.Figure]: Figure to plot.
    """
    if type(weight_img) != np.ndarray:
        weight_img = np.array(weight_img)
    if len(weight_img.shape) != 2:
        raise ValueError(
            "weight_img is expected to be a 2D matrix, but got an array in shape %s" % str(weight_img.shape)
        )
    fig = plt.figure()
    ax = fig.add_subplot()
    if background_img is not None:
        ax.imshow(background_img)
        weight_img[np.where(background_img == 0)] = 0

    weight_pos_coords = np.where(weight_img > 0)
    weight_neg_coords = np.where(weight_img < 0)

    ax.plot(weight_pos_coords[1], weight_pos_coords[0], color_marker_pos, markersize=marker_size)
    ax.plot(weight_neg_coords[1], weight_neg_coords[0], color_marker_neg, markersize=marker_size)

    return fig


def plot_multi_images(images, n_cols=10, n_rows=None, marker_locs=None, marker_size=6):
    """Plot multiple images with markers in one figure.

    Args:
        images (array-like): Images to plot, shape(n_samples, dim1, dim2)
        n_cols (int, optional): Number of columns for plotting multiple images. Defaults to 10.
        n_rows (int, optional): Number of rows for plotting multiple images. If None, n_rows = n_samples / n_cols.
        marker_locs (array-like, optional): Locations of markers, shape (n_samples, 2 * n_markers). Defaults to None.
        marker_size (int, optional): Marker size. Defaults to 6.

    Returns:
        [matplotlib.figure.Figure]: Figure to plot.
    """
    if n_rows is None:
        n_rows = int(images.shape[0] / n_cols) + 1

    fig = plt.figure(figsize=(20, 36))

    for i in range(images.shape[0]):
        fig.add_subplot(n_rows, n_cols, i + 1)
        plt.axis("off")
        plt.imshow(images[i, ...])
        if marker_locs is not None:
            coords = marker_locs[i, :].reshape((-1, 2))
            n_landmark = coords.shape[0]
            for j in range(n_landmark):
                ix = coords[j, 0]
                iy = coords[j, 1]
                plt.plot(
                    ix,
                    iy,
                    marker="o",
                    markersize=marker_size,
                    markerfacecolor=(1, 1, 1, 0.1),
                    markeredgewidth=1.5,
                    markeredgecolor="r",
                )
        plt.title(i + 1)

    return fig
