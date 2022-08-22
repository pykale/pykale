# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _none2dict(kwarg):
    if kwarg is None:
        return {}
    else:
        return kwarg


def plot_weights(
    weight_img, background_img=None, color_marker_pos="rs", color_marker_neg="gs", im_kwargs=None, marker_kwargs=None
):
    """Visualize model weights

    Args:
        weight_img (array-like): Model weight/coefficients in 2D, could be a 2D slice of a 3D or higher order tensor.
        background_img (array-like, optional): 2D background image. Defaults to None.
        color_marker_pos (str, optional): Color and marker for weights in positive values. Defaults to red "rs".
        color_marker_neg (str, optional): Color and marker for weights in negative values. Defaults to blue "gs".
        im_kwargs (dict, optional): Keyword arguments for background images. Defaults to None.
        marker_kwargs (dict, optional): Keyword arguments for background images. Defaults to None.

    Returns:
        [matplotlib.figure.Figure]: Figure to plot.
    """
    if type(weight_img) != np.ndarray:
        weight_img = np.array(weight_img)
    if len(weight_img.shape) != 2:
        raise ValueError(
            "weight_img is expected to be a 2D matrix, but got an array in shape %s" % str(weight_img.shape)
        )
    im_kwargs = _none2dict(im_kwargs)
    marker_kwargs = _none2dict(marker_kwargs)
    fig = plt.figure()
    ax = fig.add_subplot()
    if background_img is not None:
        ax.imshow(background_img, **im_kwargs)
        weight_img[np.where(background_img == 0)] = 0

    weight_pos_coords = np.where(weight_img > 0)
    weight_neg_coords = np.where(weight_img < 0)

    ax.plot(weight_pos_coords[1], weight_pos_coords[0], color_marker_pos, **marker_kwargs)
    ax.plot(weight_neg_coords[1], weight_neg_coords[0], color_marker_neg, **marker_kwargs)

    return fig


def plot_multi_images(
    images,
    n_cols=1,
    n_rows=None,
    marker_locs=None,
    image_titles=None,
    marker_titles=None,
    marker_cmap=None,
    figsize=None,
    im_kwargs=None,
    marker_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
):
    """Plot multiple images with markers in one figure.

    Args:
        images (array-like): Images to plot, shape(n_samples, dim1, dim2)
        n_cols (int, optional): Number of columns for plotting multiple images. Defaults to 1.
        n_rows (int, optional): Number of rows for plotting multiple images. If None, n_rows = n_samples / n_cols.
        marker_locs (array-like, optional): Locations of markers, shape (n_samples, 2 * n_markers). Defaults to None.
        marker_titles (list, optional): Names of the markers, where len(marker_names) == n_markers. Defaults to None.
        marker_cmap (str, optional): Name of the color map used for plotting markers. Default to None.
        image_titles (list, optional): List of title for each image, where len(image_names) == n_samples.
            Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to None.
        im_kwargs (dict, optional): Keyword arguments for plotting images. Defaults to None.
        marker_kwargs (dict, optional): Keyword arguments for markers. Defaults to None.
        legend_kwargs (dict, optional): Keyword arguments for legend. Defaults to None.
        title_kwargs (dict, optional): Keyword arguments for title. Defaults to None.

    Returns:
        [matplotlib.figure.Figure]: Figure to plot.
    """
    image_var_type = type(images)
    if image_var_type == np.ndarray:
        n_samples = images.shape[0]
    elif image_var_type == list:
        n_samples = len(images)
    else:
        raise ValueError("Unsupported variable type %s for 'images'" % image_var_type)
    if n_rows is None:
        n_rows = int(n_samples / n_cols) + 1
    im_kwargs = _none2dict(im_kwargs)
    marker_kwargs = _none2dict(marker_kwargs)
    legend_kwargs = _none2dict(legend_kwargs)
    title_kwargs = _none2dict(title_kwargs)
    if figsize is None:
        figsize = (20, 36)
    fig = plt.figure(figsize=figsize)
    if image_titles is None:
        image_titles = np.arange(n_samples) + 1
    elif type(image_titles) != list or len(image_titles) != n_samples:
        raise ValueError("Invalid type or length of 'image_names'!")
    if marker_cmap is None:
        marker_colors = None
    elif type(marker_cmap) == str:
        marker_colors = plt.get_cmap(marker_cmap).colors
    else:
        raise ValueError("Unsupported type %s for argument 'marker_cmap'" % type(marker_cmap))
    for i in range(n_samples):
        fig.add_subplot(n_rows, n_cols, i + 1)
        plt.axis("off")
        plt.imshow(images[i], **im_kwargs)
        if marker_locs is not None:
            coords = marker_locs[i, :].reshape((-1, 2))
            n_marker = coords.shape[0]
            if marker_titles is not None and len(marker_titles) == n_marker:
                plt_legend = True
            else:
                plt_legend = False
            for j in range(n_marker):
                ix = coords[j, 0]
                iy = coords[j, 1]
                if marker_colors is not None:
                    marker_kwargs["color"] = marker_colors[j]
                if plt_legend:
                    marker_kwargs["label"] = marker_titles[j]
                plt.scatter(ix, iy, **marker_kwargs)
            if plt_legend:
                plt.legend(**legend_kwargs)
        plt.title(image_titles[i], **title_kwargs)

    return fig


def distplot_1d(
    data,
    labels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    figsize=None,
    colors=None,
    title_kwargs=None,
    hist_kwargs=None,
):
    """Plot distribution of 1D data.

    Args:
        data (array-like or list): Data to plot.
        labels (list, optional): List of labels for each data. Defaults to None.
        xlabel (str, optional): Label for x-axis. Defaults to None.
        ylabel (str, optional): Label for y-axis. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        figsize (tuple, optional): Figure size. Defaults to None.
        colors (str, optional): Color of the line. Defaults to None.
        title_kwargs (dict, optional): Keyword arguments for title. Defaults to None.
        hist_kwargs (dict, optional): Keyword arguments for histogram. Defaults to None.

    Returns:
        [matplotlib.figure.Figure]: Figure to plot.
    """
    hist_kwargs = _none2dict(hist_kwargs)
    title_kwargs = _none2dict(title_kwargs)

    fig = plt.figure(figsize=figsize)
    if colors is None:
        colors = plt.get_cmap("Set1").colors

    if type(data) != list:
        data = [data]

    if labels is None:
        labels = np.range(len(data))

    for i in range(len(data)):
        sns.histplot(data[i], color=colors[i], label=labels[i], **hist_kwargs)

    if title is not None:
        plt.title(title, **title_kwargs)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()

    return fig
