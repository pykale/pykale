# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================


from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colormaps
from matplotlib.cm import get_cmap
from nilearn.plotting import plot_connectome
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.utils._param_validation import Interval, Real, validate_params
from sklearn.utils.validation import indexable

from ..prepdata.tensor_reshape import normalize_tensor
from .model_weights import get_top_symmetric_weight


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
    if not isinstance(weight_img, np.ndarray):
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
    elif not isinstance(image_titles, list) or len(image_titles) != n_samples:
        raise ValueError("Invalid type or length of 'image_names'!")
    if marker_cmap is None:
        marker_colors = None
    elif isinstance(marker_cmap, str):
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

    if not isinstance(data, list):
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


@validate_params(
    {
        "weights": ["array-like"],
        "labels": ["array-like"],
        "coords": ["array-like"],
        "p": [Interval(Real, 0, 1, closed="neither")],
        "cmap": [str],
        "marker_size": [Real],
        "legend_params": [dict],
    },
    prefer_skip_nested_validation=True,
)
def visualize_connectome(weights, labels, coords, p=1e-3, cmap="tab20", marker_size=100, legend_params={}):
    """
    Visualize the top-p weighted ROI connections as a symmetric connectome plot.

    Selects the top `p` proportion of the largest (by absolute value) ROI pairwise weights,
    builds a symmetric connectivity matrix, and plots the connectome using `nilearn.plotting.plot_connectome`.

    Args:
        weights (array-like):
            1D array of edge weights corresponding to each unique ROI pair. Typically output from a model or analysis.
        labels (array-like):
            ROI names of shape (n_rois,). These correspond to region names in the brain atlas.
        coords (array-like):
            ROI coordinates of shape (n_rois, 3). Specifies 3D spatial locations of the ROIs.
        p (float, optional):
            Proportion of top-weighted connections to include (default: 1e-3). Must be in the open interval (0, 1).
        cmap (str, optional):
            Name of matplotlib colormap used for ROI node coloring (default: 'tab20').
        marker_size (float, optional):
            Size of node markers (default: 100).
        legend_params (dict, optional):
            Additional keyword arguments passed to the legend (e.g., `loc`, `fontsize`).

    Returns:
        nilearn.plotting.displays._projectors.ConnectivityProjection:
            A `nilearn` projection object representing the plotted connectome. Supports further customization
            such as saving or adding overlays.

    Note:
        - Assumes the weights are either symmetric or should be treated as symmetric.
        - Useful for visualizing connectivity graphs derived from model interpretation or brain network analysis.
    """
    # Ensure same lengths for labels and coords
    labels, coords = indexable(labels, coords)
    marker_colors = get_cmap(cmap)(np.arange(len(labels)))
    sym_weights, labels, coords = get_top_symmetric_weight(weights, labels, coords, p)

    # Ensure same lengths for weights, labels, and coords
    sym_weights, labels, coords = indexable(sym_weights, labels, coords)

    # Visualize the connectome
    proj = plot_connectome(sym_weights, coords, colorbar=True)

    # Add markers for each ROI
    for i in range(len(labels)):
        proj.add_markers(
            [coords[i]],
            marker_color=marker_colors[i],
            marker_size=marker_size,
            label=labels[i],
        )

    # Set legend parameters
    proj.axes[next(iter(proj.axes))].ax.legend(**legend_params)

    return proj


def draw_attention_map(attention_weights, out_path, colormap="viridis", title="Attention Map", xlabel="", ylabel=""):
    """Draws a heatmap of attention weights."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights.numpy(), cmap=colormap)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def draw_mol_with_attention(attention_weights, smile, out_path, colormap="viridis"):
    """Draws a molecule with attention weights as colors."""
    mol = Chem.MolFromSmiles(smile)
    weights = normalize_tensor(attention_weights)
    weights = weights.cpu().numpy().tolist()

    # Draw with RDKit 2D drawer
    cmap = colormaps[colormap]
    atom_colors = {i: cmap(float(w))[:3] for i, w in enumerate(weights)}

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii={i: 0.3 for i in atom_colors},
    )
    drawer.FinishDrawing()

    with open(out_path, "w") as f:
        f.write(drawer.GetDrawingText())


def save_or_show_plot(save_path: Optional[str] = None, show: bool = True, **fig_kwargs) -> None:
    """Save plot to file or show it.

    Args:
        save_path (str, optional): Path to save the figure. If None, the figure will be shown instead.
        show (bool, optional): Whether to show the figure. Defaults to True.
        **fig_kwargs: Additional keyword arguments for figure configuration. Supported parameters:
            - save_dpi (int): Dots per inch for the saved figure
            - show_dpi (int):  Dots per inch for the shown figure
            - fig_size (tuple): Size of the figure in inches
            - bbox_inches (str): Bounding box for saved figure
            - pad_inches (float): Padding for saved figure
            - Any other parameters supported by `plt.savefig()`

            If these parameters are not provided, matplotlib's default values will be used.

    Raises:
        ValueError: If fig_size is not a 2-element tuple/list.
    """
    # Extract and validate parameters
    save_dpi = fig_kwargs.pop("save_dpi", None)
    show_dpi = fig_kwargs.pop("show_dpi", None)
    fig_size = fig_kwargs.pop("fig_size", None)
    bbox_inches = fig_kwargs.pop("bbox_inches", None)
    pad_inches = fig_kwargs.pop("pad_inches", None)

    # Validate fig_size format
    if fig_size is not None:
        if not (isinstance(fig_size, (tuple, list)) and len(fig_size) == 2):
            raise ValueError("fig_size must be a 2-element tuple or list (width, height)")
        if not all(isinstance(x, (int, float)) and x > 0 for x in fig_size):
            raise ValueError("fig_size elements must be positive numbers (int or float)")

    # Get current figure
    fig = plt.gcf()

    # Apply figure size if provided
    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    # Apply display DPI early (affects both save and show if both are True)
    if show_dpi is not None and show:
        fig.set_dpi(show_dpi)

    # Save the figure if path is provided
    if save_path is not None:
        plt.tight_layout()

        # Build save parameters dict - only include explicitly provided values
        save_params = {
            "dpi": save_dpi,
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
        }
        save_kwargs = {k: v for k, v in save_params.items() if v is not None}

        # Merge with additional user parameters
        save_kwargs.update(fig_kwargs)

        plt.savefig(save_path, **save_kwargs)

    # Show the figure if requested
    if show:
        plt.show()

    plt.close()
