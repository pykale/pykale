import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from rdkit import Chem

from kale.interpret import visualize
from kale.interpret.visualize import draw_attention_map, draw_mol_with_attention, save_or_show_plot

matplotlib.use("Agg")  # Use non-interactive backend for tests


@pytest.fixture
def dummy_attention():
    return torch.rand(10, 15)  # Small dummy attention matrix


@pytest.fixture
def dummy_smile():
    return "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin


def test_plot_weights():
    # Create a small 2D array with positive and negative weights
    weights = np.zeros((5, 5))
    weights[1, 2] = 1.5  # positive
    weights[3, 4] = -2.0  # negative
    fig = visualize.plot_weights(weights)
    ax = fig.axes[0]
    # Extract plotted data for markers
    lines = ax.get_lines()
    # There should be two marker sets plotted (one for pos, one for neg)
    assert len(lines) == 2
    # Check that the positive marker is at (2, 1)
    pos_line = lines[0]
    xdata, ydata = pos_line.get_xdata(), pos_line.get_ydata()
    assert (2 in xdata) and (1 in ydata)
    # Check that the negative marker is at (4, 3)
    neg_line = lines[1]
    xdata2, ydata2 = neg_line.get_xdata(), neg_line.get_ydata()
    assert (4 in xdata2) and (3 in ydata2)
    plt.close(fig)


def test_plot_multi_images():
    # Create two simple images
    images = np.zeros((2, 10, 10))
    images[0, 5, 5] = 1
    images[1, 2, 2] = 1
    # Marker locations: 2 markers per image
    marker_locs = np.array([[5, 5, 1, 1], [2, 2, 8, 8]])
    marker_titles = ["A", "B"]
    image_titles = ["Img1", "Img2"]
    fig = visualize.plot_multi_images(
        images,
        n_cols=2,
        marker_locs=marker_locs,
        image_titles=image_titles,
        marker_titles=marker_titles,
        marker_cmap=None,
        marker_kwargs={"s": 50},
    )
    # Check titles are set
    for ax, title in zip(fig.axes, image_titles):
        assert title in ax.get_title()
    plt.close(fig)


def test_distplot_1d():
    data1 = np.random.randn(100)
    data2 = np.random.randn(100) + 2
    labels = ["Group 1", "Group 2"]
    fig = visualize.distplot_1d([data1, data2], labels=labels, xlabel="Value", ylabel="Frequency", title="Dist")
    # Check that the legend contains the labels
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    for label in labels:
        assert label in legend_labels
    plt.close(fig)


def test_visualize_connectome(monkeypatch):
    # Provide dummy symmetric connectome data
    n_rois = 4
    # 1D weights for upper triangle (excluding diagonal)
    weights = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.7])
    labels = np.array(["A", "B", "C", "D"])
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

    # Patch plot_connectome and get_top_symmetric_weight to avoid heavy nilearn plotting
    class DummyProj:
        def __init__(self):
            self.axes = {
                "main": type("DummyAx", (), {"ax": type("DummyAx2", (), {"legend": lambda self, **kwargs: None})()})()
            }

        def add_markers(self, coords, marker_color, marker_size, label):
            pass

    monkeypatch.setattr(visualize, "plot_connectome", lambda *args, **kwargs: DummyProj())
    monkeypatch.setattr(
        visualize,
        "get_top_symmetric_weight",
        lambda weights, labels, coords, p: (np.ones([n_rois] * 2), labels, coords),
    )

    proj = visualize.visualize_connectome(weights, labels, coords, p=0.5)
    assert hasattr(proj, "add_markers")


def test_plot_weights_background_mask():
    weights = np.ones((5, 5))
    background = np.ones((5, 5))
    background[2, 2] = 0  # mask center
    weights[2, 2] = 5
    fig = visualize.plot_weights(weights, background_img=background)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_weights_invalid_shape():
    with pytest.raises(ValueError):
        visualize.plot_weights(np.ones((5, 5, 5)))


def test_plot_multi_images_auto_rows():
    images = np.random.rand(3, 5, 5)
    marker_locs = np.array([[1, 1, 2, 2], [3, 3, 4, 4], [0, 0, 1, 1]])
    fig = visualize.plot_multi_images(images, n_cols=2, marker_locs=marker_locs, marker_titles=["X", "Y"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_distplot_1d_single_group():
    data = np.random.randn(100)
    fig = visualize.distplot_1d(data, labels=["Only"], xlabel="X", ylabel="Y", title="Title")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_distplot_1d_invalid_label_length():
    with pytest.raises(IndexError):
        visualize.distplot_1d([np.random.randn(100), np.random.randn(100)], labels=["Only one"])


def test_plot_multi_images_invalid_image_titles():
    with pytest.raises(ValueError):
        images = np.random.rand(2, 5, 5)
        visualize.plot_multi_images(images, image_titles=["One only"])


def test_draw_attention_map_runs(dummy_attention):
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "attn_map.png")
        draw_attention_map(dummy_attention, out_path)
        assert os.path.isfile(out_path)


def test_draw_mol_with_attention(tmp_path, dummy_smile):
    mol = Chem.MolFromSmiles(dummy_smile)
    num_atoms = mol.GetNumAtoms()
    attention_tensor = torch.rand(num_atoms)

    out_path = tmp_path / "mol_attention.svg"
    draw_mol_with_attention(attention_tensor, dummy_smile, str(out_path))

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_save_or_show_plot_basic_save(tmp_path):
    """Test basic save functionality of save_or_show_plot."""
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 2])
    out_path = tmp_path / "test_plot.png"
    save_or_show_plot(save_path=str(out_path), show=False)
    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0


def test_save_or_show_plot_custom_dpi(tmp_path):
    """Test save_or_show_plot with custom DPI setting."""
    plt.figure()
    plt.plot([1, 2, 3], [2, 3, 1])
    out_path_dpi = tmp_path / "test_plot_dpi.png"
    save_or_show_plot(save_path=str(out_path_dpi), show=False, save_dpi=300)
    assert os.path.isfile(out_path_dpi)
    assert os.path.getsize(out_path_dpi) > 0


def test_save_or_show_plot_custom_figure_size(tmp_path):
    """Test save_or_show_plot with custom figure size."""
    plt.figure()
    plt.plot([1, 2, 3], [3, 1, 2])
    out_path_size = tmp_path / "test_plot_size.png"
    save_or_show_plot(save_path=str(out_path_size), show=False, fig_size=(10, 6))
    assert os.path.isfile(out_path_size)


def test_save_or_show_plot_bbox_and_padding(tmp_path):
    """Test save_or_show_plot with bbox_inches and pad_inches."""
    plt.figure()
    plt.plot([1, 2, 3], [1, 3, 2])
    out_path_bbox = tmp_path / "test_plot_bbox.png"
    save_or_show_plot(save_path=str(out_path_bbox), show=False, bbox_inches="tight", pad_inches=0.1)
    assert os.path.isfile(out_path_bbox)


def test_save_or_show_plot_extra_matplotlib_params(tmp_path):
    """Test save_or_show_plot with additional matplotlib savefig parameters."""
    plt.figure()
    plt.plot([1, 2, 3], [2, 1, 3])
    out_path_extra = tmp_path / "test_plot_extra.png"
    save_or_show_plot(save_path=str(out_path_extra), show=False, facecolor="white", edgecolor="black")
    assert os.path.isfile(out_path_extra)


def test_save_or_show_plot_show_only_mode():
    """Test save_or_show_plot in show-only mode (no save)."""
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    # This should not raise an error and should not create any file
    save_or_show_plot(save_path=None, show=False)  # show=False to avoid display in test


def test_save_or_show_plot_invalid_fig_size_length():
    """Test save_or_show_plot with invalid fig_size length."""
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="fig_size must be a 2-element tuple or list"):
        save_or_show_plot(show=False, fig_size=(10,))  # Invalid: only one element


def test_save_or_show_plot_invalid_fig_size_type():
    """Test save_or_show_plot with invalid fig_size type."""
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError, match="fig_size must be a 2-element tuple or list"):
        save_or_show_plot(show=False, fig_size=10)  # Invalid: not tuple/list


def test_save_or_show_plot_different_file_formats(tmp_path):
    """Test save_or_show_plot with different file formats."""
    for ext in [".jpg", ".pdf", ".svg"]:
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        out_path_format = tmp_path / f"test_plot{ext}"
        save_or_show_plot(save_path=str(out_path_format), show=False)
        assert os.path.isfile(out_path_format)
        assert os.path.getsize(out_path_format) > 0
