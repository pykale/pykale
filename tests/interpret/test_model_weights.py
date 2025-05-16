import numpy as np
import pytest

from kale.interpret import model_weights


def test_select_top_weight_vector():
    weights = np.array([1, -3, 0.5, 4, -2])
    selected = model_weights.select_top_weight(weights, select_ratio=0.4)
    assert np.count_nonzero(selected) == 2
    assert selected.shape == weights.shape


def test_get_pairwise_rois_output():
    rois = ["A", "B", "C"]
    pairs = model_weights._get_pairwise_rois(rois)
    expected = np.array([("A", "B"), ("A", "C"), ("B", "C")], dtype=object)
    assert (pairs == expected).all()


def test_get_top_symmetric_weight_shape_and_symmetry():
    weights = np.array([0.9, 0.1, -0.5])
    labels = ["A", "B", "C"]
    coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    sym_w, sym_labels, sym_coords = model_weights.get_top_symmetric_weight(weights, labels, coords, p=0.5)

    assert sym_w.shape[0] == sym_w.shape[1]
    assert np.allclose(sym_w, sym_w.T)
    assert len(sym_labels) == len(sym_coords)


def test_get_top_symmetric_weight_empty_if_p_zero():
    weights = np.array([1.0, 0.5, 0.2])
    labels = ["X", "Y", "Z"]
    coords = np.random.rand(3, 3)
    with pytest.raises(ValueError):
        model_weights.get_top_symmetric_weight(weights, labels, coords, p=1e-6)
