import logging

import numpy as np
import pandas as pd
import pytest
import torch

from kale.prepdata.tabular_transform import (
    apply_confidence_inversion,
    generate_struct_for_qbin,
    ToOneHotEncoding,
    ToTensor,
)
from kale.utils.seed import set_seed


class TestToTensor:
    def test_to_tensor_output(self):
        data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        to_tensor = ToTensor()
        output = to_tensor(data)
        assert isinstance(output, torch.Tensor)

    def test_to_tensor_dtype(self):
        data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        to_tensor = ToTensor(dtype=torch.float32)
        output = to_tensor(data)
        assert output.dtype == torch.float32

    def test_to_tensor_device(self):
        data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        to_tensor = ToTensor(device=torch.device("cpu"))
        output = to_tensor(data)
        assert output.device == torch.device("cpu")


class TestToOneHotEncoding:
    @pytest.mark.parametrize("num_classes", [3, -1])
    def test_onehot_encoding_output(self, num_classes: int):
        labels = [1, 0, 2]
        to_onehot = ToOneHotEncoding(num_classes=num_classes)
        output = to_onehot(labels)
        assert output.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 1]]

    @pytest.mark.parametrize("num_classes, shape", [(3, (3, 3)), (4, (3, 4)), (5, (3, 5))])
    def test_onehot_encoding_shape(self, num_classes: int, shape: tuple):
        labels = [1, 0, 2]
        to_onehot = ToOneHotEncoding(num_classes=num_classes)
        output = to_onehot(labels)
        assert output.shape == shape

    def test_onehot_encoding_dtype(self):
        data = [1, 0, 2]
        to_onehot = ToOneHotEncoding(dtype=torch.float32)
        output = to_onehot(data)
        assert output.dtype == torch.float32

    def test_onehot_encoding_device(self):
        data = [1, 0, 2]
        to_onehot = ToOneHotEncoding(device=torch.device("cpu"))
        output = to_onehot(data)
        assert output.device == torch.device("cpu")


seed = 36
set_seed(seed)

LOGGER = logging.getLogger(__name__)

EXPECTED_COLS = [
    "uid",
    "E-CPV Error",
    "E-CPV Uncertainty",
    "E-CPV Uncertainty bins",
    "E-MHA Error",
    "E-MHA Uncertainty",
    "E-MHA Uncertainty bins",
    "S-MHA Error",
    "S-MHA Uncertainty",
    "S-MHA Uncertainty bins",
    "Validation Fold",
    "Testing Fold",
    "target_idx",
]

DUMMY_DICT = pd.DataFrame({"data": [0.1, 0.2, 0.9, 1.5]})


@pytest.mark.parametrize("input, expected", [(DUMMY_DICT, [1 / 0.1, 1 / 0.2, 1 / 0.9, 1 / 1.5])])
def test_apply_confidence_inversion(input, expected):

    # test that it inverts correctly
    assert list(apply_confidence_inversion(input, "data")["data"]) == pytest.approx(expected)

    # test that a KeyError is raised successfully if key not in dict.
    with pytest.raises(KeyError, match=r".* key .*"):
        apply_confidence_inversion({}, "data") == pytest.approx(expected)


# Test that we can read csvs in the correct structure and return a dict of pandas dataframes in correct structure.
def test_get_data_struct(landmark_uncertainty_tuples_path):

    bins_all_targets, bins_targets_sep, bounds_all_targets, bounds_targets_sep = generate_struct_for_qbin(
        ["U-NET"], [0, 1], landmark_uncertainty_tuples_path[2], "SA"
    )

    assert isinstance(bins_all_targets, dict)
    assert isinstance(bins_all_targets["U-NET"], pd.DataFrame)
    assert sorted(list(bins_all_targets["U-NET"].keys())) == sorted(EXPECTED_COLS)
    assert isinstance(bins_targets_sep, dict)
    assert list(bins_targets_sep.keys()) == ["U-NET L0", "U-NET L1"]
    assert sorted(list(bins_targets_sep["U-NET L0"].keys())) == sorted(EXPECTED_COLS)

    assert isinstance(bounds_all_targets, dict)
    assert isinstance(bounds_targets_sep, dict)
