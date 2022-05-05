import logging

import pandas as pd
import pytest

from kale.prepdata.tabular_transform import apply_confidence_inversion, get_data_struct
from kale.utils.seed import set_seed

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
    "landmark",
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

    bins_all_lms, bins_lms_sep, bounds_all_lms, bounds_lms_sep = get_data_struct(
        ["U-NET"], [0, 1], landmark_uncertainty_tuples_path[2], "SA"
    )

    assert isinstance(bins_all_lms, dict)
    assert isinstance(bins_all_lms["U-NET"], pd.DataFrame)
    assert sorted(list(bins_all_lms["U-NET"].keys())) == sorted(EXPECTED_COLS)
    assert isinstance(bins_lms_sep, dict)
    assert list(bins_lms_sep.keys()) == ["U-NET L0", "U-NET L1"]
    assert sorted(list(bins_lms_sep["U-NET L0"].keys())) == sorted(EXPECTED_COLS)

    assert isinstance(bounds_all_lms, dict)
    assert isinstance(bounds_lms_sep, dict)
