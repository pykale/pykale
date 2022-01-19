import os

import numpy as np
import pytest

from kale.loaddata.tabular_access import load_csv_columns
from kale.utils.seed import set_seed

seed = 36
set_seed(seed)

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
url = "https://github.com/pykale/data/blob/landmark-data/tabular/cardiac_landmark_uncertainty/Uncertainty_tuples.zip?raw=true"


EXPECTED_COLS = [
    "uid",
    "E-CPV Error",
    "E-CPV Uncertainty",
    "E-MHA Error",
    "E-MHA Uncertainty",
    "S-MHA Error",
    "S-MHA Uncertainty",
    "Validation Fold",
    "Testing Fold",
]


@pytest.mark.parametrize("source_test_file", ["PHD-Net/4CH/uncertainty_pairs_test_l0"])
def test_load_csv_columns(landmark_uncertainty_dl, source_test_file):

    # ensure if cols_to_return is "All" that all columns are returned
    returned_cols = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file), "Testing Fold", np.arange(8), cols_to_return="All"
    )
    assert list(returned_cols.columns) == EXPECTED_COLS

    # ensure if cols_to_return is an empty dataframe
    returned_empty_cols = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file), "Testing Fold", np.arange(8), cols_to_return=[]
    )
    assert returned_empty_cols.empty

    # ensure if cols_to_return is a single value, not in list it works
    returned_smha = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file),
        "Testing Fold",
        np.arange(8),
        cols_to_return="S-MHA Error",
    )

    assert list(returned_smha.columns) == ["S-MHA Error"]

    # ensure a list of columns work
    returned_multiple = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file),
        "Testing Fold",
        np.arange(8),
        cols_to_return=["S-MHA Error", "E-MHA Error"],
    )

    assert list(returned_multiple.columns) == ["S-MHA Error", "E-MHA Error"]

    # Ensure getting a single fold works
    returned_single_fold = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file),
        "Testing Fold",
        0,
        cols_to_return=["S-MHA Error", "E-MHA Error", "Testing Fold"],
    )
    assert list(returned_single_fold["Testing Fold"]).count(0) == len(list(returned_single_fold["Testing Fold"]))

    # Ensure getting a list of folds only return those folds and
    returned_list_of_folds = load_csv_columns(
        os.path.join(landmark_uncertainty_dl, source_test_file),
        "Validation Fold",
        [0, 1, 2],
        cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
    )
    assert all(elem in [0, 1, 2] for elem in list(returned_list_of_folds["Validation Fold"]))

    # Ensure all samples are being returned
    assert len(returned_list_of_folds.index) == 159
