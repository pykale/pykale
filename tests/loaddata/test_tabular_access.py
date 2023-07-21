import logging

import numpy as np
import pytest

from kale.loaddata.tabular_access import load_csv_columns

# from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

# import os


seed = 36
set_seed(seed)

LOGGER = logging.getLogger(__name__)

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


@pytest.mark.parametrize(
    "return_columns",
    [
        ("All", EXPECTED_COLS),
        ([], []),
        ("S-MHA Error", ["S-MHA Error"]),
        (["S-MHA Error", "E-MHA Error"], ["S-MHA Error", "E-MHA Error"]),
    ],
)
def test_load_csv_columns_cols_return(landmark_uncertainty_tuples_path, return_columns):
    returned_cols = load_csv_columns(
        landmark_uncertainty_tuples_path[0], "Testing Fold", np.arange(8), cols_to_return=return_columns[0],
    )
    assert list(returned_cols.columns) == return_columns[1]


# Ensure getting a single fold works
@pytest.mark.parametrize("folds", [0])
def test_load_csv_columns_single_fold(landmark_uncertainty_tuples_path, folds):
    returned_single_fold = load_csv_columns(
        landmark_uncertainty_tuples_path[0],
        "Validation Fold",
        folds,
        cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
    )
    assert list(returned_single_fold["Validation Fold"]).count(folds) == len(
        list(returned_single_fold["Validation Fold"])
    )


# Ensure getting a list of folds only return those folds and
# Ensure all samples are being returned
@pytest.mark.parametrize("folds", [[3, 1, 2]])
def test_load_csv_columns_multiple_folds(landmark_uncertainty_tuples_path, folds):
    returned_list_of_folds = load_csv_columns(
        landmark_uncertainty_tuples_path[0],
        "Validation Fold",
        folds,
        cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
    )

    assert all(elem in folds for elem in list(returned_list_of_folds["Validation Fold"]))

    assert len(returned_list_of_folds.index) == 114
