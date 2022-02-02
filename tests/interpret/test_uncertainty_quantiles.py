import logging

import numpy as np
import pytest

from kale.interpret.uncertainty_quantiles import quantile_binning_and_est_errors
from kale.loaddata.tabular_access import load_csv_columns

# from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

# import os
LOGGER = logging.getLogger(__name__)


seed = 36
set_seed(seed)

ERRORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UNCERTAINTIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# UNCERTAINTIES = [0, 0.1,0.2,0.]


@pytest.fixture(scope="module")
def dummy_test_data(landmark_uncertainty_dl):
    dummy_tabular_data_dict = load_csv_columns(
        landmark_uncertainty_dl[1], "Testing Fold", np.arange(0), cols_to_return="All"
    )

    dummy_errors = dummy_tabular_data_dict["S-MHA Error"].values
    dummy_uncertainties = dummy_tabular_data_dict["S-MHA Error"].values

    return dummy_errors, dummy_uncertainties


class TestQuantileBinningAndEstErrors:
    def test_empty(self):
        with pytest.raises(ValueError, match=r"Length of errors .*"):
            quantile_binning_and_est_errors(ERRORS, [0, 1, 2], num_bins=5)
        with pytest.raises(ValueError, match=r"Length of errors .*"):
            quantile_binning_and_est_errors([], [0, 1, 2], num_bins=5)

    # Using 11 datapoints from 0-N, we test if we can create 10 bins between these:
    # <0.1, 0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8,0.9, >0.9
    # same logic with expected errors
    def test_dummy_1(self):
        est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=10)

        assert pytest.approx(np.squeeze(est_bounds)) == UNCERTAINTIES[1:-1]
        assert pytest.approx(np.squeeze(est_errors)) == ERRORS[1:-1]


# # Ensure getting a single fold works
# @pytest.mark.parametrize("source_test_file", ["PHD-Net/4CH/uncertainty_pairs_test_l0"])
# @pytest.mark.parametrize("folds", [0])
# def test_load_csv_columns_single_fold(landmark_uncertainty_dl, source_test_file, folds):

#     returned_single_fold = load_csv_columns(
#         landmark_uncertainty_dl[0],
#         "Validation Fold",
#         folds,
#         cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
#     )
#     assert list(returned_single_fold["Validation Fold"]).count(folds) == len(
#         list(returned_single_fold["Validation Fold"])
#     )


# # Ensure getting a list of folds only return those folds and
# # Ensure all samples are being returned
# @pytest.mark.parametrize("folds", [[3, 1, 2]])
# def test_load_csv_columns_multiple_folds(landmark_uncertainty_dl, folds):
#     returned_list_of_folds = load_csv_columns(
#         landmark_uncertainty_dl[0],
#         "Validation Fold",
#         folds,
#         cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
#     )

#     assert all(elem in folds for elem in list(returned_list_of_folds["Validation Fold"]))

#     assert len(returned_list_of_folds.index) == 114
