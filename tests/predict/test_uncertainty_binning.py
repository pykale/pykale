# import logging

# import numpy as np
# import pandas as pd
# import pytest

# from kale.loaddata.tabular_access import load_csv_columns
# from kale.predict.uncertainty_binning import quantile_binning_predictions
# from kale.utils.seed import set_seed

# seed = 36
# set_seed(seed)

# LOGGER = logging.getLogger(__name__)

# # dummy_tabular_data = {
# #     "uid": ["PHD_2154", "PHD_2158", "PHD_217", "PHD_2194"],
# #     "E-CPV Error": [1.4142135, 3.1622777, 5.0990195, 61.846584],
# #     "E-CPV Uncertainty": [4.25442667, 4.449976897, 1.912124681, 35.76085777],
# #     "E-MHA Error": [3.1622777, 3.1622777, 4, 77.00649],
# #     "E-MHA Uncertainty": [0.331125357, 0.351173535, 1.4142135, 0.142362904],
# #     "S-MHA Error": [3.1622777, 1.4142135, 5.0990195, 56.32051],
# #     "S-MHA Uncertainty": [0.500086973, 0.235296882, 1.466040241, 0.123874651],
# #     "Validation Fold": [1, 1, 1, 1],
# #     "Testing Fold": [0, 0, 0, 0],
# # }


# UNCERTAINTY_THRESH_LISTS_INP = [[], [0.2, 0.4, 0.6, 0.8]]
# UNCERTAINTY_THRESH_LISTS_EXP = [{}, [0.2, 0.4, 0.6, 0.8]]

# UNCERTAINTY_TEST_LISTS = [[], [0, 0.1, 0.19], [0.3, 0.4, 0.9]]


# @pytest.fixture(scope="module")
# def dummy_test_data(landmark_uncertainty_dl):
#     dummy_tabular_data_dict = load_csv_columns(
#         landmark_uncertainty_dl[1], "Testing Fold", np.arange(0), cols_to_return="All"
#     )

#     return dummy_tabular_data_dict


# @pytest.mark.parametrize(
#     # "uncertainty_thresh_list, expected", [([], {}), ([1, 3, 5, 9], {"PHD_2154": 1, "PHD_2158": 2, "PHD_217": 3, "PHD_2194": 4})]
#     "uncertainty_thresh_list, expected",
#     [([1, 3, 5, 9], {"PHD_2154": 1, "PHD_2158": 2, "PHD_217": 3, "PHD_2194": 4})],
# )
# def test_quantile_binning_predictions_thresh(dummy_tabular_data, uncertainty_thresh_list, expected):

#     test_dict = pd.DataFrame(dict(zip(dummy_tabular_data.uid, dummy_tabular_data["E-CPV Uncertainty"])))

#     LOGGER.info("results are:  %s " % quantile_binning_predictions(test_dict, uncertainty_thresh_list))
#     assert quantile_binning_predictions(test_dict, uncertainty_thresh_list) == expected

#     # assert all_binned_errors ==


# # @pytest.mark.parametrize(
# #     "uncertainty_test_list", UNCERTAINTY_TEST_LISTS
# # )
# # def test_quantile_binning_predictions_tests(uncertainty_test_list):

# #     # download_file_by_url(landmark_uncertainty_url, landmark_uncertainty_dl, "Uncertainty_tuples.zip", "zip")
# #     all_binned_errors = quantile_binning_predictions(uncertainty_test_list, uncertainty_thresh_list )

# #     assert list(returned_cols.columns) == return_columns[1]

# # # Ensure getting a single fold works
# # @pytest.mark.parametrize("source_test_file", ["PHD-Net/4CH/uncertainty_pairs_test_l0"])
# # @pytest.mark.parametrize("folds", [0])
# # def test_load_csv_columns_single_fold(landmark_uncertainty_dl, source_test_file, folds):

# #     returned_single_fold = load_csv_columns(
# #         landmark_uncertainty_dl,
# #         "Validation Fold",
# #         folds,
# #         cols_to_return=["S-MHA Error", "E-MHA Error", "Validation Fold"],
# #     )
# #     assert list(returned_single_fold["Validation Fold"]).count(folds) == len(
# #         list(returned_single_fold["Validation Fold"])
# #     )
