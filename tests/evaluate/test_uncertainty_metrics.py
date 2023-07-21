import logging

import pytest

from kale.evaluate.uncertainty_metrics import evaluate_bounds, evaluate_jaccard
from kale.prepdata.tabular_transform import generate_struct_for_qbin

# from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

# import os
LOGGER = logging.getLogger(__name__)


seed = 36
set_seed(seed)

ERRORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UNCERTAINTIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


@pytest.fixture(scope="module")
def dummy_test_preds(landmark_uncertainty_tuples_path):
    bins_all_targets, bins_targets_sep, bounds_all_targets, bounds_targets_sep = generate_struct_for_qbin(
        ["U-NET"], [0, 1], landmark_uncertainty_tuples_path[2], "SA"
    )

    return bins_all_targets, bounds_all_targets


class TestEvaluateJaccard:

    # Using one uncertainty type, test numerous bins
    @pytest.mark.parametrize("num_bins", [2, 3, 4, 5])
    def test_one_uncertainty(self, dummy_test_preds, num_bins):
        jacc_dict = evaluate_jaccard(
            dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], num_bins, [0, 1], num_folds=8
        )
        all_jaccard_data = jacc_dict["Jaccard All"]
        all_jaccard_bins_targets_sep = jacc_dict["Jaccard targets seperated"]

        assert list(all_jaccard_data.keys()) == ["U-NET S-MHA"]
        assert len(all_jaccard_data["U-NET S-MHA"]) == num_bins

        assert list(all_jaccard_bins_targets_sep.keys()) == ["U-NET S-MHA"]
        assert len(all_jaccard_bins_targets_sep["U-NET S-MHA"]) == num_bins
        assert (
            len(all_jaccard_bins_targets_sep["U-NET S-MHA"][0]) == 8 * 2
        )  # because each landmark has 8 folds - they are seperate

    def test_one_fold(self, dummy_test_preds):
        jacc_dict = evaluate_jaccard(
            dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], 5, [0, 1], num_folds=1
        )

        all_jaccard_data = jacc_dict["Jaccard All"]
        all_jaccard_bins_targets_sep = jacc_dict["Jaccard targets seperated"]

        assert list(all_jaccard_data.keys()) == ["U-NET S-MHA"]
        assert len(all_jaccard_data["U-NET S-MHA"]) == 5

        assert list(all_jaccard_bins_targets_sep.keys()) == ["U-NET S-MHA"]
        assert len(all_jaccard_bins_targets_sep["U-NET S-MHA"]) == 5
        assert (
            len(all_jaccard_bins_targets_sep["U-NET S-MHA"][0]) == 2
        )  # because each landmark has 1 folds - they are sep

    def test_multiple_uncerts(self, dummy_test_preds):
        jacc_dict = evaluate_jaccard(
            dummy_test_preds[0],
            [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"], ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"]],
            5,
            [0, 1],
            num_folds=1,
        )

        all_jaccard_data = jacc_dict["Jaccard All"]
        all_jaccard_bins_targets_sep = jacc_dict["Jaccard targets seperated"]

        assert list(all_jaccard_data.keys()) == ["U-NET S-MHA", "U-NET E-MHA"]
        assert len(all_jaccard_data["U-NET S-MHA"]) == len(all_jaccard_data["U-NET E-MHA"]) == 5

        assert list(all_jaccard_bins_targets_sep.keys()) == ["U-NET S-MHA", "U-NET E-MHA"]
        assert len(all_jaccard_bins_targets_sep["U-NET S-MHA"]) == len(all_jaccard_bins_targets_sep["U-NET E-MHA"]) == 5
        assert (
            len(all_jaccard_bins_targets_sep["U-NET S-MHA"][0])
            == len(all_jaccard_bins_targets_sep["U-NET E-MHA"][0])
            == 2
        )  # because each landmark has 8 folds - they are sep


class TestEvaluateBounds:
    @pytest.mark.parametrize("num_bins", [2, 3, 4, 5])
    def test_one_uncertainty(self, dummy_test_preds, num_bins):
        bound_dict = evaluate_bounds(
            dummy_test_preds[1],
            dummy_test_preds[0],
            [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]],
            num_bins,
            [0, 1],
            num_folds=8,
        )

        all_bound_percents = bound_dict["Error Bounds All"]
        all_bound_percents_notargetsep = bound_dict["all_bound_percents_notargetsep"]

        assert list(all_bound_percents.keys()) == ["U-NET S-MHA"]
        assert len(all_bound_percents["U-NET S-MHA"]) == num_bins

        assert list(all_bound_percents_notargetsep.keys()) == ["U-NET S-MHA"]
        assert len(all_bound_percents_notargetsep["U-NET S-MHA"]) == num_bins
        assert (
            len(all_bound_percents_notargetsep["U-NET S-MHA"][0]) == 8 * 2
        )  # because each landmark has 8 folds - they are seperate

    def test_one_fold(self, dummy_test_preds):
        bound_dict = evaluate_bounds(
            dummy_test_preds[1],
            dummy_test_preds[0],
            [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]],
            5,
            [0, 1],
            num_folds=1,
        )
        all_bound_percents = bound_dict["Error Bounds All"]
        all_bound_percents_notargetsep = bound_dict["all_bound_percents_notargetsep"]

        assert list(all_bound_percents.keys()) == ["U-NET S-MHA"]
        assert len(all_bound_percents["U-NET S-MHA"]) == 5

        assert list(all_bound_percents_notargetsep.keys()) == ["U-NET S-MHA"]
        assert len(all_bound_percents_notargetsep["U-NET S-MHA"]) == 5
        assert (
            len(all_bound_percents_notargetsep["U-NET S-MHA"][0]) == 2
        )  # because each landmark has 1 folds - they are sep

    def test_multiple_uncerts(self, dummy_test_preds):

        bound_dict = evaluate_bounds(
            dummy_test_preds[1],
            dummy_test_preds[0],
            [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"], ["E-MHA", "E-MHA Error", "E-MHA Uncertainty"]],
            5,
            [0, 1],
            num_folds=8,
        )

        all_bound_percents = bound_dict["Error Bounds All"]
        all_bound_percents_notargetsep = bound_dict["all_bound_percents_notargetsep"]

        assert list(all_bound_percents.keys()) == ["U-NET S-MHA", "U-NET E-MHA"]
        assert len(all_bound_percents["U-NET S-MHA"]) == len(all_bound_percents["U-NET E-MHA"]) == 5

        assert list(all_bound_percents_notargetsep.keys()) == ["U-NET S-MHA", "U-NET E-MHA"]
        assert (
            len(all_bound_percents_notargetsep["U-NET S-MHA"])
            == len(all_bound_percents_notargetsep["U-NET E-MHA"])
            == 5
        )
        assert (
            len(all_bound_percents_notargetsep["U-NET S-MHA"][0])
            == len(all_bound_percents_notargetsep["U-NET E-MHA"][0])
            == 8 * 2
        )  # because each landmark has 8 folds - they are sep
