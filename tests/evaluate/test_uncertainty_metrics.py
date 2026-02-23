import logging

import numpy as np
import pandas as pd
import pytest

from kale.evaluate.uncertainty_metrics import (
    ColumnNames,
    DataProcessor,
    evaluate_bounds,
    evaluate_jaccard,
    EvaluationConfig,
    FoldData,
    JaccardBinResults,
    JaccardEvaluator,
    MetricsCalculator,
    QuantileCalculator,
    ResultsContainer,
)
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
        all_jaccard_data = jacc_dict["jaccard_all"]
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

        all_jaccard_data = jacc_dict["jaccard_all"]
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

        all_jaccard_data = jacc_dict["jaccard_all"]
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

        all_bound_percents = bound_dict["error_bounds_all"]
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
        all_bound_percents = bound_dict["error_bounds_all"]
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

        all_bound_percents = bound_dict["error_bounds_all"]
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


class TestEvaluationConfig:
    """Test EvaluationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.num_folds == 8
        assert config.original_num_bins == 10
        assert config.error_scaling_factor == 1.0
        assert config.combine_middle_bins is False
        assert config.combined_num_bins == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            num_folds=5, original_num_bins=15, error_scaling_factor=2.0, combine_middle_bins=True, combined_num_bins=4
        )
        assert config.num_folds == 5
        assert config.original_num_bins == 15
        assert config.error_scaling_factor == 2.0
        assert config.combine_middle_bins is True
        assert config.combined_num_bins == 4


class TestResultsContainer:
    """Test ResultsContainer functionality."""

    def test_initialization(self):
        """Test proper initialization of ResultsContainer."""
        container = ResultsContainer(num_bins=5, num_targets=3)
        assert container.num_bins == 5
        assert container.num_targets == 3
        assert len(container.target_sep_foldwise) == 3
        assert len(container.target_sep_all) == 3
        assert isinstance(container.main_results, dict)
        assert isinstance(container.target_separated_results, dict)

    def test_add_results(self):
        """Test adding results to container."""
        container = ResultsContainer(num_bins=5, num_targets=2)

        # Test adding main result
        test_data = [1, 2, 3, 4, 5]
        container.add_main_result("test_key", test_data)
        assert container.main_results["test_key"] == test_data

        # Test adding target separated result
        target_data = [[1, 2], [3, 4], [5, 6]]
        container.add_target_separated_result("test_key", target_data)
        assert container.target_separated_results["test_key"] == target_data


class TestDataProcessor:
    """Test DataProcessor utility methods."""

    def test_group_data_by_bins(self):
        """Test grouping data by bins."""
        errors_dict = {"pred1": 0.1, "pred2": 0.3, "pred3": 0.2, "pred4": 0.4}
        bins_dict = {"pred1": 0, "pred2": 1, "pred3": 0, "pred4": 1}

        bin_keys, bin_errors = DataProcessor.group_data_by_bins(errors_dict, bins_dict, 2)

        assert len(bin_keys) == 2
        assert len(bin_errors) == 2
        assert set(bin_keys[0]) == {"pred1", "pred3"}
        assert set(bin_keys[1]) == {"pred2", "pred4"}
        assert bin_errors[0] == [0.1, 0.2]
        assert bin_errors[1] == [0.3, 0.4]


class TestQuantileCalculator:
    """Test QuantileCalculator utility methods."""

    def test_calculate_error_quantiles(self):
        """Test quantile threshold calculation."""
        errors_dict = {f"pred{i}": i * 0.1 for i in range(10)}  # 0.0 to 0.9

        thresholds, error_groups, key_groups = QuantileCalculator.calculate_error_quantiles(
            errors_dict, num_bins=5, combine_middle_bins=False
        )

        assert len(thresholds) == 4  # num_bins - 1
        assert len(error_groups) == 5
        assert len(key_groups) == 5

        # Check that groups are ordered from worst to best (reversed)
        assert error_groups[0]  # Worst errors (highest values)
        assert error_groups[-1]  # Best errors (lowest values)

    def test_combine_middle_bins(self):
        """Test combining middle bins functionality."""
        errors_dict = {f"pred{i}": i * 0.1 for i in range(10)}

        thresholds, error_groups, key_groups = QuantileCalculator.calculate_error_quantiles(
            errors_dict, num_bins=5, combine_middle_bins=True
        )

        assert len(thresholds) == 2  # First and last threshold only
        assert len(error_groups) == 3  # Combined to 3 groups
        assert len(key_groups) == 3


class TestMetricsCalculator:
    """Test MetricsCalculator utility methods."""

    def test_calculate_jaccard_metrics(self):
        """Test Jaccard similarity calculation."""
        predicted_keys = ["sample1", "sample2", "sample3"]
        ground_truth_keys = ["sample1", "sample4"]

        jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics(predicted_keys, ground_truth_keys)

        # Expected: intersection = {'sample1'}, union = {'sample1', 'sample2', 'sample3', 'sample4'}
        # Jaccard = 1/4 = 0.25, Recall = 1/2 = 0.5, Precision = 1/3 ≈ 0.33
        assert jaccard == 0.25
        assert recall == 0.5
        assert abs(precision - 1 / 3) < 1e-10

    def test_empty_sets(self):
        """Test edge cases with empty sets."""
        # Both empty
        jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics([], [])
        assert jaccard == 0.0
        assert recall == 1.0  # Special case
        assert precision == 0.0

        # Empty ground truth
        jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics(["a"], [])
        assert recall == 1.0
        assert precision == 0.0

        # Empty predictions
        jaccard, recall, precision = MetricsCalculator.calculate_jaccard_metrics([], ["a"])
        assert recall == 0.0
        assert precision == 0.0

    def test_calculate_bound_accuracy(self):
        """Test bound accuracy calculation."""
        bounds = [0.1, 0.3, 0.5, 0.7]

        # Test bin 0: (0, 0.1] - errors greater than 0 and up to 0.1
        assert MetricsCalculator.calculate_bound_accuracy(0.05, 0, bounds) is True
        assert MetricsCalculator.calculate_bound_accuracy(0.1, 0, bounds) is True  # Upper bound inclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.0, 0, bounds) is False  # Exactly 0 excluded
        assert MetricsCalculator.calculate_bound_accuracy(0.15, 0, bounds) is False

        # Test middle bin 1: (0.1, 0.3] - errors greater than 0.1 and up to 0.3
        assert MetricsCalculator.calculate_bound_accuracy(0.2, 1, bounds) is True
        assert MetricsCalculator.calculate_bound_accuracy(0.3, 1, bounds) is True  # Upper bound inclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.1, 1, bounds) is False  # Lower bound exclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.35, 1, bounds) is False

        # Test middle bin 2: (0.3, 0.5] - errors greater than 0.3 and up to 0.5
        assert MetricsCalculator.calculate_bound_accuracy(0.4, 2, bounds) is True
        assert MetricsCalculator.calculate_bound_accuracy(0.5, 2, bounds) is True  # Upper bound inclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.3, 2, bounds) is False  # Lower bound exclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.55, 2, bounds) is False

        # Test middle bin 3: (0.5, 0.7] - errors greater than 0.5 and up to 0.7
        assert MetricsCalculator.calculate_bound_accuracy(0.6, 3, bounds) is True
        assert MetricsCalculator.calculate_bound_accuracy(0.7, 3, bounds) is True  # Upper bound inclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.5, 3, bounds) is False  # Lower bound exclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.75, 3, bounds) is False

        # Test last bin 4: (0.7, inf) - errors greater than 0.7
        assert MetricsCalculator.calculate_bound_accuracy(0.8, 4, bounds) is True
        assert MetricsCalculator.calculate_bound_accuracy(0.7, 4, bounds) is False  # Lower bound exclusive
        assert MetricsCalculator.calculate_bound_accuracy(0.6, 4, bounds) is False


class TestJaccardEvaluator:
    """Test JaccardEvaluator class and methods."""

    def test_initialization_default(self):
        """Test default initialization."""
        evaluator = JaccardEvaluator()
        assert evaluator.config_.num_folds == 8
        assert evaluator.config_.original_num_bins == 10
        assert evaluator.config_.combine_middle_bins is False

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = EvaluationConfig(num_folds=5, original_num_bins=7)
        evaluator = JaccardEvaluator(config)
        assert evaluator.config_.num_folds == 5
        assert evaluator.config_.original_num_bins == 7

    def test_create_simple_factory(self):
        """Test create_simple factory method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=6, num_folds=4, combine_middle_bins=True)
        assert evaluator.config_.original_num_bins == 6
        assert evaluator.config_.num_folds == 4
        assert evaluator.config_.combine_middle_bins is True

    def test_create_default_factory(self):
        """Test create_default factory method."""
        evaluator = JaccardEvaluator.create_default()
        assert evaluator.config_.num_folds == 8
        assert evaluator.config_.original_num_bins == 10

    def test_extract_target_data(self):
        """Test _extract_target_data method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=5, num_folds=1)
        evaluator.current_uncertainty_type_ = "epistemic"

        # Create mock fold data
        errors_df = pd.DataFrame(
            {
                ColumnNames.UID: ["uid1", "uid2", "uid3"],
                ColumnNames.TARGET_IDX: [0, 0, 1],
                "epistemic Error": [0.1, 0.2, 0.3],
            }
        )
        bins_df = pd.DataFrame(
            {
                ColumnNames.UID: ["uid1", "uid2", "uid3"],
                ColumnNames.TARGET_IDX: [0, 0, 1],
                "epistemic Uncertainty bins": [1, 2, 1],
            }
        )
        fold_data = FoldData(errors=errors_df, bins=bins_df)

        errors_dict, bins_dict = evaluator._extract_target_data(fold_data, target_idx=0)

        assert len(errors_dict) == 2  # Only target 0 data
        assert "uid1" in errors_dict and "uid2" in errors_dict
        assert "uid3" not in errors_dict  # Different target
        assert errors_dict["uid1"] == 0.1
        assert bins_dict["uid1"] == 1

    def test_get_predicted_bin_keys(self):
        """Test _get_predicted_bin_keys method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=3, num_folds=1)
        evaluator.current_num_bins_ = 3

        errors_dict = {"uid1": 0.1, "uid2": 0.5, "uid3": 0.3}
        bins_dict = {"uid1": 0, "uid2": 2, "uid3": 1}  # Best to worst: 0, 1, 2

        pred_bin_keys = evaluator._get_predicted_bin_keys(errors_dict, bins_dict)

        # Should be reversed (worst to best: B3 to B1)
        assert len(pred_bin_keys) == 3
        assert isinstance(pred_bin_keys, list)
        assert all(isinstance(bin_keys, list) for bin_keys in pred_bin_keys)

    def test_calculate_bin_wise_metrics(self):
        """Test _calculate_bin_wise_metrics method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=3, num_folds=1)
        evaluator.current_num_bins_ = 3

        pred_bin_keys = [["uid1", "uid2"], ["uid3"], []]
        gt_key_groups = [["uid1"], ["uid2", "uid3"], []]

        bin_jaccard, bin_recall, bin_precision = evaluator._calculate_bin_wise_metrics(pred_bin_keys, gt_key_groups)

        assert len(bin_jaccard) == 3
        assert len(bin_recall) == 3
        assert len(bin_precision) == 3
        assert all(isinstance(val, float) for val in bin_jaccard)

    def test_format_target_results(self):
        """Test _format_target_results method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=3, num_folds=1)

        bin_jaccard = [0.5, 0.3, 0.8]
        bin_recall = [0.6, 0.4, 0.9]
        bin_precision = [0.7, 0.5, 0.85]

        results = evaluator._format_target_results(bin_jaccard, bin_recall, bin_precision)

        assert "mean_jaccard" in results
        assert "mean_recall" in results
        assert "mean_precision" in results
        assert "bin_jaccard" in results
        assert "bin_recall" in results
        assert "bin_precision" in results

        assert abs(results["mean_jaccard"] - np.mean(bin_jaccard)) < 1e-10
        assert results["bin_jaccard"] == bin_jaccard

    def test_aggregate_fold_metrics(self):
        """Test _aggregate_fold_metrics method."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=2, num_folds=1)
        evaluator.current_num_bins_ = 2

        # Create mock JaccardBinResults
        results1 = JaccardBinResults(
            mean_all_targets=0.5,
            mean_all_bins=[0.4, 0.6],
            all_bins=[[0.4], [0.6]],
            all_bins_concat_targets_sep=[],
            mean_all_targets_recall=0.6,
            mean_all_bins_recall=[0.5, 0.7],
            all_bins_recall=[[0.5], [0.7]],
            mean_all_targets_precision=0.55,
            mean_all_bins_precision=[0.45, 0.65],
            all_bins_precision=[[0.45], [0.65]],
        )
        results2 = JaccardBinResults(
            mean_all_targets=0.7,
            mean_all_bins=[0.6, 0.8],
            all_bins=[[0.6], [0.8]],
            all_bins_concat_targets_sep=[],
            mean_all_targets_recall=0.8,
            mean_all_bins_recall=[0.7, 0.9],
            all_bins_recall=[[0.7], [0.9]],
            mean_all_targets_precision=0.75,
            mean_all_bins_precision=[0.65, 0.85],
            all_bins_precision=[[0.65], [0.85]],
        )

        aggregated = evaluator._aggregate_fold_metrics([results1, results2])

        assert "fold_jaccard_bins" in aggregated
        assert "fold_recall_bins" in aggregated
        assert "fold_precision_bins" in aggregated
        assert len(aggregated["fold_jaccard_bins"]) == 2  # num_bins
        assert len(aggregated["fold_jaccard_bins"][0]) == 2  # num_folds

    def test_evaluate_integration(self, dummy_test_preds):
        """Test integration with actual data using new evaluator interface."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=5, num_folds=8)

        results = evaluator.evaluate(bin_predictions=dummy_test_preds[0], uncertainty_pairs=[["S-MHA"]], targets=[0, 1])

        # Check that results match expected structure
        assert "jaccard_all" in results
        assert "Jaccard targets seperated" in results
        assert "recall_all" in results
        assert "precision_all" in results

        # Check specific keys and structure
        jaccard_all = results["jaccard_all"]
        assert "U-NET S-MHA" in jaccard_all
        assert len(jaccard_all["U-NET S-MHA"]) == 5  # 5 bins

    @pytest.mark.parametrize("num_bins", [3, 5, 7])
    def test_different_bin_numbers(self, dummy_test_preds, num_bins):
        """Test evaluator with different numbers of bins."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=num_bins, num_folds=8)

        results = evaluator.evaluate(bin_predictions=dummy_test_preds[0], uncertainty_pairs=[["S-MHA"]], targets=[0, 1])

        jaccard_all = results["jaccard_all"]
        assert len(jaccard_all["U-NET S-MHA"]) == num_bins

    def test_combine_middle_bins(self, dummy_test_preds):
        """Test evaluator with combined middle bins."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=10, num_folds=8, combine_middle_bins=True)

        results = evaluator.evaluate(bin_predictions=dummy_test_preds[0], uncertainty_pairs=[["S-MHA"]], targets=[0, 1])

        jaccard_all = results["jaccard_all"]
        # When combine_middle_bins=True, should have 3 bins
        assert len(jaccard_all["U-NET S-MHA"]) == 3

    def test_multiple_uncertainty_types(self, dummy_test_preds):
        """Test evaluator with multiple uncertainty types."""
        evaluator = JaccardEvaluator.create_simple(original_num_bins=5, num_folds=8)

        results = evaluator.evaluate(
            bin_predictions=dummy_test_preds[0], uncertainty_pairs=[["S-MHA"], ["E-MHA"]], targets=[0, 1]
        )

        jaccard_all = results["jaccard_all"]
        assert "U-NET S-MHA" in jaccard_all
        assert "U-NET E-MHA" in jaccard_all
        assert len(jaccard_all["U-NET S-MHA"]) == 5
        assert len(jaccard_all["U-NET E-MHA"]) == 5


class TestBackwardCompatibility:
    """Test backward compatibility between old functional interface and new class interface."""

    def test_jaccard_results_consistency(self, dummy_test_preds):
        """Test that old and new interfaces produce consistent results."""
        # Old functional interface
        old_results = evaluate_jaccard(dummy_test_preds[0], [["S-MHA"]], 5, [0, 1], num_folds=8)

        # New class interface
        evaluator = JaccardEvaluator.create_simple(original_num_bins=5, num_folds=8)
        new_results = evaluator.evaluate(
            bin_predictions=dummy_test_preds[0], uncertainty_pairs=[["S-MHA"]], targets=[0, 1]
        )

        # Compare key results (allowing for small numerical differences)
        old_jaccard = old_results["jaccard_all"]["U-NET S-MHA"]
        new_jaccard = new_results["jaccard_all"]["U-NET S-MHA"]

        assert len(old_jaccard) == len(new_jaccard)
        for old_bin, new_bin in zip(old_jaccard, new_jaccard):
            for old_val, new_val in zip(old_bin, new_bin):
                assert abs(old_val - new_val) < 1e-10


class TestFoldData:
    """Test FoldData dataclass."""

    def test_fold_data_creation(self):
        """Test FoldData creation and attributes."""
        errors_df = pd.DataFrame({"uid": [1, 2, 3], "Target Index": [0, 1, 0], "S-MHA Error": [0.1, 0.2, 0.3]})

        bins_df = pd.DataFrame({"uid": [1, 2, 3], "Target Index": [0, 1, 0], "S-MHA Uncertainty bins": [0, 1, 2]})

        fold_data = FoldData(errors=errors_df, bins=bins_df)

        assert fold_data.errors.equals(errors_df)
        assert fold_data.bins.equals(bins_df)
        assert fold_data.bounds is None

        # Test with bounds
        bounds = [0.1, 0.2, 0.3]
        fold_data_with_bounds = FoldData(errors=errors_df, bins=bins_df, bounds=bounds)
        assert fold_data_with_bounds.bounds == bounds


class TestJaccardBinResults:
    """Test JaccardBinResults dataclass."""

    def test_jaccard_bin_results_creation(self):
        """Test JaccardBinResults creation and inheritance."""
        results = JaccardBinResults(
            mean_all_targets=0.5,
            mean_all_bins=[0.4, 0.5, 0.6],
            all_bins=[[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            all_bins_concat_targets_sep=[[[0.3], [0.4]], [[0.5], [0.6]]],
            mean_all_targets_recall=0.6,
            mean_all_bins_recall=[0.5, 0.6, 0.7],
            all_bins_recall=[[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]],
            mean_all_targets_precision=0.7,
            mean_all_bins_precision=[0.6, 0.7, 0.8],
            all_bins_precision=[[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        )

        # Test base class attributes
        assert results.mean_all_targets == 0.5
        assert results.mean_all_bins == [0.4, 0.5, 0.6]

        # Test Jaccard-specific attributes
        assert results.mean_all_targets_recall == 0.6
        assert results.mean_all_bins_recall == [0.5, 0.6, 0.7]
        assert results.mean_all_targets_precision == 0.7
        assert results.mean_all_bins_precision == [0.6, 0.7, 0.8]
