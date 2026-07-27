import pytest

from kale.evaluate.uncertainty_metrics import (
    BaseEvaluator,
    BoundsEvaluator,
    ErrorsEvaluator,
    evaluate_bounds,
    EvaluationConfig,
    get_mean_errors,
)
from kale.prepdata.tabular_transform import generate_struct_for_qbin
from kale.utils.seed import set_seed

set_seed(36)


@pytest.fixture(scope="module")
def dummy_test_preds(landmark_uncertainty_tuples_path):
    bins_all_targets, bins_targets_sep, bounds_all_targets, bounds_targets_sep = generate_struct_for_qbin(
        ["U-NET"], [0, 1], landmark_uncertainty_tuples_path[2], "SA"
    )

    return bins_all_targets, bounds_all_targets


class TestBoundsAndErrorsEvaluators:
    """The evaluator classes behind evaluate_bounds and get_mean_errors (issue #554)."""

    def test_bounds_evaluator_matches_the_function_wrapper(self, dummy_test_preds):
        """BoundsEvaluator produces what evaluate_bounds returns, since the function delegates to it."""
        config = EvaluationConfig(num_folds=8, original_num_bins=5)
        evaluator = BoundsEvaluator(dummy_test_preds[1], config)

        direct = evaluator.evaluate(dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], [0, 1])
        via_wrapper = evaluate_bounds(
            dummy_test_preds[1],
            dummy_test_preds[0],
            [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]],
            5,
            [0, 1],
            num_folds=8,
        )

        assert direct.keys() == via_wrapper.keys()
        assert direct["error_bounds_all"] == via_wrapper["error_bounds_all"]

    def test_errors_evaluator_matches_the_function_wrapper(self, dummy_test_preds):
        """ErrorsEvaluator produces what get_mean_errors returns, since the function delegates to it."""
        config = EvaluationConfig(num_folds=8, original_num_bins=5)
        evaluator = ErrorsEvaluator(config)

        direct = evaluator.evaluate(dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], [0, 1])
        via_wrapper = get_mean_errors(
            dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], 5, [0, 1], num_folds=8
        )

        assert direct.keys() == via_wrapper.keys()
        assert direct["all_mean_error_bins_nosep"] == via_wrapper["all_mean_error_bins_nosep"]

    def test_evaluators_derive_from_the_shared_base(self):
        """Both evaluators plug into the same template as JaccardEvaluator."""
        assert issubclass(BoundsEvaluator, BaseEvaluator)
        assert issubclass(ErrorsEvaluator, BaseEvaluator)

    def test_combine_middle_bins_reduces_the_bin_count(self, dummy_test_preds):
        """combine_middle_bins collapses the middle bins, leaving the configured combined count."""
        config = EvaluationConfig(num_folds=1, original_num_bins=10, combine_middle_bins=True)
        evaluator = ErrorsEvaluator(config)

        results = evaluator.evaluate(dummy_test_preds[0], [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]], [0, 1])

        assert len(results["all_mean_error_bins_nosep"]["U-NET S-MHA"]) == config.combined_num_bins

    def test_error_scaling_factor_scales_the_reported_errors(self, dummy_test_preds):
        """The configured scaling factor is applied to the mean errors."""
        pair = [["S-MHA", "S-MHA Error", "S-MHA Uncertainty"]]
        base = ErrorsEvaluator(EvaluationConfig(num_folds=1, original_num_bins=5)).evaluate(
            dummy_test_preds[0], pair, [0, 1]
        )
        scaled = ErrorsEvaluator(EvaluationConfig(num_folds=1, original_num_bins=5, error_scaling_factor=2.0)).evaluate(
            dummy_test_preds[0], pair, [0, 1]
        )

        base_bins = base["all_mean_error_bins_nosep"]["U-NET S-MHA"]
        scaled_bins = scaled["all_mean_error_bins_nosep"]["U-NET S-MHA"]

        for base_fold_means, scaled_fold_means in zip(base_bins, scaled_bins):
            for base_value, scaled_value in zip(base_fold_means, scaled_fold_means):
                if base_value is not None and scaled_value is not None:
                    assert scaled_value == pytest.approx(base_value * 2.0)
