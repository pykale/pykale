from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from kale.evaluate.similarity_metrics import _quantile_thresholds, CorrelationConfig, evaluate_correlations

UNCERTAINTY_PAIRS = [("S-MHA", "S-MHA Error", "S-MHA Uncertainty")]
NO_INVERSION = [("S-MHA", False)]


@pytest.fixture
def bin_predictions():
    """Two testing folds of predictions for a single model."""
    return {
        "U-NET": pd.DataFrame(
            {
                "Testing Fold": [0, 0, 0, 0, 1, 1, 1, 1],
                "S-MHA Error": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "S-MHA Uncertainty": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            }
        )
    }


class TestCorrelationConfig:
    """Configuration defaults for evaluate_correlations (issue #555)."""

    def test_defaults(self):
        config = CorrelationConfig()

        assert config.num_bins == 10
        assert config.num_folds == 8
        assert config.error_scaling_factor == 1.0
        assert config.combine_middle_bins is False
        assert config.colormap == "Set1"
        assert config.save_path is None
        assert config.to_log is False


class TestQuantileThresholds:
    """Threshold derivation, previously inlined in evaluate_correlations."""

    def test_returns_one_threshold_fewer_than_bins(self):
        values = np.arange(100, dtype=float)

        thresholds = _quantile_thresholds(values, num_bins=4, combine_middle_bins=False)

        assert len(thresholds) == 3
        assert thresholds == sorted(thresholds)

    def test_combine_middle_bins_keeps_only_outer_thresholds(self):
        values = np.arange(100, dtype=float)

        full = _quantile_thresholds(values, num_bins=5, combine_middle_bins=False)
        combined = _quantile_thresholds(values, num_bins=5, combine_middle_bins=True)

        # The middle bins merge into one, leaving the two edge bins distinct.
        assert combined == [full[0], full[-1]]


class TestEvaluateCorrelations:
    """evaluate_correlations orchestration (issue #555, and the coverage gap noted in #410).

    The plotting call is mocked: saving figures is what makes this function awkward to test in CI,
    and the behaviour under test here is the data selection and configuration handling.
    """

    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_returns_stats_keyed_by_model_and_uncertainty(self, mock_analyze, bin_predictions):
        mock_analyze.return_value = {"all_folds": {"r": 0.5}}

        result = evaluate_correlations(bin_predictions, UNCERTAINTY_PAIRS, NO_INVERSION, CorrelationConfig(num_bins=4))

        assert result == {"U-NET": {"S-MHA": {"all_folds": {"r": 0.5}}}}

    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_only_requested_folds_are_used(self, mock_analyze, bin_predictions):
        """num_folds selects the testing folds, so fold 1 is excluded when only one fold is requested."""
        mock_analyze.return_value = {}

        evaluate_correlations(
            bin_predictions, UNCERTAINTY_PAIRS, NO_INVERSION, CorrelationConfig(num_bins=2, num_folds=1)
        )

        errors = mock_analyze.call_args.args[0]
        np.testing.assert_array_equal(errors, [1.0, 2.0, 3.0, 4.0])

    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_config_is_forwarded_to_the_analysis(self, mock_analyze, bin_predictions, tmp_path):
        mock_analyze.return_value = {}
        config = CorrelationConfig(
            num_bins=2, colormap="tab10", error_scaling_factor=2.5, save_path=str(tmp_path), to_log=True
        )

        evaluate_correlations(bin_predictions, UNCERTAINTY_PAIRS, NO_INVERSION, config)

        kwargs = mock_analyze.call_args.kwargs
        assert kwargs["colormap"] == "tab10"
        assert kwargs["error_scaling_factor"] == 2.5
        assert kwargs["to_log"] is True
        assert kwargs["save_path"].endswith("U-NET_S-MHA_correlation_pwr_all_targets.pdf")

    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_no_save_path_means_no_figure_path(self, mock_analyze, bin_predictions):
        mock_analyze.return_value = {}

        evaluate_correlations(bin_predictions, UNCERTAINTY_PAIRS, NO_INVERSION, CorrelationConfig(num_bins=2))

        assert mock_analyze.call_args.kwargs["save_path"] is None

    @patch("kale.evaluate.similarity_metrics.apply_confidence_inversion")
    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_uncertainty_inversion_is_applied_when_requested(self, mock_analyze, mock_invert, bin_predictions):
        mock_analyze.return_value = {}
        mock_invert.side_effect = lambda frame, key: frame

        evaluate_correlations(bin_predictions, UNCERTAINTY_PAIRS, [("S-MHA", True)], CorrelationConfig(num_bins=2))

        mock_invert.assert_called_once()
        assert mock_invert.call_args.args[1] == "S-MHA Uncertainty"

    @patch("kale.evaluate.similarity_metrics.apply_confidence_inversion")
    @patch("kale.evaluate.similarity_metrics.analyze_and_plot_uncertainty_correlation")
    def test_uncertainty_inversion_is_skipped_when_not_requested(self, mock_analyze, mock_invert, bin_predictions):
        mock_analyze.return_value = {}

        evaluate_correlations(bin_predictions, UNCERTAINTY_PAIRS, NO_INVERSION, CorrelationConfig(num_bins=2))

        mock_invert.assert_not_called()
