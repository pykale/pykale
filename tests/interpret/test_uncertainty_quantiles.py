import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kale.interpret.box_plot import (
    BoxPlotData,
    create_boxplot_config,
    create_boxplot_data,
)
from kale.interpret.uncertainty_quantiles import (
    plot_comparing_q_boxplot,
    plot_cumulative,
    plot_generic_boxplot,
    plot_per_model_boxplot,
    quantile_binning_and_est_errors,
)
from kale.loaddata.tabular_access import load_csv_columns

# from kale.utils.download import download_file_by_url
from kale.utils.seed import set_seed

# import os
LOGGER = logging.getLogger(__name__)


seed = 36
set_seed(seed)

ERRORS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
UNCERTAINTIES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


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

    def test_invalid_type(self):
        """Test error handling for invalid type parameter."""
        with pytest.raises(ValueError, match=r"type must be one of"):
            quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=5, type="invalid")

    def test_different_bin_counts(self):
        """Test function with different bin counts."""
        for num_bins in [3, 5, 8]:
            est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=num_bins)
            # Should have num_bins - 1 boundaries
            assert len(est_bounds) == num_bins - 1
            assert len(est_errors) == num_bins - 1

    def test_combine_middle_bins(self):
        """Test combine_middle_bins functionality."""
        est_bounds, est_errors = quantile_binning_and_est_errors(
            ERRORS, UNCERTAINTIES, num_bins=10, combine_middle_bins=True
        )
        # Should have only 2 bins when combining middle bins
        assert len(est_bounds) == 2
        assert len(est_errors) == 2

    def test_error_wise_type_handling(self):
        """Test that error_wise type is handled properly."""
        # error-wise type is implemented and should return valid results
        est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=5, type="error-wise")
        assert isinstance(est_bounds, list)
        assert isinstance(est_errors, list)


# Additional test fixtures for plot testing
@pytest.fixture
def sample_boxplot_data():
    """Create sample data for boxplot testing."""
    evaluation_data = {
        "ResNet50_epistemic": [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]],
        "VGG16_epistemic": [[0.12, 0.22, 0.32], [0.17, 0.27, 0.37], [0.22, 0.32, 0.42]],
        "ResNet50_aleatoric": [[0.05, 0.15, 0.25], [0.1, 0.2, 0.3], [0.15, 0.25, 0.35]],
        "VGG16_aleatoric": [[0.07, 0.17, 0.27], [0.12, 0.22, 0.32], [0.17, 0.27, 0.37]],
    }
    return create_boxplot_data(
        evaluation_data_by_bins=[evaluation_data],
        uncertainty_categories=[["epistemic"], ["aleatoric"]],
        models=["ResNet50", "VGG16"],
        category_labels=["B1", "B2", "B3"],
        num_bins=3,
    )


@pytest.fixture
def sample_boxplot_config():
    """Create sample config for boxplot testing."""
    return create_boxplot_config(
        x_label="Test Bins",
        y_label="Test Error (%)",
        show=False,
        save_path=None,
    )


@pytest.fixture
def sample_comparing_q_data():
    """Create sample data for Q-value comparison testing."""
    q_data = [
        {"model_epistemic": [[0.1, 0.2], [0.15, 0.25]]},  # Q=5
        {"model_epistemic": [[0.09, 0.19], [0.14, 0.24]]},  # Q=10
        {"model_epistemic": [[0.08, 0.18], [0.13, 0.23]]},  # Q=15
    ]
    return create_boxplot_data(
        evaluation_data_by_bins=q_data,
        uncertainty_categories=[["epistemic"]],
        models=["model"],
        category_labels=["Q=5", "Q=10", "Q=15"],
        num_bins=2,
    )


class TestPlotFunctions:
    """Test the plotting wrapper functions."""

    @patch("kale.interpret.box_plot.GenericBoxPlotter.draw_boxplot")
    def test_plot_generic_boxplot(self, mock_draw, sample_boxplot_data, sample_boxplot_config):
        """Test plot_generic_boxplot function."""
        plot_generic_boxplot(sample_boxplot_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    @patch("kale.interpret.box_plot.PerModelBoxPlotter.draw_boxplot")
    def test_plot_per_model_boxplot(self, mock_draw, sample_boxplot_data, sample_boxplot_config):
        """Test plot_per_model_boxplot function."""
        plot_per_model_boxplot(sample_boxplot_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    @patch("kale.interpret.box_plot.ComparingQBoxPlotter.draw_boxplot")
    def test_plot_comparing_q_boxplot(self, mock_draw, sample_comparing_q_data, sample_boxplot_config):
        """Test plot_comparing_q_boxplot function."""
        plot_comparing_q_boxplot(sample_comparing_q_data, sample_boxplot_config)
        mock_draw.assert_called_once()

    def test_plot_comparing_q_boxplot_missing_data(self, sample_boxplot_config):
        """Test that plot_comparing_q_boxplot raises error with missing data."""
        incomplete_data = BoxPlotData(evaluation_data_by_bins=[{}])
        with pytest.raises(ValueError, match="For comparing_q plots"):
            plot_comparing_q_boxplot(incomplete_data, sample_boxplot_config)

    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_plot_cumulative_basic(self, mock_close, mock_show, mock_savefig, mock_gca, mock_figure, mock_style):
        """Test basic cumulative plot functionality."""
        # Mock data structure with correct column names
        pd = pytest.importorskip("pandas")
        mock_df = pd.DataFrame(
            {
                "error": [1, 2, 3, 4, 5],
                "uncertainty": [0.1, 0.2, 0.3, 0.4, 0.5],
                "epistemic Uncertainty bins": [1, 1, 2, 2, 2],
                "epistemic Error": [1, 2, 3, 4, 5],
            }
        )
        data_struct = {"ResNet50": mock_df}

        # Mock axis object
        mock_ax = MagicMock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles and labels
        mock_gca.return_value = mock_ax

        plot_cumulative(
            colormap="Set1",
            data_struct=data_struct,
            models=["ResNet50"],
            uncertainty_types=[("epistemic", "")],
            bins=[1, 2],
            title="Test Plot",
            save_path=None,
        )

        mock_style.assert_called_once_with("ggplot")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.style.use")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_cumulative_with_save(self, mock_close, mock_savefig, mock_gca, mock_figure, mock_style, tmp_path):
        """Test cumulative plot with save functionality."""
        # Mock data structure with correct column names
        pd = pytest.importorskip("pandas")
        mock_df = pd.DataFrame(
            {
                "error": [1, 2, 3, 4, 5],
                "uncertainty": [0.1, 0.2, 0.3, 0.4, 0.5],
                "epistemic Uncertainty bins": [1, 1, 2, 2, 2],
                "epistemic Error": [1, 2, 3, 4, 5],
            }
        )
        data_struct = {"ResNet50": mock_df}

        # Mock axis object
        mock_ax = MagicMock()
        mock_ax.get_legend_handles_labels.return_value = ([], [])  # Return empty handles and labels
        mock_gca.return_value = mock_ax

        save_path = str(tmp_path)
        plot_cumulative(
            colormap="Set1",
            data_struct=data_struct,
            models=["ResNet50"],
            uncertainty_types=[("epistemic", "")],
            bins=[1, 2],
            title="Test Plot",
            save_path=save_path,
        )

        mock_savefig.assert_called_once()
        mock_close.assert_called()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_quantile_binning_single_bin(self):
        """Test quantile binning with single bin (edge case)."""
        # Single bin should result in no boundaries
        est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=1)
        assert len(est_bounds) == 0
        assert len(est_errors) == 0

    def test_quantile_binning_two_bins(self):
        """Test quantile binning with two bins."""
        est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=2)
        assert len(est_bounds) == 1
        assert len(est_errors) == 1

    def test_quantile_binning_identical_values(self):
        """Test quantile binning with identical uncertainty values."""
        identical_uncertainties = [0.5] * 10
        varied_errors = list(range(10))

        est_bounds, est_errors = quantile_binning_and_est_errors(varied_errors, identical_uncertainties, num_bins=5)
        # Should handle identical values gracefully
        assert len(est_bounds) == 4
        assert len(est_errors) == 4

    def test_quantile_binning_reverse_correlation(self):
        """Test quantile binning with reverse correlation (decreasing errors)."""
        reverse_errors = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        est_bounds, est_errors = quantile_binning_and_est_errors(reverse_errors, UNCERTAINTIES, num_bins=5)
        # Should handle reverse correlation
        assert len(est_bounds) == 4
        assert len(est_errors) == 4


class TestParameterValidation:
    """Test parameter validation and type checking."""

    def test_quantile_binning_non_numeric_inputs(self):
        """Test error handling for non-numeric inputs."""
        with pytest.raises((TypeError, ValueError)):
            quantile_binning_and_est_errors(["a", "b", "c"], [1, 2, 3], num_bins=2)

    def test_quantile_binning_negative_bins(self):
        """Test behavior with negative bin count."""
        # Negative bins should be handled gracefully
        est_bounds, est_errors = quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=-1)
        # Should return reasonable results even with negative input
        assert isinstance(est_bounds, list)
        assert isinstance(est_errors, list)

    def test_quantile_binning_zero_bins(self):
        """Test error handling for zero bin count."""
        # Zero bins should raise a ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            quantile_binning_and_est_errors(ERRORS, UNCERTAINTIES, num_bins=0)
