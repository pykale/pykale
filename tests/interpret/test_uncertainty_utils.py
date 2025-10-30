from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kale.interpret.uncertainty_utils import (
    analyze_and_plot_uncertainty_correlation,
    plot_cumulative,
    quantile_binning_and_est_errors,
)

SEED = 42
np.random.seed(SEED)

ERRORS = np.random.rand(50)
UNCERTAINTIES = np.random.rand(50)
QUANTILE_THRESHOLDS = [0.3, 0.6]


@pytest.fixture(scope="module")
def dummy_corr_data():
    # Fixture to provide dummy errors and uncertainties
    errors = np.random.rand(30)
    uncertainties = np.random.rand(30)
    quantile_thresholds = [0.2, 0.5, 0.8]
    return errors, uncertainties, quantile_thresholds


class TestBasicCorrelation:
    """Test basic correlation analysis functionality."""

    def test_basic_functionality(self, tmp_path):
        """Test basic correlation analysis with default parameters."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        result = analyze_and_plot_uncertainty_correlation(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
            to_log=False,
            error_scaling_factor=1.0,
            save_path=out_path,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result
        assert len(result["spearman"]) == 2
        assert len(result["pearson"]) == 2
        assert all(isinstance(x, float) for x in result["spearman"])
        assert all(isinstance(x, float) for x in result["pearson"])

    def test_with_logging(self, dummy_corr_data, tmp_path):
        """Test correlation analysis with logarithmic scaling."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        errors, uncertainties, quantile_thresholds = dummy_corr_data
        result = analyze_and_plot_uncertainty_correlation(
            errors=errors,
            uncertainties=uncertainties,
            quantile_thresholds=quantile_thresholds,
            to_log=True,
            error_scaling_factor=2.0,
            save_path=out_path,
        )
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result

    def test_save_functionality(self, tmp_path):
        """Test that plots are properly saved and files are created."""
        out_path = tmp_path / "test_correlation_plot.pdf"

        result = analyze_and_plot_uncertainty_correlation(
            errors=ERRORS,
            uncertainties=UNCERTAINTIES,
            quantile_thresholds=QUANTILE_THRESHOLDS,
            to_log=False,
            error_scaling_factor=1.0,
            save_path=out_path,
        )

        # Verify the function returns expected results
        assert isinstance(result, dict)
        assert "spearman" in result
        assert "pearson" in result

        # Verify the file was actually created
        assert out_path.exists(), f"Expected plot file was not created at {out_path}"

        # Verify the file is not empty
        assert out_path.stat().st_size > 0, "Created plot file is empty"

        # Verify reasonable file size (PDF should be substantial but not huge)
        file_size = out_path.stat().st_size
        assert file_size > 1000, f"Plot file seems too small ({file_size} bytes)"
        assert file_size < 10_000_000, f"Plot file seems too large ({file_size} bytes)"

        # tmp_path automatically cleans up after test completes


class TestCorrelationInputValidation:
    """Test input validation and error handling."""

    def test_mismatched_array_lengths(self, tmp_path):
        """Test that mismatched array lengths raise appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        with pytest.raises(ValueError, match="must have the same length"):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([1, 2, 3]),
                uncertainties=np.array([1, 2]),
                quantile_thresholds=[0.5],
                save_path=out_path,
            )

    def test_empty_arrays(self, tmp_path):
        """Test that empty arrays raise appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        with pytest.raises(ValueError, match="cannot be empty"):
            analyze_and_plot_uncertainty_correlation(
                errors=np.array([]), uncertainties=np.array([]), quantile_thresholds=[0.5], save_path=out_path
            )

    def test_negative_quantile_thresholds(self, tmp_path):
        """Test that negative quantile thresholds cause appropriate errors."""
        out_path = tmp_path / "test_correlation_plot.pdf"
        errors = np.array([-1, -2, -3])
        uncertainties = np.array([-1, -2, -3])
        quantile_thresholds = [-0.5, -0.2]
        # Negative quantile thresholds are likely to cause issues in piecewise fitting
        with pytest.raises((ValueError, RuntimeError, Exception)):
            analyze_and_plot_uncertainty_correlation(errors, uncertainties, quantile_thresholds, save_path=out_path)


class TestPlotFunctions:
    """Test the plotting wrapper functions."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.style.use")
    def test_plot_cumulative_basic(self, mock_style, mock_gca, mock_show):
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

    @patch("matplotlib.pyplot.close")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.gca")
    def test_plot_cumulative_with_save(self, mock_gca, mock_savefig, mock_close, tmp_path):
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
        # Create specific sorted test data with 11 points for 10 bins
        test_uncertainties = np.linspace(0, 1, 11)
        test_errors = np.linspace(0, 10, 11)

        est_bounds, est_errors = quantile_binning_and_est_errors(test_errors, test_uncertainties, num_bins=10)

        # Should have 9 boundaries (num_bins - 1)
        assert len(est_bounds) == 9
        assert len(est_errors) == 9
        # Boundaries should be approximately at the quantile positions (excluding first and last)
        assert pytest.approx(np.squeeze(est_bounds), abs=0.1) == test_uncertainties[1:-1]
        assert pytest.approx(np.squeeze(est_errors), abs=0.1) == test_errors[1:-1]

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
        # Create matching uncertainties array with same length
        reverse_uncertainties = np.linspace(0, 1, 11)

        est_bounds, est_errors = quantile_binning_and_est_errors(reverse_errors, reverse_uncertainties, num_bins=5)
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
